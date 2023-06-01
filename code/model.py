import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# source of the code: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, heads, dropout):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_en = PositionalEncoding(hidden_size, max_len=embed_size)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # permuting captions to form [batch size, caption size]
        cpt_per = torch.permute(captions, (1,0)).to(device)
        emb = self.embed(cpt_per)
        pos_emb = self.pos_en(emb)
        # permuting to get batch first
        pos_emb = torch.permute(pos_emb, (1,0,2)).to(device)
        # applying subsequent mask to avoid looking forward
        mask = nn.Transformer.generate_square_subsequent_mask(pos_emb.size(1), device=device)
        dec_out = self.decoder(tgt=pos_emb, memory=features, tgt_mask=mask)
        outputs = self.linear(dec_out)
        return outputs

class EncoderVIT(nn.Module):
    def __init__(self, embed_size):
        super(EncoderVIT, self).__init__()
        # loading pre-trained weights
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.fc = nn.Linear(self.vit.num_classes, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.fc(self.vit(images))
        return self.dropout(self.relu(features)).view(features.size(0), 1, features.size(1))
    



class VITtoDT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, heads, dropout):
        super(VITtoDT, self).__init__()
        self.encoder = EncoderVIT(embed_size)
        self.decoder = DecoderTransformer(embed_size, hidden_size, vocab_size, num_layers, heads, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = None

        with torch.no_grad():
            img_enc = self.encoder(image.unsqueeze(0))
            sos_id = vocabulary.stoi["<SOS>"]
            sentence = torch.tensor([sos_id]).unsqueeze(0).to(device)

            for _ in range(max_length):
                output = self.decoder(img_enc, sentence)
                # getting last word in output as a predicted one
                predicted = output[:,-1,:].argmax(1).unsqueeze(0).to(device)
                sentence = torch.cat((sentence,predicted), 1).to(device)
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
            result_caption = sentence

        return [vocabulary.itos[idx.item()] for idx in result_caption.flatten() if idx not in  [vocabulary.stoi["<SOS>"], vocabulary.stoi["<EOS>"], vocabulary.stoi["."]]]
    
