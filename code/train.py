import os 
import json
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples, CustomScheduler
from get_loader import get_loader, build_vocab
from get_loader import transform_data as transform
from model import VITtoDT
from evaluate import evaluate_dataset_new

# building vocabulary
vocab = build_vocab("../flickr8k_split/captions.txt", freq_threshold = 5)

# switches
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True
train_encoder = False

# Hyperparameters
batch_size = 128
embed_size = 512
hidden_size = 512
vocab_size = len(vocab)
num_layers = 4
learning_rate = 1e-4
num_epochs = 71
heads = 8
dropout = 0.5

# checkpoints
run_name = 'vit_to_dt_ft5'
output_dir = os.path.join('checkpoints', run_name)
os.makedirs(output_dir , exist_ok= True)
step = 0

# dataloaders
train_loader, dataset = get_loader(
    root_folder="../flickr8k_split/train_images",
    annotation_file="../flickr8k_split/train_captions.txt",
    transform=transform,
    vocab = vocab, # None
    num_workers=8,
    batch_size = batch_size
)


val_loader, _ = get_loader(
    root_folder="../flickr8k_split/val_images",
    annotation_file="../flickr8k_split/val_captions.txt",
    transform=transform,
    vocab = vocab, #dataset.vocab,
    num_workers=1,
    batch_size = 5,
    shuffle = False,
) 

train_loader_eval, _ = get_loader(
    root_folder="../flickr8k_split/train_images",
    annotation_file="../flickr8k_split/train_captions.txt",
    transform=transform,
    vocab = vocab, # None
    num_workers=1,
    batch_size = 5, # each image has 5 different GT captions, for each batch we will load only one image and corresponding captions
    shuffle = False,
)

# defining model
model = VITtoDT(embed_size, hidden_size, vocab_size, num_layers, heads, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"], label_smoothing=0.1) # it will ignore 'PAD' words 
optimizer = optim.Adam(model.parameters(), betas = (0.9, 0.98), eps=1.0e-9)
scheduler = CustomScheduler(optimizer, embed_size, 4000)

# fine-tuning
for name, param in model.encoder.named_parameters():
    if "encoder.ln" in name or "heads.head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = train_encoder

# loading checkpoint
if load_model:
    step = load_checkpoint(torch.load('./checkpoints/vit_to_dt_4/checkpoints_40.pth.tar'), model, optimizer)

# training loop
model.train()
for epoch in range(num_epochs):
    # printing test examples
    if epoch % 5 == 0: # modify for your case
        print_examples(model, device, dataset)
    
    if epoch % 5 ==0: 
        # calculate BLUE score on the validation set
        blue_score_val = evaluate_dataset_new(model, val_loader, vocab, device)
        print('BLUE SCORES validation ', epoch, blue_score_val)
        # calculate BLUE score on the training set
        blue_score_train = evaluate_dataset_new(model, train_loader_eval, vocab, device)
        print('BLUE SCORES TRAIN', epoch, blue_score_train)
        
        # logging
        log_stats = {'BLUE-1-VAL': blue_score_val,
                     'BLUE-1-TRAIN': blue_score_train,
                    'epoch': epoch}
        f = open(os.path.join(output_dir, "log_blue.txt"), "a+")
        f.write(json.dumps(log_stats) + "\n")
        f.close()

    # saving checkpoint
    if save_model and epoch % 5 == 0 and epoch > 0:
      checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
		        "lr": scheduler.optimizer.param_groups[0]['lr'],
      }
      save_checkpoint(checkpoint, filename = os.path.join(output_dir, 'checkpoints_'+str(epoch)+'.pth.tar'))
    
    for idx, (imgs, captions) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=False
    ):

        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, torch.permute(captions[:-1], (1,0)))
        loss = criterion(
            torch.permute(outputs, (0,2,1)), torch.permute(captions[1:], (1,0))
        )
        step += 1

        optimizer.zero_grad()
        loss.backward(loss)
        # clipping gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.)
        optimizer.step()

        scheduler.step()
        # logging
        log_stats = {'loss': loss.item(),
                    'epoch': epoch,
                    'step': step, 'lr':scheduler.optimizer.param_groups[0]['lr']}
        f = open(os.path.join(output_dir, "log_loss.txt"), "a+")
        f.write(json.dumps(log_stats) + "\n")
        f.close()
    print("Training loss", loss.item(), epoch)
