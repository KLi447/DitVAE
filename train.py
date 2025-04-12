
import torch
import clip
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CocoCaptions
from tqdm.auto import tqdm
import torch.utils.data as data
from PIL import Image
import os
from models.models import Decoder
from preprocess_coco import MyDataset

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    vae_final_fn = 'models/decoderv4_t2_1000.pt'
    vae_ckpt_fn = vae_final_fn + 'h'

    loaded_checkpoint = torch.load('data/train30k_clip_dataset_checkpoint.pth', weights_only=False)
    ld = loaded_checkpoint['dataset']

    batch_size = 64
    n_epochs = 1000

    train_loader = DataLoader(ld, batch_size=batch_size, shuffle=True)
    decoder = Decoder(input_dim=512, hidden_dim=1024).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    start_epoch = 0
    if os.path.exists(vae_ckpt_fn):
        print(f"Found checkpoint {vae_ckpt_fn}. Resuming training.")
        checkpoint = torch.load(vae_ckpt_fn)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    decoder.train()
    for epoch in range(start_epoch, n_epochs):
        
        epoch_loss = 0

        for imgs, captions in tqdm(train_loader):
            imgs = imgs.float() / 255.0  # Scale images from [0, 255] to [0, 1]
            imgs = imgs.to(device)

            captions = captions.to(device)
            B, N, D = captions.shape

            captions_flat = captions.view(B * N, D)
            preds = decoder(captions_flat)
            
            # Replicate each image 5 times to match the caption dimension
            imgs_repeated = imgs.unsqueeze(1).expand(B, N, *imgs.shape[1:]).reshape(B * N, *imgs.shape[1:])

            loss = criterion(preds, imgs_repeated)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.8f}")

        if (epoch + 1) % 5 == 0:
            latest_checkpoint = {
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss
            }
            torch.save(latest_checkpoint, vae_ckpt_fn)
            print(f"Latest checkpoint saved as {vae_ckpt_fn}")

    torch.save(decoder.state_dict(), vae_final_fn)
    print(f"Final model saved: {vae_final_fn}")