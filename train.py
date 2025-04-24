import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.utils.data as data
import os
from models.models import Decoder
from data.dataset import CocoHDF5

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    n_epochs = 25
    batch_size = 128
    
    vae_final_fn = 'models/decoderv3.pt'
    vae_ckpt_fn = vae_final_fn + 'h'

    ds     = CocoHDF5("coco_clip.h5")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)


    decoder = Decoder(input_dim=512, hidden_dim=1024, gamma=0.05).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    start_epoch = 0
    if os.path.exists(vae_ckpt_fn):
        print(f"Found checkpoint {vae_ckpt_fn}. Resuming training.")
        checkpoint = torch.load(vae_ckpt_fn)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, n_epochs):
        
        epoch_loss = 0
        count = 0
        for imgs, captions, _ in tqdm(loader):
            # images should be scaled already in preprocess_coco.py
            # unless noted otherwise
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

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.8f}")

        if (epoch + 1) % 1 == 0:
            latest_checkpoint = {
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }
            torch.save(latest_checkpoint, vae_ckpt_fn)
            print(f"Latest checkpoint saved as {vae_ckpt_fn}")

        if (epoch + 1) % 5 == 0:
            latest_checkpoint = {
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }
            vae_save_state_fn = "models/decoderv3_" + str((epoch+1)) + ".pth"
            torch.save(latest_checkpoint, vae_save_state_fn)
            print(f"Latest 5 checkpoint saved as {vae_save_state_fn}")

    torch.save(decoder.state_dict(), vae_final_fn)
    print(f"Final model saved: {vae_final_fn}")