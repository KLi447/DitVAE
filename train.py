
import torch
import clip
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CocoCaptions
from tqdm.auto import tqdm
import torch.utils.data as data
from PIL import Image
import os
from models.models import Decoder
from preprocess_coco import MyDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision.utils import save_image

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    vae_final_fn = 'models/decoder.pt'
    vae_ckpt_fn = vae_final_fn + 'h'

    train_checkpoint = torch.load('data/train30k_0_1_clip_dataset_checkpoint.pth', weights_only=False)
    train_data = train_checkpoint['dataset']

    val_checkpoint = torch.load('data/val5k_0_1_clip_dataset_checkpoint.pth', weights_only=False)
    val_dataset = val_checkpoint['dataset']
    # choose subset of 1000 to run validation
    val_subset = Subset(val_dataset, indices=list(range(1000)))

    batch_size = 64
    n_epochs = 1000

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    decoder = Decoder(input_dim=512, hidden_dim=1024, gamma=0.05).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    warmup_epochs = 20
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_epochs)

    cosine_epochs = n_epochs - warmup_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)

    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

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

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    decoder.train()
    for epoch in range(start_epoch, n_epochs):
        
        epoch_loss = 0
        count = 0
        for imgs, captions in tqdm(train_loader):
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
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.8f}")

        decoder.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs = imgs.to(device)
                captions = captions.to(device)
                B, N, D = captions.shape

                captions_flat = captions.view(B * N, D)
                preds = decoder(captions_flat)
                imgs_repeated = imgs.unsqueeze(1).expand(B, N, *imgs.shape[1:]).reshape(B * N, *imgs.shape[1:])
                loss = criterion(preds, imgs_repeated)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"Epoch {epoch+1}/{n_epochs} Validation Loss: {avg_val_loss:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_checkpoint = {
                'epoch': epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss
            }
            torch.save(best_checkpoint, vae_final_fn)
            print(f"Best model updated and saved as {vae_final_fn}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
            break

        if (epoch + 1) % 25 == 0:
            decoder.eval()
            with torch.no_grad():
                sample_imgs, sample_captions = next(iter(train_loader))

                sample_captions = sample_captions.to(device)
                B_s, N_s, D_s = sample_captions.shape

                sample_captions_flat = sample_captions.view(B_s * N_s, D_s)
                sample_preds = decoder(sample_captions_flat)
                sample_filename = f"./samples/sample_epoch_{epoch+1}.png"
                os.makedirs("./samples", exist_ok=True)
                save_image(sample_preds, sample_filename, nrow=N_s)
                print(f"Sample output saved to {sample_filename}")
        
        decoder.train()

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