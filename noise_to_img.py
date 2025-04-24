import os
import time
import clip
import lpips

from data.dataset import CocoHDF5
from models.models import UNetGenerator, UNetGenerator, WrappedModel

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    # Hyperparameters
    lr = 0.0001
    batch_size = 128
    iterations = 100000
    perceptual_weight = 0.1
    print_every = 1
    save_every = 1000

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad_(False)

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    ds     = CocoHDF5("coco_clip.h5")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    final_path = "models/noise_to_img.pt"
    checkpoint_path = final_path + "h"

    # --- Model, Path, and Optimizer ---
    model = UNetGenerator(in_channels=3).to(device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Optional Checkpoint Loading ---
    start_iter = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iteration', 0)
        print(f"Resuming from iteration {start_iter}")

    # --- Training Loop ---
    model.train()
    data_iter = iter(loader)
    start_time = time.time()
    for i in range(start_iter, iterations):
        try:
            images, text_emb, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, text_emb, _ = next(data_iter)
        
        images = images.to(device)  # Target images: [B, 3, 256, 256]
        text_emb = text_emb.to(device)
        # Generate Gaussian noise with the same shape as images.
        noise = torch.randn_like(images).to(device)
        # Sample a random time for each sample (in [0,1]).
        t = torch.rand(images.size(0), device=device)
        
        # Generate an interpolated sample along the path.
        path_sample = path.sample(t=t, x_0=noise, x_1=images)
        # Compute the loss: mean squared error between the velocity field and the target dx_t.
        pred = model(path_sample.x_t, path_sample.t, text_emb)
        loss = ((pred - path_sample.dx_t) ** 2).mean()

        imgs_lpips = images * 2.0 - 1.0
        pred_lpips = torch.clamp(pred, 0, 1) * 2.0 - 1.0
        perceptual_loss = loss_fn_vgg(pred_lpips, imgs_lpips).mean()
        loss += perceptual_weight * perceptual_loss # perceptual loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(f'| iter {i+1:6d} | {elapsed*1000/print_every:5.2f} ms/step | loss {loss.item():8.3f}')
            start_time = time.time()
        if (i + 1) % save_every == 0:
            # Save checkpoint.
            torch.save({
                'iteration': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

    torch.save(model.state_dict(), final_path)

if __name__ == "__main__":
    main()
