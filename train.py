
from tqdm.auto import tqdm
import torch.optim as optim
from preprocess_coco import MyDataset

import numpy as np
import torch
import os
from models import Decoder

def train_vae(model, train_loader, n_epochs=20, lr=0.01, device="cpu"):
    train_op = optim.Adam(model.parameters(), lr)
    model.to(device)

    model.train()
    for epoch in tqdm(range(n_epochs)):
        loss_ = []
        for _, (img, emb) in enumerate(train_loader):
            # pick one caption at random
            # FIXME -> slices one row for whole batch, maybe fix?

            # or run for ALL embeddings, not just a random one

            # idx = np.random.randint(0, emb.shape[1])
            # embedding = emb[:, idx:idx+1, :, :]
            embedding = emb[:, 0:1, :, :]

            img, embedding = img.to(device), embedding.to(device)
            img /= 255 #scale down to [0, 1]
            loss = model.get_loss(embedding, img)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()
        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
    return model

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    #tbd
    batch_size = 19 # factor 82783
    n_epochs=40
    lr=0.005
    vae_ckpt_fn = 'model_vae.pt'

    loaded_checkpoint = torch.load('dataset_checkpoint.pth', weights_only=False)
    ld = loaded_checkpoint['dataset']

    train_loader = torch.utils.data.DataLoader(ld, batch_size=batch_size, shuffle=True)

    if not os.path.exists(vae_ckpt_fn):
        model_vae = Decoder(512, 1024)
        model_vae = train_vae(model_vae, train_loader, n_epochs=n_epochs, lr=lr, device=device)

        torch.save(model_vae.state_dict(), vae_ckpt_fn)
