
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
from models import Decoder
from preprocess_coco import MyDataset

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    # clip_model.eval()

    # cap = CocoCaptions(root = './data/coco/images/train2017',
    #                         annFile = './data/coco/annotations/captions_train2017.json',
    #                         transform=transforms.Compose([
    #                             transforms.Resize((256, 256)), 
    #                             transforms.PILToTensor()
    # ]))
    
    vae_ckpt_fn = 'decoderv3_1000.pt'

    loaded_checkpoint = torch.load('train10k_clip_dataset_checkpoint.pth', weights_only=False)
    ld = loaded_checkpoint['dataset']

    batch_size = 32
    n_epochs = 1000

    train_loader = DataLoader(ld, batch_size=batch_size, shuffle=True)
    decoder = Decoder(input_dim=512, hidden_dim=1024).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    if not os.path.exists(vae_ckpt_fn):
        for epoch in range(n_epochs):
            decoder.train()
            epoch_loss = 0
            print(f"Epoch: {epoch+1}")

            for imgs, captions in tqdm(train_loader):
                imgs /= 255
                imgs = imgs.to(device)
                c = captions[:, 0:1, :]
                c = c.to(device)

                # tokens = clip.tokenize(captions).to(device)
        
                # with torch.no_grad():
                #     embeddings = clip_model.encode_text(tokens)
                
                preds = decoder(c.squeeze())

                loss = criterion(preds, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            scheduler.step()
        
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

        torch.save(decoder.state_dict(), vae_ckpt_fn)
