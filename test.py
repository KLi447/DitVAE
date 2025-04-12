from preprocess_coco import MyDataset
import numpy as np
import torch
from PIL import Image

loaded_checkpoint = torch.load('train10k_dataset_checkpoint.pth', weights_only=False)
ld = loaded_checkpoint['dataset']

dummy_loader = torch.utils.data.DataLoader(ld, batch_size=1, shuffle=True)

for _, (img, emb) in enumerate(dummy_loader):
    if img.ndim == 4:
        img = img.squeeze(0)

    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    img = img.byte()

    image = Image.fromarray(img.numpy())
    image.save("img.jpg")
    break
