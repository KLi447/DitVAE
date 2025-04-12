import torch
import clip
from torchvision import datasets as dset
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        assert(len(data) == len(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx].float(), self.labels[idx].float())

if __name__ == "__main__":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available() and torch.backends.cuda.is_built():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    cap = dset.CocoCaptions(root = './data/coco/images/train2017',
                            annFile = './data/coco/annotations/captions_train2017.json',
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)), 
                                transforms.ToTensor()
    ]))

    MAX_SAMPLES = 30000
    data_tensor = torch.empty((MAX_SAMPLES, 3, 256, 256), dtype=torch.float32)
    labels_tensor = torch.zeros((MAX_SAMPLES, 5, 512), dtype=torch.float32)
    count = 0
    MAX_LABELS = 5

    for img, captions in cap:
        if count == MAX_SAMPLES:
            break
        img = img.to(device)
        embs = []
        for text in captions[:MAX_LABELS]:
            tokens = clip.tokenize([text]).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_text(tokens)
            embs.append(embedding)
        
        for j, emb in enumerate(embs):
            labels_tensor[count, j] = emb.cpu()

        data_tensor[count] = img.cpu()
        if count % 1000 == 0:
            print(count)
        count += 1


    dataset = MyDataset(data_tensor, labels_tensor)
    checkpoint = {'dataset': dataset}
    torch.save(checkpoint, '/projects/beis/kli44/DitVAE/data/train30k_clip_dataset_checkpoint.pth')
