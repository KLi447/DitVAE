import torch
from models.models_old import Decoder
import torchvision.transforms as transforms
import clip
from torchvision import datasets as dset
from torch.utils.data import Dataset

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

    cap = dset.CocoCaptions(root = './data/coco/images/val2017',
                            annFile = './data/coco/annotations/captions_val2017.json',
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)), 
                                transforms.ToTensor()
    ]))

    d_model = Decoder(512, 1024)
    state_dict = torch.load('./decoderv3_1000.pt', weights_only=True)
    d_model.load_state_dict(state_dict)
    d_model = torch.compile(d_model)
    d_model.to(device)
    d_model.eval()

    MAX_SAMPLES = 5000
    count = 0

    coco_img_tensor = torch.empty((MAX_SAMPLES, 3, 256, 256), dtype=torch.float32)
    decoder_img_tensor = torch.empty((MAX_SAMPLES, 3, 256, 256), dtype=torch.float32)

    for img, captions in cap:
        if count == MAX_SAMPLES:
            break
        # use first caption for easy
        text = captions[0]
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_text(tokens)

        embedding = embedding.to(torch.float32)
        with torch.no_grad():
            decoded = d_model.sample(embedding, device=device)

        coco_img_tensor[count] = img
        decoder_img_tensor[count] = decoded.cpu()
        if count % 1000 == 0:
            print(count)
        count += 1

    dataset = MyDataset(coco_img_tensor, decoder_img_tensor)
    checkpoint = {'dataset': dataset}
    torch.save(checkpoint, '/projects/beis/kli44/DitVAE/data/img_and_decoded_5k.pth')
