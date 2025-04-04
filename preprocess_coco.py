import torchvision.datasets as dset
import torchvision.transforms as transforms
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset
import torch

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
    tokenizer = T5Tokenizer.from_pretrained("t5-small") # or "t5-base", "t5-large", etc.
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    for param in model.parameters():
        param.requires_grad = False

    cap = dset.CocoCaptions(root = './data/coco/train2014/train2014',
                            annFile = './data/coco/annotations_trainval2014/annotations/captions_train2014.json',
                            transform=transforms.Compose([
                                transforms.Resize((256, 256)), 
                                transforms.PILToTensor()
    ]))

    data_tensor = torch.empty((len(cap), 3, 256, 256))
    labels_tensor = torch.zeros((len(cap), 5, 512))
    count = 0
    MAX_LABELS = 5

    for img, target in cap:
        img = img.to(device)
        embs = []
        for text in target[:MAX_LABELS]:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model.encoder(**inputs)
                embeddings = outputs.last_hidden_state
                sentence_embedding = embeddings.mean(dim=1) 
                embs.append(sentence_embedding.squeeze(0))
        for j, emb in enumerate(embs):
            labels_tensor[count, j] = emb.cpu()

        data_tensor[count] = img.cpu()
        if count % 1000 == 0:
            print(count)
        count += 1


    dataset = MyDataset(data_tensor, labels_tensor)
    checkpoint = {'dataset': dataset}
    torch.save(checkpoint, '/u/kli44/DitVAE/dataset_checkpoint.pth')