import h5py
import torch
import clip
from torchvision import datasets as dset, transforms
from tqdm.auto import tqdm
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    for p in clip_model.parameters(): p.requires_grad_(False)

    coco = dset.CocoCaptions(
        root    = "./data/coco/images/train2017",
        annFile = "./data/coco/annotations/captions_train2017.json",
        transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
    )

    N = len(coco)
    MAX_LABELS = 5
    H, W = 256, 256
    TOKEN_LEN = 77 

    with h5py.File("coco_clip.h5", "w") as f:
        img_ds = f.create_dataset(
            "images",
            shape=(N, 3, H, W),
            dtype="f4",
            chunks=(1,3,H,W),
        )
        emb_ds = f.create_dataset(
            "clip_embs",
            shape=(N, MAX_LABELS, 512),
            dtype="f4",
            chunks=(1,MAX_LABELS,512),
        )
        tok_ds    = f.create_dataset(
            "clip_tokens",
            shape=(N, MAX_LABELS, TOKEN_LEN),
            dtype="i8",
            chunks=(1,MAX_LABELS,TOKEN_LEN),
        )

        for idx, (img, captions) in enumerate(tqdm(coco, total=N)):
            img_ds[idx, ...] = img.numpy()

            embs   = np.zeros((MAX_LABELS, 512), dtype="f4")
            toks   = np.zeros((MAX_LABELS, TOKEN_LEN), dtype="i8")

            for j, text in enumerate(captions[:MAX_LABELS]):
                tok = clip.tokenize([text]).to(device)
                with torch.no_grad():
                    emb = clip_model.encode_text(tok)
                embs[j] = emb.cpu().numpy().reshape(512,)
                toks[j] = tok.cpu().numpy().reshape(TOKEN_LEN,)

            emb_ds[idx, ...] = embs
            tok_ds[idx, ...] = toks

            if idx % 1000 == 0:
                f.flush()
