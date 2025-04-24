import torch
from models.models import Decoder
import time
import torchvision.utils as vutils
import os
from PIL import Image
from data.dataset import CocoHDF5
from torch.utils.data import DataLoader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds     = CocoHDF5("coco_clip.h5")
    loader = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True)

    _, text_emb, _ = ds[118286]
    
    d_model = Decoder(512, 1024)
    state_dict = torch.load('models/decoderv3.pt', weights_only=True)
    d_model.load_state_dict(state_dict)
    d_model = torch.compile(d_model)
    d_model.to(device)
    d_model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        x_hat = d_model.sample(text_emb, device=device)

    x_hat = (x_hat * 255).clamp(0, 255).byte()
    grid = vutils.make_grid(x_hat, nrow=4, padding=2)
    img = grid.permute(1, 2, 0).cpu().numpy()

    os.makedirs("./samples", exist_ok=True)
    Image.fromarray(img).save(os.path.join("./samples", "sampled_from_checkpoint.jpg"))

    end_time = time.time()

    inference_time = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time:.3f} ms")
    
