from data.dataset import CocoCLIPDataset
from models.models import UNetGenerator, UNetGenerator, WrappedModel

import matplotlib.pyplot as plt

import torch
import clip

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from flow_matching.path import AffineProbPath
from utils.ode_solver import ODESolver
from flow_matching.path.scheduler import CondOTScheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

batch_size = 16
checkpoint_path = "models/noise_to_img.pt"

ds = CocoCLIPDataset(
    root        = "./data/coco/images/train2017",
    annFile     = "./data/coco/annotations/captions_train2017.json",
    clip_model  = clip_model,
    device      = device,
    max_labels  = 5,
)

loader = DataLoader(
    ds,
    batch_size    = batch_size,
    shuffle       = True,
    pin_memory    = True,
)

model = UNetGenerator(in_channels=3).to(device)
wrapped_model = WrappedModel(model)
path = AffineProbPath(scheduler=CondOTScheduler())

# ???
# checkpoint = torch.load(checkpoint_path, map_location=device)
# state_dict = checkpoint["model_state_dict"]

# if list(state_dict.keys())[0].startswith("module."):
#     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# model.load_state_dict(state_dict, strict=True)

state_dict = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(state_dict)
model = torch.compile(model)

data_iter = iter(loader)
_, text_emb, _ = next(data_iter)

text_emb = text_emb.to(device)
text_emb = text_emb[:, 0, :]  #first caption

model.eval()
sample_batch_size = 16
x_init = torch.randn((sample_batch_size, 3, 256, 256), device=device)
# Define a time grid for the ODE solver (from t=0 to t=1).
T = torch.linspace(0, 1, 10).to(device)
solver = ODESolver(velocity_model=wrapped_model)
with torch.no_grad():
    # Sample intermediate states along the flow.
    sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=0.05, return_intermediates=True, txt=text_emb)

# Use the final time step as the generated images.
generated_images = sol[-1]
# Create a grid of images.
grid = vutils.make_grid(generated_images, nrow=4, normalize=True, scale_each=True)
# Convert grid to a NumPy array for plotting.
np_grid = grid.cpu().numpy().transpose(1, 2, 0)

plt.figure(figsize=(8, 8))
plt.imshow(np_grid)
plt.axis('off')
plt.title("Generated Images")
plt.tight_layout()
plt.savefig("samples/noise_to_img_generated.png")
plt.close()
print("Saved generated images to samples/noise_to_generated.png")