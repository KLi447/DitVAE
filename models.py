import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
from PIL import Image

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.float()
        
        #should be 512, 1024
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        batch_size = z.shape[0]
        z = self.fc(z)
        z = z.view(batch_size, 1024, 4, 4)
        return self.decoder(z)
        
    def get_loss(self, emb, x):
        x_hat = self.forward(emb)
        l = nn.MSELoss(reduction="mean")
        loss = l(x_hat, x)
        return loss
    
    @torch.no_grad()
    def sample(self, samples, device, filename="sample.jpg", save_dir="./samples"):
        samples = samples.to(device)
        x_hat = self.forward(samples)

        x_hat = (x_hat * 255).clamp(0, 255).byte()
        grid = vutils.make_grid(x_hat, nrow=4, padding=2)
        img = grid.permute(1, 2, 0).cpu().numpy()

        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(img).save(os.path.join(save_dir, filename))