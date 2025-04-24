import torch
from torch import nn, Tensor
from flow_matching.utils import ModelWrapper

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, gamma=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
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
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 768, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 512, kernel_size=4, stride=2, padding=1),
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
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        batch_size = z.shape[0]
        # adding noise to inputs 
        gamma = 0.05
        z = z + self.gamma * torch.randn_like(z)
        z = self.fc(z)
        z = z.view(batch_size, self.hidden_dim, 4, 4)
        return self.decoder(z)
        
    def get_loss(self, emb, x):
        x_hat = self.forward(emb)
        l = nn.MSELoss(reduction="mean")
        loss = l(x_hat, x)
        return loss
    
    @torch.no_grad()
    def sample(self, samples, device):
        samples = samples.to(device)
        x_hat = self.forward(samples)

        return x_hat

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, text_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.text_mlp = nn.Linear(text_emb_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor, t_emb: Tensor, txt_emb: Tensor) -> Tensor:
        h = self.relu(self.gn1(self.conv1(x)))
        t_proj = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(1)
        txt_proj = self.text_mlp(txt_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj + txt_proj
        h = self.relu(self.gn2(self.conv2(h)))
        return h

class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 128, text_emb_dim: int = 512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.down1 = UNetBlock(in_channels, base_channels, time_emb_dim, text_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = UNetBlock(base_channels, base_channels * 2, time_emb_dim, text_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = UNetBlock(base_channels * 2, base_channels * 4, time_emb_dim, text_emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = UNetBlock(base_channels * 4, base_channels * 8, time_emb_dim, text_emb_dim)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(base_channels * 8, base_channels * 16, time_emb_dim, text_emb_dim)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.up_block4 = UNetBlock(base_channels * 16, base_channels * 8, time_emb_dim, text_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up_block3 = UNetBlock(base_channels * 8, base_channels * 4, time_emb_dim, text_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_block2 = UNetBlock(base_channels * 4, base_channels * 2, time_emb_dim, text_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_block1 = UNetBlock(base_channels * 2, base_channels, time_emb_dim, text_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor, t: Tensor, text_emb: Tensor) -> Tensor:
        B = x.shape[0]
        if t.dim() == 0 or t.numel() == 1:
            t = t * torch.ones(B, device=x.device)
        t = t.view(B, 1)
        t_emb = self.time_mlp(t)

        d1 = self.down1(x, t_emb, text_emb)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, t_emb, text_emb)
        p2 = self.pool2(d2)
        d3 = self.down3(p2, t_emb, text_emb)
        p3 = self.pool3(d3)
        d4 = self.down4(p3, t_emb, text_emb)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4, t_emb, text_emb)

        up4 = self.up4(bn)
        cat4 = torch.cat([up4, d4], dim=1)
        upb4 = self.up_block4(cat4, t_emb, text_emb)
        up3 = self.up3(upb4)
        cat3 = torch.cat([up3, d3], dim=1)
        upb3 = self.up_block3(cat3, t_emb, text_emb)
        up2 = self.up2(upb3)
        cat2 = torch.cat([up2, d2], dim=1)
        upb2 = self.up_block2(cat2, t_emb, text_emb)
        up1 = self.up1(upb2)
        cat1 = torch.cat([up1, d1], dim=1)
        upb1 = self.up_block1(cat1, t_emb, text_emb)

        out = self.final_conv(upb1)
        return out

class WrappedModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, txt: torch.Tensor, **extras):
        return self.model(x, t, txt)