import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.float()
        
        #should be 512, 1024
        self.dec_lin1 = nn.Linear(input_dim, hidden_dim)
        #below tbd, maybe add skip layer?
        self.decoder =  nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        z = torch.relu(self.dec_lin1(z))
        z = z.view(z.shape[0], 16, 8, 8)

        x_hat = self.decoder(z)
        return x_hat
        
    def get_loss(self, emb, x):
        x_hat = self.forward(emb)
        l = nn.MSELoss(reduction="mean")
        loss = l(x_hat, x)
        return loss
    
    # @torch.no_grad()
    # NEED TO SCALE UP BY 255
    # def sample(self, n_samples, device):
    #     random_z = torch.randn(n_samples, self.latent_dim).to(device)
    #     x_hat = self.decoding(random_z)
    #     return x_hat