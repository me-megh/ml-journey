# train.py
import torch
from torch import nn, optim
from generator import Generator
from discriminator import Discriminator
from utils import get_dataloader

# Hyperparams
latent_dim = 100
lr = 0.0002
batch_size = 128
epochs = 1  # Start small

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
dataloader = get_dataloader(batch_size)

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Real and fake labels
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Generator
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), valid)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # Train Discriminator
        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        if i % 100 == 0:
            print(f"[{epoch}/{epochs}] [Batch {i}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
