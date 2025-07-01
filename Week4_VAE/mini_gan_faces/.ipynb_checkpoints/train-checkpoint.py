import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import get_celeba_loader

# === Hyperparameters ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 25
z_dim = 100
batch_size = 128
save_dir = "outputs"
os.makedirs(save_dir, exist_ok=True)

def train():
    # Load models
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    # Loss and Optimizers
    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Data
    data_loader = get_celeba_loader("data/celeba", batch_size=batch_size)

    # Fixed noise for evaluation
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    # === Training Loop ===
    for epoch in range(epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            real_labels = torch.ones(bs, 1).to(device)
            fake_labels = torch.zeros(bs, 1).to(device)

            # --- Train Discriminator ---
            noise = torch.randn(bs, z_dim, 1, 1, device=device)
            fake_imgs = G(noise)

            D_real = D(real_imgs)
            D_fake = D(fake_imgs.detach())

            D_loss = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
            opt_D.zero_grad()
            D_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            output = D(fake_imgs)
            G_loss = criterion(output, real_labels)
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

            pbar.set_postfix(G_loss=G_loss.item(), D_loss=D_loss.item())

        # Save samples after each epoch
        with torch.no_grad():
            fake_samples = G(fixed_noise).detach().cpu()
            save_image(fake_samples, f"{save_dir}/epoch_{epoch+1}.png", normalize=True)

        # Save model
        torch.save(G.state_dict(), f"{save_dir}/gen_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()
