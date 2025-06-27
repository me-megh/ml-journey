# train.py

import os
import torch
from torch import nn, optim
import torchvision.utils as vutils

from generator import Generator
from discriminator import Discriminator
from utils import get_dataloader

# -----------------------
# 🧠 Hyperparameters
# -----------------------
LATENT_DIM = 100
LR = 0.0002
BATCH_SIZE = 128
EPOCHS = 10  # You can increase this
SAVE_DIR = "outputs"

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# ⚙️ Device setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# -----------------------
# 📦 Model setup
# -----------------------
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
dataloader = get_dataloader(BATCH_SIZE)

# -----------------------
# 🔁 Training Loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    for batch_idx, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Labels
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # =====================
        # 🎨 Train Generator
        # =====================
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), valid)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # =====================
        # 🛡️ Train Discriminator
        # =====================
        real_loss = criterion(D(real_imgs), valid)
        fake_loss = criterion(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Logging
        if batch_idx % 100 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {batch_idx}] "
                  f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    # Save samples after each epoch
    with torch.no_grad():
        fixed_noise = torch.randn(64, LATENT_DIM, device=device)
        fake_images = G(fixed_noise).detach().cpu()
        save_path = os.path.join(SAVE_DIR, f"fake_samples_epoch_{epoch}.png")
        vutils.save_image(fake_images, save_path, normalize=True, nrow=8)
        print(f"[INFO] Saved sample image to: {save_path}")

print("[✅] Training Complete!")
