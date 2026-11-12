import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from medmnist import DermaMNIST
from torchmetrics.image.fid import FrechetInceptionDistance
from model import Generator, latent_dim

EPOCHS = 20
N_FID = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def init_weights(layer):
    name = layer.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
dataset = DermaMNIST(split="train", download=True, size=64, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
seen = 0
for real, _ in dataloader:
    fid.update(((real + 1) / 2).to(device), real=True)
    seen += real.size(0)
    if seen >= N_FID:
        break

grid_noise = torch.randn(16, latent_dim, 1, 1, generator=torch.Generator().manual_seed(123)).to(device)


def run(name, real_label, use_init):
    torch.manual_seed(0)
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    if use_init:
        netG.apply(init_weights)
        netD.apply(init_weights)
    optG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    rows = []
    start = time.time()
    for epoch in range(EPOCHS):
        for real, _ in dataloader:
            real = real.to(device)
            b = real.size(0)
            netD.zero_grad()
            lossD_real = criterion(netD(real), torch.full((b,), real_label, device=device))
            fake = netG(torch.randn(b, latent_dim, 1, 1, device=device))
            lossD_fake = criterion(netD(fake.detach()), torch.zeros(b, device=device))
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()
            netG.zero_grad()
            lossG = criterion(netD(fake), torch.ones(b, device=device))
            lossG.backward()
            optG.step()
        rows.append((epoch + 1, lossD.item(), lossG.item()))
        print(f"[{name}] epoch {epoch + 1}/{EPOCHS}  Loss D: {lossD.item():.4f}  Loss G: {lossG.item():.4f}")
    minutes = (time.time() - start) / 60
    netG.eval()
    with torch.no_grad():
        save_image((netG(grid_noise) + 1) / 2, f"results/grid_{name}.png", nrow=4)
        for _ in range(N_FID // 100):
            fake = netG(torch.randn(100, latent_dim, 1, 1, device=device))
            fid.update(((fake + 1) / 2).clamp(0, 1), real=False)
        score = fid.compute().item()
    fid.reset()
    torch.save(netG.state_dict(), f"results/G_{name}.pth")
    with open(f"results/losses_{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_D", "loss_G"])
        writer.writerows(rows)
    print(f"[{name}] FID: {score:.1f}   training time: {minutes:.1f} min")
    return score, minutes


configs = [
    ("baseline", 1.0, False),
    ("smoothing", 0.9, False),
    ("init", 1.0, True),
    ("both", 0.9, True),
]
summary = []
for name, real_label, use_init in configs:
    score, minutes = run(name, real_label, use_init)
    summary.append((name, real_label, use_init, EPOCHS, round(score, 1), round(minutes, 1)))

with open("results/summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "real_label", "dcgan_init", "epochs", "fid", "train_minutes"])
    writer.writerows(summary)
print("done, see the results folder")