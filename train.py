import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.model import PhysicsGuidedUNet
from utils.dataset import UIEBDataset

# -------- Setup --------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

os.makedirs("outputs", exist_ok=True)

# -------- Dataset --------
dataset = UIEBDataset("data/raw", "data/reference")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# -------- Model --------
model = PhysicsGuidedUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l1_loss = nn.L1Loss()

# -------- Metrics --------
def compute_metrics(pred, gt):
    pred = pred.cpu().numpy().transpose(1,2,0)
    gt = gt.cpu().numpy().transpose(1,2,0)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_val = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

    return psnr, ssim_val

def uiqm(img):
    img = (img * 255).astype(np.uint8)

    rg = img[:,:,0] - img[:,:,1]
    yb = 0.5*(img[:,:,0] + img[:,:,1]) - img[:,:,2]
    uicm = np.std(rg) + np.std(yb)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    uism = cv2.Laplacian(gray, cv2.CV_64F).var()
    uiconm = np.std(gray)

    return 0.0282*uicm + 0.2953*uism + 3.5753*uiconm

# -------- Gradient --------
def gradient(x):
    gx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    gy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return gx, gy

def edge_loss(pred, gt):
    px, py = gradient(pred)
    gx, gy = gradient(gt)
    return (px - gx).abs().mean() + (py - gy).abs().mean()

# -------- Save Preview --------
def save_images(inp, pred, gt, epoch):
    inp = inp[0].cpu().numpy().transpose(1,2,0)
    pred = pred[0].detach().cpu().numpy().transpose(1,2,0)
    gt = gt[0].cpu().numpy().transpose(1,2,0)

    def to_img(x):
        x = (x * 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"outputs/epoch{epoch}_input.png", to_img(inp[:,:,:3]))
    cv2.imwrite(f"outputs/epoch{epoch}_pred.png", to_img(pred))
    cv2.imwrite(f"outputs/epoch{epoch}_gt.png", to_img(gt))

# -------- Training --------
best_psnr = 0

for epoch in range(30):
    model.train()
    total_loss = 0

    for inp, gt in loader:
        inp = inp.to(device)
        gt = gt.to(device)

        pred = model(inp)

        # ---- Loss ----
        l1 = l1_loss(pred, gt)
        ssim_loss = 1 - ssim(pred, gt, data_range=1.0)

        color_loss = torch.mean(
            torch.abs(torch.mean(pred, dim=[2,3]) - torch.mean(gt, dim=[2,3]))
        )

        e_loss = edge_loss(pred, gt)

        loss = 1.2*l1 + 0.4*ssim_loss + 0.2*color_loss + 0.1*e_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # -------- Evaluation --------
    model.eval()
    total_psnr, total_ssim, total_uiqm = 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, (inp, gt) in enumerate(loader):
            inp = inp.to(device)
            gt = gt.to(device)

            pred = model(inp)

            for b in range(pred.shape[0]):
                p = pred[b]
                g = gt[b]

                psnr, ssim_val = compute_metrics(p, g)
                u = uiqm(p.cpu().numpy().transpose(1,2,0))

                total_psnr += psnr
                total_ssim += ssim_val
                total_uiqm += u
                count += 1

            # save preview (first batch only)
            if i == 0:
                save_images(inp, pred, gt, epoch+1)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_uiqm = total_uiqm / count

    print(f"PSNR: {avg_psnr:.2f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"UIQM: {avg_uiqm:.2f}")

    # -------- Save Best Model --------
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved Best Model!")