# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import wandb
import argparse


# ==========================================================
# 0. Seed (재현성 보장)
# ==========================================================
seed = 2021
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Utils
# ==========================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_samples_to_wandb(title, X, Y_true, Y_pred, freq, max_samples=100):
    num_samples = min(max_samples, len(X))
    idxs = np.random.choice(len(X), num_samples, replace=False)

    imgs = []
    for idx in idxs:
        pixel = X[idx, 0]
        t_curve = Y_true[idx]
        p_curve = Y_pred[idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(pixel, cmap="gray_r", vmin=0, vmax=1)
        axes[0].set_title(f"{title} Pixel idx={idx}")
        axes[0].axis("off")

        axes[1].plot(freq, t_curve, label="True", lw=2)
        axes[1].plot(freq, p_curve, '--', label="Pred", lw=2)
        axes[1].set_title(f"{title} S11 idx={idx}")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[1].set_ylabel("S11 (dB)")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        imgs.append(wandb.Image(fig))
        plt.close()

    wandb.log({f"{title} Samples": imgs})


# ==========================================================
# Residual Block
# ==========================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act="lrelu", negative_slope=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        self.act = nn.LeakyReLU(negative_slope, inplace=True) if act == "lrelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj:
            identity = self.proj(identity)
        return self.act(out + identity)


def make_stage(in_ch, out_ch, num_blocks, act="lrelu"):
    layers = [ResidualBlock(in_ch, out_ch, stride=1, act=act)]
    for _ in range(num_blocks - 1):
        layers.append(ResidualBlock(out_ch, out_ch, stride=1, act=act))
    return nn.Sequential(*layers)


# ==========================================================
# Deep CNN 10×10
# ==========================================================
class DeepCNN10x10(nn.Module):
    def __init__(self, num_points,
                 stage_blocks=(3,4,4,5),
                 stage_channels=(128,256,512,512),  # ★ 채널 증가 버전
                 act="lrelu"):
        super().__init__()

        stages = []
        in_ch = 1
        for nb, out_ch in zip(stage_blocks, stage_channels):
            stages.append(make_stage(in_ch, out_ch, nb, act))
            in_ch = out_ch

        self.features = nn.Sequential(*stages, nn.AdaptiveAvgPool2d(1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stage_channels[-1], num_points)
        )

    def forward(self, x):
        z = self.features(x)
        return self.fc(z)


# ==========================================================
# TRAINING PIPELINE
# ==========================================================
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Load dataset
    train_npz = np.load(os.path.join(args.data_root, "dataset_train.npz"))
    valid_npz = np.load(os.path.join(args.data_root, "dataset_valid.npz"))
    test_npz  = np.load(os.path.join(args.data_root, "dataset_test.npz"))

    X_train = train_npz["X"].astype(np.float32)[:, None, :, :]
    Y_train = train_npz["Y"].astype(np.float32)
    X_val   = valid_npz["X"].astype(np.float32)[:, None, :, :]
    Y_val   = valid_npz["Y"].astype(np.float32)
    X_test  = test_npz["X"].astype(np.float32)[:, None, :, :]
    Y_test  = test_npz["Y"].astype(np.float32)

    freq = train_npz["freq"].astype(np.float32)
    print("Using original resolution:", len(freq), "points")

    num_points = len(freq)

    # tensor 변환
    X_train_t = torch.tensor(X_train)
    Y_train_t = torch.tensor(Y_train)
    X_val_t   = torch.tensor(X_val)
    Y_val_t   = torch.tensor(Y_val)
    X_test_t  = torch.tensor(X_test)
    Y_test_t  = torch.tensor(Y_test)

    # 모델 구성
    model = DeepCNN10x10(num_points=num_points, act=args.act).to(device)

    # wandb
    wandb.init(project="251123_3")

    n_params = count_parameters(model)
    print(f"📏 Model parameters = {n_params:,}")
    wandb.log({"num_parameters": n_params})

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t),
                              batch_size=args.batch_size, shuffle=False)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_train = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * xb.size(0)

        train_loss = total_train / len(train_loader.dataset)

        # validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv)
                total_val += criterion(pred, yv).item() * xv.size(0)

        val_loss = total_val / len(val_loader.dataset)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] train={train_loss:.6f}, val={val_loss:.6f}")

    # Prediction
    model.eval()
    with torch.no_grad():
        Y_pred_train = model(X_train_t.to(device)).cpu().numpy()
        Y_pred_test  = model(X_test_t.to(device)).cpu().numpy()

    log_samples_to_wandb("TRAIN", X_train, Y_train, Y_pred_train, freq)
    log_samples_to_wandb("TEST",  X_test,  Y_test,  Y_pred_test,  freq)

    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "trained_251123_3_stride1_128_256_512_512.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path} !!")


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root", type=str,
        default=os.path.join(os.path.expanduser("~"), "Desktop", "1122_dataset")
    )
    parser.add_argument("--save_dir", type=str, default="./saved_models")

    parser.add_argument("--act", type=str, default="lrelu")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.3)

    args = parser.parse_args()
    main(args)
