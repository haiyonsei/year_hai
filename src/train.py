# -*- coding: utf-8 -*-
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ---------------------------
# Utils
# ---------------------------
def parse_int_list(s):
    """comma로 구분된 정수 리스트 파서: '2,2,2,2' -> [2,2,2,2]"""
    vals = [int(x.strip()) for x in s.split(',') if len(x.strip()) > 0]
    return vals

def fmt_list(xs):
    """리스트를 run name에 넣기 위한 짧은 문자열로 변환: [2,2,2,2] -> '2-2-2-2'"""
    return '-'.join(str(x) for x in xs)


# ---------------------------
# Model blocks
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act='lrelu', negative_slope=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # shortcut (채널/stride 다르면 1x1 proj)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out


def make_stage(in_ch, out_ch, num_blocks, act='lrelu', negative_slope=0.1, stride=1):
    blocks = []
    blocks.append(ResidualBlock(in_ch, out_ch, stride=stride, act=act, negative_slope=negative_slope))
    for _ in range(num_blocks - 1):
        blocks.append(ResidualBlock(out_ch, out_ch, stride=1, act=act, negative_slope=negative_slope))
    return nn.Sequential(*blocks)


class DeepHybridCNN_Res(nn.Module):
    """
    ResNet 스타일: 4개 스테이지의 블록 수/채널/stride를 인자로 조절.
    입력 4x4라면 strides=(1,1,1,1)을 권장.
    """
    def __init__(
        self,
        num_points,
        stage_blocks=(2, 2, 2, 2),
        stage_channels=(64, 128, 256, 512),
        strides=(1, 1, 1, 1),
        act='lrelu'
    ):
        super().__init__()
        assert len(stage_blocks) == 4 and len(stage_channels) == 4 and len(strides) == 4, \
            "stage_blocks/stage_channels/strides 모두 길이 4여야 합니다."
        neg = 0.1
        in_ch = 1
        feats = []

        for nb, out_ch, st in zip(stage_blocks, stage_channels, strides):
            feats.append(make_stage(in_ch, out_ch, nb, act=act, negative_slope=neg, stride=st))
            in_ch = out_ch

        feats.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*feats)

        last_ch = stage_channels[-1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_ch, num_points)
        )

    def forward(self, x):
        #x = 2*x - 1
        x = self.features(x)
        x = self.fc(x)
        return x


# ---------------------------
# Data utils
# ---------------------------
def restore_pixel_order(X_flat):
    """
    입력이 4x4 flatten되어 있을 때 (아래->위 순서) 1x4x4로 복원
    """
    N = X_flat.shape[0]
    X_restored = np.zeros((N, 1, 4, 4), dtype=np.float32)
    for i in range(N):
        mat = np.zeros((4, 4), dtype=np.float32)
        for row in range(4):
            mat[3 - row, :] = X_flat[i, row * 4:(row + 1) * 4]
        X_restored[i, 0] = mat
    return X_restored


# ---------------------------
# Train
# ---------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}")

    # --- Data ---
    data_root = args.data_root
    train_data = np.load(os.path.join(data_root, "training", "training_data.npz"), mmap_mode='r')
    val_data   = np.load(os.path.join(data_root, "validation", "validation_data.npz"), mmap_mode='r')
    test_data  = np.load(os.path.join(data_root, "test", "test_data.npz"), mmap_mode='r')

    X_train = train_data["X"].astype(np.float32)
    Y_train = train_data["Y"].astype(np.float32)
    X_val   = val_data["X"].astype(np.float32)
    Y_val   = val_data["Y"].astype(np.float32)
    X_test  = test_data["X"].astype(np.float32)
    Y_test  = test_data["Y"].astype(np.float32)
    freq    = train_data["freq"]
    num_points = Y_train.shape[1]

    print(f"✅ Dataset loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | freq_points={num_points}")

    X_train_t = torch.tensor(restore_pixel_order(X_train))
    Y_train_t = torch.tensor(Y_train)
    X_val_t   = torch.tensor(restore_pixel_order(X_val))
    Y_val_t   = torch.tensor(Y_val)
    X_test_t  = torch.tensor(restore_pixel_order(X_test))
    Y_test_t  = torch.tensor(Y_test)

    # --- Model ---
    model = DeepHybridCNN_Res(
        num_points=num_points,
        stage_blocks=args.stage_blocks,
        stage_channels=args.stage_channels,
        strides=args.strides,
        act=args.act
    ).to(device)

    # --- W&B init (run name = 주요 HP 요약) ---
    run_name = f"{args.run_prefix or 'res'}-" \
               f"b{fmt_list(args.stage_blocks)}-" \
               f"C{fmt_list(args.stage_channels)}-" \
               f"lr{args.lr}-wd{args.weight_decay}-" \
               f"bs{args.batch_size}-" \
               f"step{args.step_size}-g{args.gamma}"
    wandb.init(project=args.project, name=run_name)
    wandb.config.update({
        "stage_blocks": args.stage_blocks,
        "stage_channels": args.stage_channels,
        "strides": args.strides,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "act": args.act,
        "data_root": args.data_root,
    })

    # --- Optim / Sched / Loss / Loader ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=args.batch_size, shuffle=False)

    # --- Train loop ---
    num_epochs = args.epochs
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xv.to(device)), yv.to(device)).item() * xv.size(0)
                for xv, yv in val_loader
            ) / len(val_loader.dataset)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | lr={current_lr:.6e}")

    print("✅ Training complete.")

    # --- Loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training & Validation Loss ({num_epochs} epochs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    wandb.log({"Loss Curve": wandb.Image(plt)})
    plt.close()

    # --- Test preds (샘플 플롯 업로드) ---
    model.eval()
    with torch.no_grad():
        Y_pred_test = model(X_test_t.to(device)).cpu().numpy()

    num_samples = min(100, len(X_test))
    rand_idx = np.random.choice(len(X_test), num_samples, replace=False)
    test_images = []
    for i, idx in enumerate(rand_idx):
        plt.figure(figsize=(5, 3))
        plt.plot(freq, Y_test[idx], 'g-', label='True S11', linewidth=1.8)
        plt.plot(freq, Y_pred_test[idx], 'r--', label='Predicted', linewidth=1.8)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("S11 (dB)")
        plt.title(f"Test Sample #{idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        test_images.append(wandb.Image(plt, caption=f"Test Sample #{idx}"))
        plt.close()
    wandb.log({"Test Predictions": test_images})

    # --- Save ---
    model_path = os.path.join(args.save_dir, "trained_best_model.pth")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 데이터/출력
    parser.add_argument("--data_root", type=str, default="../data/Rogers_dataset")
    parser.add_argument("--save_dir", type=str, default=".")
    # 모델 하이퍼파라미터
    parser.add_argument("--stage_blocks", type=parse_int_list, default=parse_int_list("2,2,2,2"))
    parser.add_argument("--stage_channels", type=parse_int_list, default=parse_int_list("512,1024,2048,4096"))
    parser.add_argument("--strides", type=parse_int_list, default=parse_int_list("1,1,1,1"))
    parser.add_argument("--act", type=str, default="lrelu", choices=["lrelu", "relu"])
    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    # 스케줄러
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.1)
    # W&B
    parser.add_argument("--project", type=str, default="251102_best")
    parser.add_argument("--run_prefix", type=str, default="long_run")

    args = parser.parse_args()
    main(args)

