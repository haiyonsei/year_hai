# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

BOS_IDX = 2   # 0,1: 실제 비트, 2: BOS 토큰


# ---------------------------
# (옵션) 4x4 복원: 디버깅/시각화용
# ---------------------------
def restore_pixel_order(X_flat):
    """
    입력이 4x4 flatten되어 있을 때 (아래->위 순서) 1x4x4로 복원
    (forward 코드와 동일한 형태)
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
# 작은 Transformer AR 모델
# ---------------------------
class SmallTransformerAR(nn.Module):
    """
    Autoregressive Transformer:
      - 입력 토큰: [BOS, x0, x1, ..., x14] (길이 L = 16, num_bits=16 가정)
      - 출력: 각 위치 t에서 '다음 비트 x_t'에 대한 logit (길이 L = 16)
        logits[:,0] -> x0, logits[:,1] -> x1, ..., logits[:,15] -> x15
      - 조건: y (S11 스펙트럼) → Linear → d_model → 모든 토큰 embedding에 더함
      - causal mask 사용 (i 위치는 <= i만 attend)
    """
    def __init__(
        self,
        num_points,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_len=16,      # = num_bits
        vocab_size=3,    # 0,1,BOS
        dropout=0.1,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.max_len = max_len

        # 토큰/포지션/컨디션 임베딩
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.cond_linear = nn.Linear(num_points, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(d_model, 1)  # 각 위치의 logit

    def _generate_causal_mask(self, L, device):
        # (L,L) 상삼각 부분 True → 미래를 mask
        # TransformerEncoder: mask에서 True = "볼 수 없음"
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, y, tokens):
        """
        y:      (B, num_points)
        tokens: (B, L<=max_len)  [BOS, x0, ..., x_{L-2}]
        반환:   logits: (B, L), 각 위치의 '다음 비트' logit
        """
        B, L = tokens.shape
        device = tokens.device

        tok_emb = self.token_embed(tokens)              # (B,L,d_model)

        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B,L)
        pos_emb = self.pos_embed(pos_idx)               # (B,L,d_model)

        cond = self.cond_linear(y)                      # (B,d_model)
        cond = cond.unsqueeze(1).expand(B, L, self.d_model)  # (B,L,d_model)

        x = tok_emb + pos_emb + cond                    # (B,L,d_model)

        src_mask = self._generate_causal_mask(L, device=device)  # (L,L) bool

        h = self.transformer(x, mask=src_mask)          # (B,L,d_model)

        logits = self.fc_out(h).squeeze(-1)             # (B,L)
        return logits


# ---------------------------
# Train / Eval 루프
# ---------------------------
def train_one_epoch_transformer(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    for yb, xb in loader:
        # yb: (B, num_points), xb: (B,16) {0,1}
        yb = yb.to(device)
        xb = xb.to(device)
        B, num_bits = xb.shape

        # 입력 토큰: [BOS, x0, ..., x14]  (길이 16)
        bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        x_prev = xb[:, :-1].long()                      # (B,15)
        tokens_in = torch.cat([bos, x_prev], dim=1)     # (B,16)

        targets = xb                                   # (B,16) = [x0,...,x15]

        optimizer.zero_grad()
        logits = model(yb, tokens_in)                  # (B,16)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        # accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()              # (B,16)

            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


def eval_transformer(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    with torch.no_grad():
        for yb, xb in loader:
            yb = yb.to(device)
            xb = xb.to(device)
            B, num_bits = xb.shape

            bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            x_prev = xb[:, :-1].long()
            tokens_in = torch.cat([bos, x_prev], dim=1)  # (B,16)

            targets = xb                                  # (B,16)

            logits = model(yb, tokens_in)                 # (B,16)
            loss = criterion(logits, targets)

            total_loss += loss.item() * B
            total_samples += B

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


# ---------------------------
# 샘플링: y 주어졌을 때 X 시퀀스를 AR로 생성
# ---------------------------
def sample_transformer_ar(model, y, num_bits=16, greedy=False):
    """
    y: (B, num_points)
    반환: 샘플링된 X_flat: (B,16)  in {0,1}
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    # 시작 토큰: [BOS]
    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)

    bits = []
    with torch.no_grad():
        for t in range(num_bits):
            # tokens: [BOS, x0, ..., x_{t-1}]  (길이 t+1 <= 16)
            logits = model(y, tokens)      # (B, t+1)
            next_logit = logits[:, -1]     # 마지막 위치 → x_t 예측

            probs = torch.sigmoid(next_logit)
            if greedy:
                x_t = (probs > 0.5).long()
            else:
                x_t = torch.bernoulli(probs).long()

            bits.append(x_t.unsqueeze(1))  # (B,1)

            tokens = torch.cat([tokens, x_t.unsqueeze(1)], dim=1)  # prefix 확장

    bits = torch.cat(bits, dim=1)          # (B,16)
    return bits


# ---------------------------
# Main
# ---------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}")

    # --- Data load ---
    data_root = args.data_root
    train_data = np.load(os.path.join(data_root, "training", "training_data.npz"), mmap_mode='r')
    val_data   = np.load(os.path.join(data_root, "validation", "validation_data.npz"), mmap_mode='r')
    test_data  = np.load(os.path.join(data_root, "test", "test_data.npz"), mmap_mode='r')

    X_train_flat = train_data["X"].astype(np.float32)  # (N,16) binary {0,1}
    Y_train      = train_data["Y"].astype(np.float32)  # (N,num_points)
    X_val_flat   = val_data["X"].astype(np.float32)
    Y_val        = val_data["Y"].astype(np.float32)
    X_test_flat  = test_data["X"].astype(np.float32)
    Y_test       = test_data["Y"].astype(np.float32)

    num_points = Y_train.shape[1]
    num_bits   = X_train_flat.shape[1]
    print(f"✅ Dataset loaded: train={len(X_train_flat)}, val={len(X_val_flat)}, test={len(X_test_flat)}, "
          f"num_points={num_points}, num_bits={num_bits}")

    # --- per-sample 입력 normalization (옵션) ---
    if args.normalize_input:
        eps = 1e-8
        # 각 sample에 대해 (freq dimension) 평균 0, 표준편차 1로
        def norm_per_sample(Y):
            mean = Y.mean(axis=1, keepdims=True)
            #std = Y.std(axis=1, keepdims=True) + eps
            return (Y - mean) #/ std

        Y_train = norm_per_sample(Y_train)
        Y_val   = norm_per_sample(Y_val)
        Y_test  = norm_per_sample(Y_test)
        print("🔧 Per-sample normalization applied to Y (train/val/test).")

    # torch tensor
    Y_train_t = torch.tensor(Y_train)
    X_train_t = torch.tensor(X_train_flat)
    Y_val_t   = torch.tensor(Y_val)
    X_val_t   = torch.tensor(X_val_flat)
    Y_test_t  = torch.tensor(Y_test)
    X_test_t  = torch.tensor(X_test_flat)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Y_val_t,   X_val_t),
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Y_test_t,  X_test_t),
                              batch_size=args.batch_size, shuffle=False)

    # --- W&B init ---
    run_name = (
        f"{args.run_prefix}-"
        f"L{args.num_layers}-d{args.d_model}-h{args.nhead}-ff{args.dim_ff}-"
        f"dr{args.dropout}-lr{args.lr}-bs{args.batch_size}"
    )
    wandb.init(project=args.project, name=run_name)
    # 하이퍼파라미터 sweep-friendly: 모든 args를 config에 기록
    wandb.config.update(vars(args))

    # --- Model / Loss / Optim ---
    model = SmallTransformerAR(
        num_points=num_points,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=num_bits,   # =16
        vocab_size=3,
        dropout=args.dropout,
    ).to(device)

    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "transformer_ar_inverse.pth")

    best_val_loss = float("inf")

    # --- Train loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss, train_bit_acc, train_pat_acc = train_one_epoch_transformer(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_bit_acc, val_pat_acc = eval_transformer(
            model, val_loader, criterion, device
        )

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f}, BitAcc: {train_bit_acc:.4f}, PatAcc: {train_pat_acc:.4f} | "
              f"Val Loss: {val_loss:.6f}, BitAcc: {val_bit_acc:.4f}, PatAcc: {val_pat_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bit_acc": train_bit_acc,
            "train_pattern_acc": train_pat_acc,
            "val_loss": val_loss,
            "val_bit_acc": val_bit_acc,
            "val_pattern_acc": val_pat_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Best model updated & saved to {save_path}")
            wandb.run.summary["best_val_loss"] = best_val_loss

    # --- Test ---
    print("🔍 Evaluating best model on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_loss, test_bit_acc, test_pat_acc = eval_transformer(
        model, test_loader, criterion, device
    )
    print(f"📊 Test Loss: {test_loss:.6f}, Test BitAcc: {test_bit_acc:.4f}, Test PatAcc: {test_pat_acc:.4f}")

    wandb.log({
        "test_loss": test_loss,
        "test_bit_acc": test_bit_acc,
        "test_pattern_acc": test_pat_acc,
    })
    wandb.run.summary["test_loss"] = test_loss
    wandb.run.summary["test_bit_acc"] = test_bit_acc
    wandb.run.summary["test_pattern_acc"] = test_pat_acc

    # --- 샘플링 예시 + GT/PRED 비교 이미지 업로드 ---
    model.eval()
    with torch.no_grad():
        num_samples = min(32, len(X_test_t))   # 이미지로 올릴 샘플 수
        idx = torch.randperm(len(X_test_t))[:num_samples]
        Y_sample = Y_test_t[idx].to(device)
        X_true   = X_test_t[idx].cpu().numpy()
        X_gen    = sample_transformer_ar(model, Y_sample, num_bits=num_bits, greedy=False).cpu().numpy()

    # 4x4로 복원
    X_true_img = restore_pixel_order(X_true)  # (N,1,4,4)
    X_gen_img  = restore_pixel_order(X_gen)

    images = []
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(X_true_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("GT")
        axes[0].axis("off")

        axes[1].imshow(X_gen_img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Pred")
        axes[1].axis("off")

        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Sample #{int(idx[i])}"))
        plt.close(fig)

    wandb.log({"GT_vs_Pred_Layouts": images})

    print("👉 Example samples (first few, flatten):")
    print("True:")
    print(X_true[:8])
    print("Generated (sampled):")
    print(X_gen[:8])

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/Rogers_dataset")
    parser.add_argument("--save_dir", type=str, default="./transformer_ar_inverse_models")

    # Transformer 하이퍼파라미터 (기본값: 좀 더 큰 모델)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 학습 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)

    # 입력 normalization 옵션
    parser.add_argument("--normalize_input", action="store_true",
                        help="If set, per-sample normalize Y (zero mean, unit std).")

    # W&B 옵션
    parser.add_argument("--project", type=str, default="rogers_inverse_ar")
    parser.add_argument("--run_prefix", type=str, default="ar")

    args = parser.parse_args()
    main(args)

