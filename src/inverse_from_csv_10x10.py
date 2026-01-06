# inverse_from_csv_10x10.py
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BOS_IDX = 2   # 0,1: 실제 비트, 2: BOS 토큰


def sample_transformer_ar(model, y, num_bits, greedy,
                          ordering, height, width):
    """
    y: (B, num_points)
    반환: X_flat: (B,num_bits) in {0,1}, original flatten ordering
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    B = y.size(0)

    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
    bits_chain = []

    with torch.no_grad():
        for t in range(num_bits):
            logits = model(y, tokens)      # (B, t+1)
            next_logit = logits[:, -1]     # (B,)

            probs = torch.sigmoid(next_logit)
            if greedy:
                x_t = (probs > 0.5).long()
            else:
                x_t = torch.bernoulli(probs).long()   # ★ 진짜 random sampling

            bits_chain.append(x_t.unsqueeze(1))       # (B,1)
            tokens = torch.cat([tokens, x_t.unsqueeze(1)], dim=1)

    bits_chain = torch.cat(bits_chain, dim=1)        # (B,num_bits) in chain-order

    # chain-order → original-order
    order_idx_np = get_order_indices(ordering, num_bits, height, width)
    order_idx = torch.from_numpy(order_idx_np).to(device)
    bits_flat = torch.zeros_like(bits_chain)
    bits_flat[:, order_idx] = bits_chain
    return bits_flat.cpu().numpy()


# ---------------------------
# HxW 복원 (row-major, train 코드와 동일)
# ---------------------------
def restore_pixel_order(X_flat, height, width):
    """
    입력이 (N, H*W) flatten되어 있을 때
    1xHxW로 복원 (row-major 기준).
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width, f"num_bits({num_bits}) != H*W({height*width})"
    X_restored = X_flat.reshape(N, 1, height, width)
    return X_restored.astype(np.float32)


# ---------------------------
# Canonicalization for y-axis symmetry (좌우 대칭)
# ---------------------------
def horizontal_flip_y_axis(X_flat, height, width):
    """
    HxW 패턴을 y축(좌우) 대칭으로 플립.
    X_flat: (N, H*W), row-major ordering 가정.
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]   # 열 방향 반전
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    """
    각 패턴에 대해 y축 대칭 패턴을 만들고,
    두 벡터 중 사전순(lexicographic)으로 작은 쪽을 canonical 정답으로 사용.
    """
    X_flat = X_flat.copy()
    X_flip = horizontal_flip_y_axis(X_flat, height, width)
    N = X_flat.shape[0]
    X_can = np.empty_like(X_flat)

    for i in range(N):
        a = X_flat[i]
        b = X_flip[i]
        if list(a) <= list(b):
            X_can[i] = a
        else:
            X_can[i] = b
    return X_can


# ---------------------------
# Hilbert curve utilities (general 2^k x 2^k)
# ---------------------------
def _hilbert_rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(n, d):
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def get_order_indices(ordering, num_bits, height, width):
    """
    HxW grid 상에서의 순회 순서 정의.
    반환: order_idx (np.array of shape (num_bits,))
      - order_idx[t] = original flattened index at AR position t
    """
    assert num_bits == height * width, "num_bits must be H*W"

    if ordering == "raster":
        order = np.arange(num_bits, dtype=np.int64)

    elif ordering == "snake":
        order_list = []
        for r in range(height):
            if r % 2 == 0:
                cols = range(width)
            else:
                cols = reversed(range(width))
            for c in cols:
                idx = r * width + c
                order_list.append(idx)
        order = np.array(order_list, dtype=np.int64)

    elif ordering == "hilbert":
        max_side = max(height, width)
        n_side = 1
        while n_side < max_side:
            n_side *= 2   # 2^k >= max_side

        coords = []
        for d in range(n_side * n_side):
            x, y = _hilbert_d2xy(n_side, d)
            if x < width and y < height:
                coords.append((x, y))
            if len(coords) == num_bits:
                break

        assert len(coords) == num_bits, \
            f"Hilbert coords length {len(coords)} != num_bits {num_bits}"

        order = np.array([y * width + x for (x, y) in coords], dtype=np.int64)

    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    assert len(order) == num_bits
    return order


# ---------------------------
# 1D ResNet encoder for spectral condition
# ---------------------------
class ResNet1DEncoder(nn.Module):
    """
    y: (B, P) → (B, 1, P) → 1D ResNet → global average pooling → (B, d_model)
    """
    def __init__(self, d_model, num_blocks=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.input_bn   = nn.BatchNorm1d(d_model)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y):
        """
        y: (B, 1, P)
        """
        x = self.input_proj(y)        # (B, d_model, P)
        x = self.input_bn(x)
        x = self.act(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.act(out + residual)

        x = x.mean(dim=-1)           # (B, d_model)
        return x


# ---------------------------
# 10x10용 SmallTransformerAR (train 코드와 동일 구조)
# ---------------------------
class SmallTransformerAR(nn.Module):
    def __init__(
        self,
        num_points,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_len=100,      # = num_bits = H*W
        vocab_size=3,    # 0,1,BOS
        dropout=0.1,
        spectral_cond="linear",   # 'linear' or 'mlp' or 'transformer' or 'resnet1d'
        use_2d_pos=False,
        chain2spatial=None,       # Tensor of shape (max_len,)
        height=10,
        width=10,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.max_len = max_len
        self.spectral_cond_type = spectral_cond
        self.use_2d_pos = use_2d_pos
        self.height = height
        self.width = width

        # 토큰/포지션/컨디션 임베딩
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(max_len, d_model)   # 1D (chain index)

        # 2D positional embedding (optional)
        if self.use_2d_pos:
            assert chain2spatial is not None, "use_2d_pos=True 이면 chain2spatial을 넘겨줘야 합니다."
            assert chain2spatial.numel() >= max_len
            self.pos2d_embed = nn.Embedding(height * width, d_model)
            self.register_buffer("chain2spatial", chain2spatial.clone())

        # spectral condition encoder
        if spectral_cond == "linear":
            self.cond_encoder = nn.Linear(num_points, d_model)
        elif spectral_cond == "mlp":
            self.cond_encoder = nn.Sequential(
                nn.Linear(num_points, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif spectral_cond == "transformer":
            self.freq_in_proj = nn.Linear(1, d_model)
            self.freq_pos_embed = nn.Embedding(num_points, d_model)
            cond_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.cond_transformer = nn.TransformerEncoder(
                cond_encoder_layer,
                num_layers=2
            )
        elif spectral_cond == "resnet1d":
            self.resnet1d = ResNet1DEncoder(d_model=d_model, num_blocks=3, kernel_size=3)
        else:
            raise ValueError(f"Unknown spectral_cond: {spectral_cond}")

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

        self.fc_out = nn.Linear(d_model, 1)

    def _generate_causal_mask(self, L, device):
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def encode_condition(self, y):
        if self.spectral_cond_type in ["linear", "mlp"]:
            return self.cond_encoder(y)
        elif self.spectral_cond_type == "transformer":
            B, P = y.shape
            device = y.device
            y_seq = y.unsqueeze(-1)                 # (B,P,1)
            feat = self.freq_in_proj(y_seq)        # (B,P,d_model)

            pos_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
            pos_emb = self.freq_pos_embed(pos_idx) # (B,P,d_model)

            feat = feat + pos_emb                  # (B,P,d_model)
            h = self.cond_transformer(feat)        # (B,P,d_model)

            cond_vec = h.mean(dim=1)               # (B,d_model)
            return cond_vec
        elif self.spectral_cond_type == "resnet1d":
            y_seq = y.unsqueeze(1)                 # (B,1,P)
            cond_vec = self.resnet1d(y_seq)        # (B,d_model)
            return cond_vec
        else:
            raise RuntimeError("Invalid spectral_cond_type")

    def forward(self, y, tokens):
        """
        y:      (B, num_points)
        tokens: (B, L<=max_len) [BOS, x0, ..., x_{L-2}]
        반환:   logits: (B, L), 각 위치의 '다음 비트' logit
        """
        B, L = tokens.shape
        device = tokens.device

        tok_emb = self.token_embed(tokens)              # (B,L,d_model)
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embed(pos_idx)               # (B,L,d_model)

        if self.use_2d_pos:
            spatial_idx = self.chain2spatial[:L].to(device)
            spatial_idx = spatial_idx.unsqueeze(0).expand(B, L)
            pos2d_emb = self.pos2d_embed(spatial_idx)
            pos_emb = pos_emb + pos2d_emb

        cond_vec = self.encode_condition(y)             # (B,d_model)
        cond = cond_vec.unsqueeze(1).expand(B, L, self.d_model)

        x = tok_emb + pos_emb + cond

        src_mask = self._generate_causal_mask(L, device=device)
        h = self.transformer(x, mask=src_mask)          # (B,L,d_model)

        logits = self.fc_out(h).squeeze(-1)             # (B,L)
        return logits


# ---------------------------
# Beam search 디코딩 (일반 num_bits 버전)
# ---------------------------
def beam_search_inverse(model, y, num_bits, beam_size,
                        ordering, height, width):
    """
    y: (1, num_points) tensor (normalized if needed)
    beam_size: top-K 개 시퀀스
    """
    model.eval()
    device = next(model.parameters()).device
    y = y.to(device)
    assert y.ndim == 2 and y.size(0) == 1, "beam_search_inverse는 y.shape = (1, num_points)만 지원"

    eps = 1e-9

    beams = [{
        "tokens": torch.tensor([BOS_IDX], dtype=torch.long, device=device),
        "bits": [],
        "log_prob": 0.0,
    }]

    with torch.no_grad():
        for t in range(num_bits):
            all_candidates = []

            token_batch = torch.stack([b["tokens"] for b in beams], dim=0)  # (B_beam, t+1)
            B_beam = token_batch.size(0)
            y_rep = y.repeat(B_beam, 1)  # (B_beam, num_points)

            logits = model(y_rep, token_batch)      # (B_beam, t+1)
            next_logit = logits[:, -1]              # (B_beam,)

            probs = torch.sigmoid(next_logit)
            probs = torch.clamp(probs, min=eps, max=1.0 - eps)
            log_p1 = torch.log(probs)
            log_p0 = torch.log(1.0 - probs)

            for i, beam in enumerate(beams):
                base_log_prob = beam["log_prob"]
                tokens_i = beam["tokens"]
                bits_i = beam["bits"]

                # bit=0
                new_tokens_0 = torch.cat(
                    [tokens_i, torch.tensor([0], device=device, dtype=torch.long)],
                    dim=0
                )
                new_bits_0 = bits_i + [0]
                new_lp_0 = base_log_prob + float(log_p0[i])
                all_candidates.append({
                    "tokens": new_tokens_0,
                    "bits": new_bits_0,
                    "log_prob": new_lp_0,
                })

                # bit=1
                new_tokens_1 = torch.cat(
                    [tokens_i, torch.tensor([1], device=device, dtype=torch.long)],
                    dim=0
                )
                new_bits_1 = bits_i + [1]
                new_lp_1 = base_log_prob + float(log_p1[i])
                all_candidates.append({
                    "tokens": new_tokens_1,
                    "bits": new_bits_1,
                    "log_prob": new_lp_1,
                })

            all_candidates.sort(key=lambda b: b["log_prob"], reverse=True)
            beams = all_candidates[:beam_size]

    K = min(beam_size, len(beams))
    bits_chain = torch.zeros((K, num_bits), dtype=torch.long, device=device)
    for k in range(K):
        bits_chain[k] = torch.tensor(beams[k]["bits"], dtype=torch.long, device=device)

    # chain-order → original-order
    order_idx_np = get_order_indices(ordering, num_bits, height, width)
    order_idx = torch.from_numpy(order_idx_np).to(device)
    bits_flat = torch.zeros_like(bits_chain)
    bits_flat[:, order_idx] = bits_chain
    return bits_flat.cpu().numpy()


# ---------------------------
# CSV 로딩 → S11 벡터 추출
# ---------------------------
def load_s11_from_csv(csv_path, s11_col_name=None):
    """
    CSV에서 S11(dB) 벡터를 읽어와 1D numpy array 로 반환.
    기본으로 'dB(S(1,1)) []' 컬럼을 우선 사용.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    if s11_col_name is not None:
        col = s11_col_name
    else:
        # 기본 컬럼 이름 시도
        candidates = ["dB(S(1,1)) []", "S11", "S_11", "S(1,1)", "dB_S11"]
        col = None
        for c in candidates:
            if c in df.columns:
                col = c
                break
        if col is None:
            raise ValueError(
                f"S11 column not found. Available columns: {list(df.columns)}. "
                f"Specify --s11_col_name."
            )

    y = df[col].values.astype(np.float32)  # (num_points,)
    return y


# ---------------------------
# Main
# ---------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}")

    # --- CSV에서 S11 로드 ---
    print(f"📄 Loading CSV: {args.csv_path}")
    y_raw = load_s11_from_csv(args.csv_path, s11_col_name=args.s11_col_name)
    num_points = y_raw.shape[0]
    print(f"✅ Loaded S11 vector with length = {num_points}")

    # --- 10x10 설정 ---
    H = args.height
    W = args.width
    num_bits = H * W
    print(f"🧩 Using layout: {H}x{W} (num_bits={num_bits})")

    # --- model 입력용 Y normalization (train 스크립트 설정과 맞춰야 함) ---
    if args.normalize_input:
        mean = y_raw.mean(keepdims=True)
        y_in = (y_raw - mean)[None, :]   # (1,num_points)
        print("🔧 Per-sample normalization applied to input S11.")
    else:
        y_in = y_raw[None, :]            # (1,num_points)

    y_in_t = torch.tensor(y_in, dtype=torch.float32, device=device)

    # --- ordering (chain-order) ---
    order_idx = get_order_indices(args.ordering, num_bits, H, W)
    print(f"🔁 Using ordering = {args.ordering}, order_idx[:20] = {order_idx[:20].tolist()}")
    chain2spatial = torch.from_numpy(order_idx).long()

    # --- 모델 구성 & load ---
    model = SmallTransformerAR(
        num_points=num_points,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=num_bits,
        vocab_size=3,
        dropout=args.dropout,
        spectral_cond=args.spectral_cond,
        use_2d_pos=args.use_2d_pos,
        chain2spatial=chain2spatial,
        height=H,
        width=W,
    ).to(device)

    print(f"📁 Loading model from: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


    # --- 후보 생성: beam search vs random sampling ---
    if args.decode_method == "beam":
        print(f"🎯 Running beam search with beam_size={args.beam_size}...")
        X_candidates = beam_search_inverse(
            model,
            y_in_t,
            num_bits=num_bits,
            beam_size=args.beam_size,
            ordering=args.ordering,
            height=H,
            width=W,
        )  # (K,num_bits) 0/1  where K = beam_size
        X_candidates = X_candidates[:args.num_candidates]

    elif args.decode_method == "sampling":
        print(f"🎲 Running random sampling with num_candidates={args.num_candidates}...")
        # y_in_t: (1, num_points) → (num_candidates, num_points)
        y_rep = y_in_t.repeat(args.num_candidates, 1)
        X_candidates = sample_transformer_ar(
            model,
            y_rep,
            num_bits=num_bits,
            greedy=False,                # ★ 확률적으로 뽑기
            ordering=args.ordering,
            height=H,
            width=W,
        )  # (num_candidates, num_bits) 0/1

    else:
        raise ValueError(f"Unknown decode_method: {args.decode_method}")

    print(f"✅ Generated {X_candidates.shape[0]} candidates.")

        # --- canonical_target 옵션이 켜져 있으면, 좌우 대칭 canonicalization ---
    if args.canonical_target:
        print("🔁 Applying canonicalization under y-axis flip to generated candidates.")
        X_candidates = canonicalize_under_yflip(X_candidates, height=H, width=W)

    # --- 저장 경로 및 파일명 설정 ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_stem = os.path.splitext(os.path.basename(args.csv_path))[0]

    # --- 각 candidate에 대해 이미지 저장 + CSV용 row 만들기 ---
    X_imgs = restore_pixel_order(X_candidates, H, W)  # (K,1,H,W)

    all_rows = []  # 각 candidate의 1D 패턴을 모아 둘 리스트

    for idx in range(X_candidates.shape[0]):
        pattern_flat = X_candidates[idx]     # (num_bits,)
        pattern_img  = X_imgs[idx, 0]       # (H,W)

        # 2D(HxW) -> bottom row 먼저, 그 다음 위 row ... → 1D 한 줄
        pattern_2d_reordered = np.flipud(pattern_img)        # (H,W), 아래→위 순서로
        pattern_1d = pattern_2d_reordered.reshape(1, -1)     # (1, H*W)

        all_rows.append(pattern_1d)  # 나중에 한 번에 CSV로 저장

        # 이미지 파일명: 입력 csv 이름 + candidate #
        base_name = f"{csv_stem}_candidate{idx}"
        png_path  = os.path.join(args.output_dir, base_name + ".png")

        # 이미지 저장
        plt.figure(figsize=(3, 3))
        plt.imshow(pattern_img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.title(base_name)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"💾 Saved image for candidate #{idx}: {png_path}")

    # --- 모든 candidates를 한 CSV로 저장 ---
    # all_rows: [ (1,H*W), (1,H*W), ... ] -> (K, H*W)
    all_rows_arr = np.vstack(all_rows)               # (K, H*W)
    csv_path_all = os.path.join(args.output_dir, f"{csv_stem}_candidates.csv")

    # 각 row = 한 candidate, 순서: 맨 아래 row → 맨 위 row 이어붙인 1D
    np.savetxt(csv_path_all, all_rows_arr.astype(int), fmt="%d", delimiter=",")

    print(f"💾 Saved all {X_candidates.shape[0]} candidates to CSV: {csv_path_all}")


    print("🎉 Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 필수 인자
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Input CSV file path containing S11 data.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pth from 10x10 inverse training script.")

    # CSV 컬럼 이름 (없으면 자동 탐색)
    parser.add_argument("--s11_col_name", type=str, default=None,
                        help="Column name for S11 dB in CSV (default: try 'dB(S(1,1)) []', etc.)")

    # 레이아웃 정보 (기본 10x10)
    parser.add_argument("--height", type=int, default=10,
                        help="Height of layout (default: 10)")
    parser.add_argument("--width", type=int, default=10,
                        help="Width of layout (default: 10)")

    # 모델 하이퍼파라미터 (train 스크립트와 동일하게 맞춰야 함)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--dim_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--spectral_cond", type=str, default="resnet1d",
                        choices=["linear", "mlp", "transformer", "resnet1d"])
    parser.add_argument("--ordering", type=str, default="hilbert",
                        choices=["raster", "snake", "hilbert"])
    parser.add_argument(
        "--use_2d_pos",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use 2D positional encoding (True/False)",
    )

    parser.add_argument(
        "--decode_method",
        type=str,
        default="beam",
        choices=["beam", "sampling"],
        help="How to generate candidates: 'beam' for beam search, 'sampling' for random sampling.",
    )

    # canonical_target / normalization 옵션
    parser.add_argument(
        "--canonical_target",
        type=lambda x: x.lower() == "true",
        default=True,   # ← 요청대로 default를 True로 설정
        help="If True, apply canonicalization under y-axis flip to generated patterns."
    )
    parser.add_argument(
        "--normalize_input",
        action="store_true",
        help="If set, per-sample normalize Y (zero mean) before feeding to model."
    )

    # beam search / 출력 옵션
    parser.add_argument("--beam_size", type=int, default=200,
                        help="Beam size (K) for beam search (number of candidates).")
    parser.add_argument("--num_candidates", type=int, default=50,
                        help="number of candidates")
    parser.add_argument("--output_dir", type=str, default="./inverse_from_csv_outputs",
                        help="Directory to save candidate images and npy files.")

    args = parser.parse_args()
    main(args)

# python3 ./inverse_from_csv_10x10.py --csv_path ../specs/desired/S_Parameter_Plot_60_1_1994.csv  --model_path ./transformer_ar_inverse_models/ar10x10-L9-d192-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr0.0005-bs128-seed42/best_model.pth
