# inverse_from_csv_10x10.py
# -*- coding: utf-8 -*-
import os
import re
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn

BOS_IDX = 2   # 0,1 bits + 2 BOS token

# ============================================================
# Ordering utilities
# ============================================================
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
    return: order_idx (num_bits,)
      - order_idx[t] = original flattened index at AR position t
    """
    assert num_bits == height * width, "num_bits must be H*W"

    if ordering == "raster":
        order = np.arange(num_bits, dtype=np.int64)

    elif ordering == "snake":
        order_list = []
        for r in range(height):
            cols = range(width) if (r % 2 == 0) else reversed(range(width))
            for c in cols:
                order_list.append(r * width + c)
        order = np.array(order_list, dtype=np.int64)

    elif ordering == "hilbert":
        max_side = max(height, width)
        n_side = 1
        while n_side < max_side:
            n_side *= 2

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


# ============================================================
# Canonicalization under left-right flip (y-axis symmetry)
# ============================================================
def restore_pixel_order(X_flat, height, width):
    """
    (N,H*W) -> (N,1,H,W) row-major
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width, f"num_bits({num_bits}) != H*W({height*width})"
    X_restored = X_flat.reshape(N, 1, height, width)
    return X_restored.astype(np.float32)


def horizontal_flip_y_axis(X_flat, height, width):
    """
    left-right flip
    """
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    """
    canonicalize under left-right flip by lexicographic min
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


# ============================================================
# 1D ResNet encoder for spectral condition
# ============================================================
class ResNet1DEncoder(nn.Module):
    def __init__(self, d_model, num_blocks=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.input_bn = nn.BatchNorm1d(d_model)

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
        x = self.input_proj(y)
        x = self.input_bn(x)
        x = self.act(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.act(out + residual)

        x = x.mean(dim=-1)
        return x


# ============================================================
# SmallTransformerAR
# ============================================================
class SmallTransformerAR(nn.Module):
    def __init__(
        self,
        num_points,
        d_model=256,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_len=100,
        vocab_size=3,
        dropout=0.1,
        spectral_cond="linear",
        use_2d_pos=False,
        chain2spatial=None,
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

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        if self.use_2d_pos:
            assert chain2spatial is not None
            assert chain2spatial.numel() >= max_len
            self.pos2d_embed = nn.Embedding(height * width, d_model)
            self.register_buffer("chain2spatial", chain2spatial.clone())

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
                d_model=d_model, nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            )
            self.cond_transformer = nn.TransformerEncoder(cond_encoder_layer, num_layers=2)
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def _generate_causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def encode_condition(self, y):
        if self.spectral_cond_type in ["linear", "mlp"]:
            return self.cond_encoder(y)
        elif self.spectral_cond_type == "transformer":
            B, P = y.shape
            device = y.device
            y_seq = y.unsqueeze(-1)
            feat = self.freq_in_proj(y_seq)
            pos_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
            pos_emb = self.freq_pos_embed(pos_idx)
            feat = feat + pos_emb
            h = self.cond_transformer(feat)
            return h.mean(dim=1)
        elif self.spectral_cond_type == "resnet1d":
            y_seq = y.unsqueeze(1)
            return self.resnet1d(y_seq)
        else:
            raise RuntimeError("Invalid spectral_cond_type")

    def forward(self, y, tokens):
        B, L = tokens.shape
        device = tokens.device

        tok_emb = self.token_embed(tokens)
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embed(pos_idx)

        if self.use_2d_pos:
            spatial_idx = self.chain2spatial[:L].to(device)
            spatial_idx = spatial_idx.unsqueeze(0).expand(B, L)
            pos2d_emb = self.pos2d_embed(spatial_idx)
            pos_emb = pos_emb + pos2d_emb

        cond_vec = self.encode_condition(y)
        cond = cond_vec.unsqueeze(1).expand(B, L, self.d_model)

        x = tok_emb + pos_emb + cond
        src_mask = self._generate_causal_mask(L, device=device)
        h = self.transformer(x, mask=src_mask)
        logits = self.fc_out(h).squeeze(-1)
        return logits


# ============================================================
# CSV loader
# ============================================================
def load_freq_and_s11_from_csv(csv_path, freq_col_name=None, s11_col_name=None):
    import pandas as pd
    df = pd.read_csv(csv_path)

    if s11_col_name is not None:
        s_col = s11_col_name
    else:
        candidates = ["dB(S(1,1)) []", "dB(S(1,1))", "S11", "S_11", "S(1,1)", "dB_S11"]
        s_col = None
        for c in candidates:
            if c in df.columns:
                s_col = c
                break
        if s_col is None:
            raise ValueError(
                f"S11 column not found. Available columns: {list(df.columns)}. "
                f"Specify --s11_col_name."
            )
    s11 = df[s_col].values.astype(np.float32)

    if freq_col_name is not None and freq_col_name in df.columns:
        f_col = freq_col_name
    else:
        f_candidates = ["Freq [GHz]", "Freq[GHz]", "Freq", "Frequency", "frequency", "GHz"]
        f_col = None
        for c in f_candidates:
            if c in df.columns:
                f_col = c
                break

    if f_col is None:
        freq = np.arange(len(s11), dtype=np.float32)
    else:
        freq = df[f_col].values.astype(np.float32)

    return freq, s11


# ============================================================
# Forward surrogate (DeepCNN10x10)
# ============================================================
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


def make_stage2d(in_ch, out_ch, n):
    return nn.Sequential(
        *[ResidualBlock2D(in_ch if i == 0 else out_ch, out_ch) for i in range(n)]
    )


class DeepCNN10x10(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.features = nn.Sequential(
            make_stage2d(1, 128, 3),
            make_stage2d(128, 256, 4),
            make_stage2d(256, 512, 4),
            make_stage2d(512, 512, 5),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_points)
        )

    def forward(self, x):
        return self.fc(self.features(x))


# ============================================================
# Notch helper
# ============================================================
def _find_true_segments(mask_bool_1d: np.ndarray):
    segs = []
    P = mask_bool_1d.size
    in_seg = False
    s = 0
    for i in range(P):
        if mask_bool_1d[i] and not in_seg:
            in_seg = True
            s = i
        elif (not mask_bool_1d[i]) and in_seg:
            segs.append((s, i - 1))
            in_seg = False
    if in_seg:
        segs.append((s, P - 1))
    return segs


def _expand_segment(s, e, P, expand_pts):
    if expand_pts <= 0:
        return s, e
    s2 = max(0, s - expand_pts)
    e2 = min(P - 1, e + expand_pts)
    return s2, e2


def check_overlap_ok_from_segments(pred_db_1d: np.ndarray,
                                  target_notch_segments,
                                  threshold_db: float = -10.0,
                                  expand_pts: int = 0) -> bool:
    P = pred_db_1d.size
    if not target_notch_segments:
        return True

    thr = float(threshold_db)
    for (s, e) in target_notch_segments:
        s2, e2 = _expand_segment(s, e, P, expand_pts)
        seg_min = float(np.min(pred_db_1d[s2:e2+1]))
        if seg_min > thr:
            return False
    return True


def compute_bw_below_threshold_ghz(freq_1d: np.ndarray,
                                  pred_db_1d: np.ndarray,
                                  threshold_db: float = -10.0) -> float:
    below = (pred_db_1d <= float(threshold_db))
    segs = _find_true_segments(below.astype(bool))
    if not segs:
        return 0.0

    bw = 0.0
    for (s, e) in segs:
        if e <= s:
            continue
        bw += float(freq_1d[e] - freq_1d[s])
    return float(bw)


def loss_function_notch_mse(pred_db_t: torch.Tensor,
                            target_db_t: torch.Tensor,
                            notch_threshold_db: float = -10.0) -> torch.Tensor:
    if target_db_t.ndim == 2:
        target_1d = target_db_t.squeeze(0)
    else:
        target_1d = target_db_t

    mask = (target_1d <= float(notch_threshold_db))
    if torch.count_nonzero(mask) == 0:
        return pred_db_t.new_zeros((pred_db_t.size(0),))

    diff = pred_db_t[:, mask] - target_1d[mask].unsqueeze(0)
    return (diff * diff).mean(dim=1)


# ============================================================
# Forward pth auto pick + load
# ============================================================
def _find_default_forward_pth(forward_dir: str) -> str:
    if not os.path.isdir(forward_dir):
        raise FileNotFoundError(f"forward_surrogate_models dir not found: {forward_dir}")
    pths = [os.path.join(forward_dir, f) for f in os.listdir(forward_dir) if f.lower().endswith(".pth")]
    if len(pths) == 0:
        raise FileNotFoundError(f"No .pth found in: {forward_dir}")
    pths.sort()
    return pths[0]


def _safe_load_state_dict(model: nn.Module, ckpt_path: str, device):
    st = torch.load(ckpt_path, map_location=device)
    if isinstance(st, dict):
        st = {k.replace("module.", ""): v for k, v in st.items()}
    model.load_state_dict(st, strict=True)


def _pixel_to_row_bottom_to_top(img_hw: np.ndarray) -> np.ndarray:
    return np.flipud(img_hw).reshape(1, -1).astype(int)


def save_pixel_plus_spectrum_png(out_png: str,
                                 pixel_hw: np.ndarray,
                                 freq_1d: np.ndarray,
                                 pred_db_1d: np.ndarray,
                                 target_db_1d: np.ndarray = None,
                                 title: str = None):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(pixel_hw, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Pixel (10x10)")
    axes[0].axis("off")

    axes[1].plot(freq_1d, pred_db_1d, label="Pred (forward)")
    if target_db_1d is not None:
        axes[1].plot(freq_1d, target_db_1d, label="Target (input)")
    axes[1].axhline(float(-10.0), linestyle="--", linewidth=1, label="-10 dB")

    axes[1].set_xlabel("Freq")
    axes[1].set_ylabel("dB(S11)")
    axes[1].grid(True, alpha=0.3)
    if title:
        axes[1].set_title(title)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ============================================================
# NEW: AR sampling with fixed prefix (micro-batched)
# ============================================================
def _arm_to_prefix_bits(arm_id: int, prefix_len: int):
    bits = []
    for i in range(prefix_len):
        shift = (prefix_len - 1 - i)
        bits.append((arm_id >> shift) & 1)
    return bits


def _prefix_bits_to_str(bits):
    return "".join(str(int(b)) for b in bits)


def _ar_sample_one_microbatch(
    model: nn.Module,
    y_1xP: torch.Tensor,          # (1,P) device
    num_bits: int,
    prefix_bits: list,
    batch_n: int,
    sampling: str
):
    device = next(model.parameters()).device
    model.eval()

    B = int(batch_n)
    prefix_len = len(prefix_bits)

    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
    y_rep = y_1xP.to(device).repeat(B, 1)

    prefix_t = torch.tensor(prefix_bits, dtype=torch.long, device=device).view(1, -1).repeat(B, 1)

    with torch.no_grad():
        for t in range(num_bits):
            logits = model(y_rep, tokens)
            next_logit = logits[:, -1]
            probs = torch.sigmoid(next_logit).clamp(1e-9, 1 - 1e-9)

            if t < prefix_len:
                next_bit = prefix_t[:, t]
            else:
                if sampling == "greedy":
                    next_bit = (probs >= 0.5).long()
                else:
                    next_bit = torch.bernoulli(probs).long()

            tokens = torch.cat([tokens, next_bit.unsqueeze(1)], dim=1)

    bits_chain = tokens[:, 1:1+num_bits].detach().cpu().numpy().astype(np.int64)
    return bits_chain


def ar_sample_with_fixed_prefix_total(
    model: nn.Module,
    y_1xP: torch.Tensor,
    num_bits: int,
    prefix_bits: list,
    total_n: int,
    sampling: str = "stochastic",
    micro_batch: int = 512,
    deadline: float = None,
):
    """
    total_n개를 생성하되, micro_batch 단위로 끊어서 생성.
    deadline(perf_counter 기준)이 주어지면 시간 초과 시 조기 중단.
    """
    outs = []
    made = 0
    total_n = int(total_n)
    micro_batch = max(1, int(micro_batch))

    while made < total_n:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        b = min(micro_batch, total_n - made)
        bits = _ar_sample_one_microbatch(
            model=model,
            y_1xP=y_1xP,
            num_bits=num_bits,
            prefix_bits=prefix_bits,
            batch_n=b,
            sampling=sampling
        )
        outs.append(bits)
        made += b

    if not outs:
        return np.zeros((0, num_bits), dtype=np.int64)
    return np.concatenate(outs, axis=0)


# ============================================================
# main
# ============================================================
def main(args):
    import csv as pycsv
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}", flush=True)

    if device.type == "cuda" and args.gpu_mem_frac < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_frac), device=0)
            print(f"🧩 Set GPU memory fraction = {args.gpu_mem_frac}", flush=True)
        except Exception as e:
            print(f"⚠️ set_per_process_memory_fraction failed: {e}", flush=True)

    print(f"📄 Loading CSV: {args.csv_path}", flush=True)
    freq_raw, y_raw = load_freq_and_s11_from_csv(
        args.csv_path,
        freq_col_name=args.freq_col_name,
        s11_col_name=args.s11_col_name
    )
    num_points = y_raw.shape[0]
    print(f"✅ Loaded S11 vector with length = {num_points}", flush=True)

    H = args.height
    W = args.width
    num_bits = H * W
    print(f"🧩 Using layout: {H}x{W} (num_bits={num_bits})", flush=True)

    if args.normalize_input:
        mean = y_raw.mean(keepdims=True)
        y_in = (y_raw - mean)[None, :]
        print("🔧 Per-sample normalization applied to input S11.", flush=True)
    else:
        y_in = y_raw[None, :]

    y_in_t = torch.tensor(y_in, dtype=torch.float32, device=device)

    order_idx = get_order_indices(args.ordering, num_bits, H, W)
    print(f"🔁 Using ordering = {args.ordering}, order_idx[:20] = {order_idx[:20].tolist()}", flush=True)
    chain2spatial = torch.from_numpy(order_idx).long()

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

    print(f"📁 Loading inverse model from: {args.model_path}", flush=True)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    target_s11_np = y_raw.astype(np.float32)
    target_s11_t = torch.tensor(target_s11_np, dtype=torch.float32, device=device)

    target_notch_mask = (target_s11_np <= float(args.notch_threshold_db))
    target_notch_segments = _find_true_segments(target_notch_mask.astype(bool))

    print(f"🎯 Target notch segments (thr={args.notch_threshold_db} dB): {target_notch_segments}", flush=True)
    print(f"🎯 Found {len(target_notch_segments)} notch segment(s). min_target_notches={args.min_target_notches}", flush=True)

    if len(target_notch_segments) < int(args.min_target_notches):
        raise RuntimeError(
            f"target notch segment count={len(target_notch_segments)} < min_target_notches={args.min_target_notches}"
        )

    if args.forward_model_path is None:
        forward_dir = os.path.join(os.path.dirname(__file__), "..", "forward_surrogate_models")
        args.forward_model_path = _find_default_forward_pth(forward_dir)
        print(f"📌 Auto-selected forward surrogate pth: {args.forward_model_path}", flush=True)

    print(f"📁 Loading forward surrogate from: {args.forward_model_path}", flush=True)
    fwd = DeepCNN10x10(num_points=num_points).to(device)
    _safe_load_state_dict(fwd, args.forward_model_path, device=device)
    fwd.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    forward_loss_dir = os.path.join(args.output_dir, "forward_loss")
    os.makedirs(forward_loss_dir, exist_ok=True)

    candidates_loss_csv = os.path.join(forward_loss_dir, "candidates_loss.csv")
    loss_f = open(candidates_loss_csv, "w", newline="", encoding="utf-8")
    loss_writer = pycsv.writer(loss_f)
    loss_writer.writerow(["candidate_idx", "loss", "overlap_ok", "bw_10dB", "prefix"])

    all_candidates_csv = os.path.join(args.output_dir, "all_candidates.csv")
    all_f = open(all_candidates_csv, "w", newline="", encoding="utf-8")
    all_writer = pycsv.writer(all_f)
    all_writer.writerow(["candidate_idx"] + [f"b{i}" for i in range(num_bits)])

    selected_pixels_csv = os.path.join(args.output_dir, "selected_pixels.csv")
    spectra_dir = os.path.join(args.output_dir, "selected_spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    masks_dir = os.path.join(args.output_dir, "selected_mask")
    os.makedirs(masks_dir, exist_ok=True)

    selected_png_dir = os.path.join(args.output_dir, "selected_png")
    os.makedirs(selected_png_dir, exist_ok=True)

    sel_f = open(selected_pixels_csv, "w", newline="", encoding="utf-8")
    sel_writer = pycsv.writer(sel_f)

    # ============================================================
    # Time-boxed MAB(UCB) on fixed prefix
    # ============================================================
    prefix_len = int(args.prefix_len)
    if prefix_len <= 0:
        raise ValueError("--prefix_len must be > 0")
    if prefix_len > num_bits:
        raise ValueError("--prefix_len must be <= num_bits")

    num_arms = 1 << prefix_len
    warmup = int(args.warmup_pulls_per_arm)
    mab_c = float(args.mab_c)

    pull_batch = int(args.pull_batch)
    sample_micro_batch = int(args.sample_micro_batch)
    if pull_batch <= 0:
        raise ValueError("--pull_batch must be > 0")
    if sample_micro_batch <= 0:
        raise ValueError("--sample_micro_batch must be > 0")

    time_limit = float(args.time_limit_sec)
    max_candidates = int(args.num_candidates)  # upper bound
    loss_clip = float(args.loss_clip)
    penalty = float(args.overlap_fail_penalty)
    suffix_sampling = str(args.suffix_sampling).strip().lower()

    print(f"🧠 Search: MAB(UCB) | prefix_len={prefix_len} -> {num_arms} arms | time_limit_sec={time_limit}", flush=True)
    print(f"🧠 UCB: warmup_pulls_per_arm={warmup}, mab_c={mab_c}", flush=True)
    print(f"🧠 Pull: pull_batch={pull_batch} candidates per arm selection | sample_micro_batch={sample_micro_batch}", flush=True)
    print(f"🧠 Suffix sampling={suffix_sampling}", flush=True)
    print(f"🧠 Reward: r = -clip(loss,{loss_clip}) - (penalty if overlap_ok==0 else 0), penalty={penalty}", flush=True)
    print(f"🧠 max_candidates(upper bound)={max_candidates} (time-box stops earlier)", flush=True)

    N = np.zeros((num_arms,), dtype=np.int64)
    S = np.zeros((num_arms,), dtype=np.float64)
    mu = np.zeros((num_arms,), dtype=np.float64)

    # ✅ tie-break 및 초기 탐색을 위해 RNG 추가(재현성)
    rng = np.random.default_rng(0)

    best_idx_any = None
    best_loss_any = None
    best_pred_any = None
    best_img_any = None
    best_is_selected = False
    best_prefix_str = None

    selected_count = 0
    selected_seen = set()

    t0 = time.perf_counter()
    deadline = t0 + time_limit
    cand_idx = 0

    print_every_sec = 60.0
    last_print_t = t0

    def _reward_from(loss_val: float, overlap_ok: bool) -> float:
        lv = float(loss_val)
        if loss_clip > 0:
            lv = min(lv, loss_clip)
        r = -lv
        if not overlap_ok:
            r -= penalty
        return float(r)

    # ✅✅✅ 핵심 수정: arm 선택이 000으로 고정되지 않도록 "균형 탐색 + 랜덤 tie-break"
    def _select_arm(total_pulls_done: int) -> int:
        # 1) warmup이 있는 경우: N < warmup 인 arm 중에서 "가장 적게 뽑힌 arm"들만 후보로 두고 랜덤 선택
        if warmup > 0:
            need = np.flatnonzero(N < warmup)
            if need.size > 0:
                minN = int(N[need].min())
                cand = need[N[need] == minN]
                return int(rng.choice(cand))

        # 2) warmup이 0이거나 warmup 끝난 뒤: 아직 한 번도 안 뽑힌 arm(N==0)이 있으면 그 중 랜덤 선택
        untried = np.flatnonzero(N == 0)
        if untried.size > 0:
            return int(rng.choice(untried))

        # 3) UCB
        t = max(1, int(total_pulls_done))
        ln_t = math.log(float(t) + 1.0)

        # N>0이 보장됨
        ucb = mu + mab_c * np.sqrt(ln_t / N.astype(np.float64))

        # 4) 동률이면 랜덤 tie-break (np.argmax 고정 방지)
        m = float(ucb.max())
        cand = np.flatnonzero(np.isclose(ucb, m))
        return int(rng.choice(cand))

    def _fmt_time(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(sec - 60 * m)
        return f"{m:02d}m{s:02d}s"

    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                print(f"⏱️ Time limit reached: {time_limit} sec. Stop searching.", flush=True)
                break
            if max_candidates > 0 and cand_idx >= max_candidates:
                print(f"🧷 Reached max_candidates upper bound: {max_candidates}. Stop searching.", flush=True)
                break

            total_pulls_done = int(N.sum())
            arm = _select_arm(total_pulls_done + 1)
            prefix_bits = _arm_to_prefix_bits(arm, prefix_len)
            prefix_str = _prefix_bits_to_str(prefix_bits)

            remain_upper = (max_candidates - cand_idx) if max_candidates > 0 else pull_batch
            want = min(pull_batch, remain_upper)

            bits_chain_batch = ar_sample_with_fixed_prefix_total(
                model=model,
                y_1xP=y_in_t,
                num_bits=num_bits,
                prefix_bits=prefix_bits,
                total_n=want,
                sampling=suffix_sampling,
                micro_batch=sample_micro_batch,
                deadline=deadline
            )

            if bits_chain_batch.shape[0] == 0:
                break

            bits_flat_batch = np.zeros_like(bits_chain_batch, dtype=np.int64)
            bits_flat_batch[:, order_idx] = bits_chain_batch

            if args.canonical_target:
                bits_flat_batch = canonicalize_under_yflip(bits_flat_batch, height=H, width=W)

            X_imgs_batch = restore_pixel_order(bits_flat_batch, H, W)  # (B,1,H,W)

            Btot = X_imgs_batch.shape[0]
            pred_np_all = np.zeros((Btot, num_points), dtype=np.float32)
            loss_np_all = np.zeros((Btot,), dtype=np.float64)

            fwd_batch = int(args.fwd_batch)
            with torch.no_grad():
                for s in range(0, Btot, fwd_batch):
                    if time.perf_counter() >= deadline:
                        Btot = s
                        pred_np_all = pred_np_all[:Btot]
                        loss_np_all = loss_np_all[:Btot]
                        X_imgs_batch = X_imgs_batch[:Btot]
                        break
                    e = min(Btot, s + fwd_batch)
                    xb = torch.tensor(X_imgs_batch[s:e], dtype=torch.float32, device=device)
                    pred_db = fwd(xb)
                    loss_b = loss_function_notch_mse(
                        pred_db_t=pred_db,
                        target_db_t=target_s11_t,
                        notch_threshold_db=float(args.notch_threshold_db),
                    )
                    pred_np_all[s:e] = pred_db.detach().cpu().numpy().astype(np.float32)
                    loss_np_all[s:e] = loss_b.detach().cpu().numpy().astype(np.float64)

                if device.type == "cuda" and args.empty_cache_each_batch:
                    torch.cuda.empty_cache()

            best_reward_in_pull = None

            for j in range(Btot):
                if time.perf_counter() >= deadline:
                    break
                if max_candidates > 0 and cand_idx >= max_candidates:
                    break

                lv = float(loss_np_all[j])
                img_hw = X_imgs_batch[j, 0]
                pixel_row_bt = _pixel_to_row_bottom_to_top(img_hw).reshape(-1).tolist()

                overlap_ok = check_overlap_ok_from_segments(
                    pred_db_1d=pred_np_all[j],
                    target_notch_segments=target_notch_segments,
                    threshold_db=float(args.notch_threshold_db),
                    expand_pts=int(args.overlap_expand_pts),
                )
                overlap_ok_int = 1 if overlap_ok else 0

                bw_10db = compute_bw_below_threshold_ghz(
                    freq_1d=freq_raw,
                    pred_db_1d=pred_np_all[j],
                    threshold_db=float(args.notch_threshold_db)
                )

                loss_writer.writerow([cand_idx, f"{lv:.10f}", overlap_ok_int, f"{bw_10db:.2f}", prefix_str])
                all_writer.writerow([cand_idx] + pixel_row_bt)

                if (best_loss_any is None) or (lv < best_loss_any):
                    best_loss_any = lv
                    best_idx_any = cand_idx
                    best_pred_any = pred_np_all[j].copy()
                    best_img_any = img_hw.copy()
                    best_is_selected = False
                    best_prefix_str = prefix_str

                if overlap_ok:
                    key = tuple(pixel_row_bt)
                    if key not in selected_seen:
                        selected_seen.add(key)
                        selected_count += 1
                        sel_writer.writerow(pixel_row_bt)

                        out_spec = os.path.join(spectra_dir, f"candidate_{selected_count}.csv")
                        import pandas as pd
                        pd.DataFrame({
                            "Freq [GHz]": freq_raw,
                            "dB(S(1,1)) []": pred_np_all[j],
                        }).to_csv(out_spec, index=False)

                        mask_db = np.where(
                            target_s11_np <= float(args.notch_threshold_db),
                            target_s11_np,
                            0.0
                        ).astype(np.float32)

                        out_mask = os.path.join(masks_dir, f"candidate_{selected_count}.csv")
                        pd.DataFrame({
                            "Freq [GHz]": freq_raw,
                            "dB(S(1,1)) []": mask_db,
                        }).to_csv(out_mask, index=False)

                        if args.save_selected_png:
                            out_png = os.path.join(selected_png_dir, f"candidate_{selected_count}_pixel_plus_spectrum.png")
                            title_txt = f"sel#{selected_count} | cand_idx={cand_idx} | prefix={prefix_str} | loss={lv:.2f} | bw@thr={bw_10db:.2f}GHz"
                            save_pixel_plus_spectrum_png(
                                out_png=out_png,
                                pixel_hw=img_hw,
                                freq_1d=freq_raw,
                                pred_db_1d=pred_np_all[j],
                                target_db_1d=target_s11_np if args.plot_target_in_png else None,
                                title=title_txt,
                            )

                        if (best_loss_any is None) or (lv <= best_loss_any + 1e-18):
                            best_loss_any = lv
                            best_idx_any = cand_idx
                            best_pred_any = pred_np_all[j].copy()
                            best_img_any = img_hw.copy()
                            best_is_selected = True
                            best_prefix_str = prefix_str

                r = _reward_from(lv, overlap_ok)
                best_reward_in_pull = r if (best_reward_in_pull is None or r > best_reward_in_pull) else best_reward_in_pull

                cand_idx += 1

            if best_reward_in_pull is not None:
                N[arm] += 1
                S[arm] += float(best_reward_in_pull)
                mu[arm] = S[arm] / float(N[arm])

            now2 = time.perf_counter()
            if (now2 - last_print_t) >= print_every_sec:
                elapsed = now2 - t0
                remaining = max(0.0, deadline - now2)
                cps = (cand_idx / elapsed) if elapsed > 1e-9 else 0.0
                best_arm = int(np.argmax(mu)) if N.sum() > 0 else 0
                best_arm_pref = _prefix_bits_to_str(_arm_to_prefix_bits(best_arm, prefix_len))
                print(
                    f"[MAB] elapsed={_fmt_time(elapsed)} | remaining={_fmt_time(remaining)} | "
                    f"candidates={cand_idx} | cps={cps:.2f}/s | selected={selected_count} | "
                    f"last_arm_prefix={prefix_str} | best_mu_arm={best_arm_pref} mu={mu[best_arm]:.3f} N={N[best_arm]}",
                    flush=True
                )
                last_print_t = now2

    finally:
        sel_f.close()
        all_f.close()
        loss_f.close()

    elapsed = time.perf_counter() - t0
    cps = (cand_idx / elapsed) if elapsed > 1e-9 else 0.0

    print("🎉 Done.", flush=True)
    print(f"⏱️ elapsed_sec = {elapsed:.2f} | elapsed={_fmt_time(elapsed)} | cps={cps:.2f}/s", flush=True)
    print(f"✅ generated candidates = {cand_idx}", flush=True)
    print(f"✅ selected_count(overlap_ok, unique) = {selected_count}", flush=True)
    print(f"💾 selected pixels CSV (NO HEADER): {selected_pixels_csv}", flush=True)
    print(f"💾 all candidates csv: {all_candidates_csv}", flush=True)
    print(f"💾 candidates loss csv: {candidates_loss_csv}", flush=True)

    # ============================================================
    # ✅✅✅ 변경: "appropriate candidate 0"이어도 종료(raise)하지 않음
    #           cand_idx==0도 예외로 죽지 않게 -> DONE.ok 찍고 return
    # ============================================================
    done_ok_path = os.path.join(args.output_dir, "DONE.ok")

    if cand_idx <= 0:
        print("[WARN] No candidates generated (time_limit too small or runtime issue).", flush=True)
        try:
            with open(done_ok_path, "w", encoding="utf-8") as f:
                f.write("DONE (no candidates generated)\n")
        except Exception as e:
            print(f"[WARN] failed to write DONE.ok: {e}", flush=True)
        return

    if best_idx_any is None or best_pred_any is None or best_img_any is None:
        print("[WARN] No BEST candidate found. BEST artifacts skipped.", flush=True)
        try:
            with open(done_ok_path, "w", encoding="utf-8") as f:
                f.write("DONE (no best candidate)\n")
        except Exception as e:
            print(f"[WARN] failed to write DONE.ok: {e}", flush=True)
        return

    best_bw = compute_bw_below_threshold_ghz(
        freq_1d=freq_raw,
        pred_db_1d=best_pred_any,
        threshold_db=float(args.notch_threshold_db),
    )

    if best_prefix_str is None:
        best_prefix_str = "???"

    root_best_png = os.path.join(args.output_dir, "RUN_BEST_pixel_plus_spectrum.png")
    if args.save_best_png:
        save_pixel_plus_spectrum_png(
            out_png=root_best_png,
            pixel_hw=best_img_any,
            freq_1d=freq_raw,
            pred_db_1d=best_pred_any,
            target_db_1d=target_s11_np if args.plot_target_in_png else None,
            title=f"RUN BEST | cand_idx={best_idx_any} | prefix={best_prefix_str} | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
        )

    forward_loss_dir = os.path.join(args.output_dir, "forward_loss")
    best_folder = os.path.join(forward_loss_dir, f"BEST_candidate_idx{best_idx_any}_loss{best_loss_any:.6f}")
    os.makedirs(best_folder, exist_ok=True)

    best_report_png = os.path.join(best_folder, "BEST_report_pixel_plus_spectrum.png")
    if args.save_best_png:
        save_pixel_plus_spectrum_png(
            out_png=best_report_png,
            pixel_hw=best_img_any,
            freq_1d=freq_raw,
            pred_db_1d=best_pred_any,
            target_db_1d=target_s11_np if args.plot_target_in_png else None,
            title=f"BEST REPORT | cand_idx={best_idx_any} | prefix={best_prefix_str} | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
        )

    best_pixel_csv = os.path.join(best_folder, "BEST_pixel.csv")
    best_pixel_row = _pixel_to_row_bottom_to_top(best_img_any).reshape(-1)
    import pandas as pd
    pd.DataFrame([best_pixel_row]).to_csv(best_pixel_csv, index=False, header=False)

    best_spectrum_csv = os.path.join(best_folder, "BEST_spectrum.csv")
    pd.DataFrame({
        "Freq [GHz]": freq_raw,
        "dB(S(1,1)) []": best_pred_any,
    }).to_csv(best_spectrum_csv, index=False)

    print("\n[RUN BEST SAVED]", flush=True)
    print(f" - root_best_png: {root_best_png}", flush=True)
    print(f" - best_folder  : {best_folder}", flush=True)
    print(f" - best_report  : {best_report_png}", flush=True)
    print(f" - best_pixel   : {best_pixel_csv}", flush=True)
    print(f" - best_spectrum: {best_spectrum_csv}", flush=True)
    print(f" - best_is_selected(overlap_ok?) = {best_is_selected}", flush=True)
    print(f" - best_prefix  : {best_prefix_str}", flush=True)

    print("\n[MAB STATS]", flush=True)
    for a in range(num_arms):
        pref = _prefix_bits_to_str(_arm_to_prefix_bits(a, prefix_len))
        print(f" arm {a:02d} prefix={pref} | N={int(N[a])} | mu={float(mu[a]):.4f}", flush=True)

    # ============================================================
    # ✅✅✅ 변경: 완료 표시 파일 생성
    # ============================================================
    try:
        with open(done_ok_path, "w", encoding="utf-8") as f:
            f.write("DONE\n")
            f.write(f"generated_candidates={cand_idx}\n")
            f.write(f"selected_count={selected_count}\n")
            f.write(f"best_idx={best_idx_any}\n")
            f.write(f"best_loss={best_loss_any}\n")
            f.write(f"best_prefix={best_prefix_str}\n")
    except Exception as e:
        print(f"[WARN] failed to write DONE.ok: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--freq_col_name", type=str, default=None)
    parser.add_argument("--s11_col_name", type=str, default=None)

    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)

    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=9)
    parser.add_argument("--dim_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--spectral_cond", type=str, default="resnet1d",
                        choices=["linear", "mlp", "transformer", "resnet1d"])
    parser.add_argument("--ordering", type=str, default="hilbert",
                        choices=["raster", "snake", "hilbert"])
    parser.add_argument("--use_2d_pos", type=lambda x: x.lower() == "true", default=False)

    parser.add_argument("--num_candidates", type=int, default=1000000,
                        help="(upper bound) max candidates; time limit may stop earlier")

    parser.add_argument("--output_dir", type=str, default="./inverse_from_csv_outputs")
    parser.add_argument("--canonical_target", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--normalize_input", action="store_true")

    parser.add_argument("--forward_model_path", type=str, default=None)
    parser.add_argument("--fwd_batch", type=int, default=8)
    parser.add_argument("--gpu_mem_frac", type=float, default=1.0)
    parser.add_argument("--empty_cache_each_batch", action="store_true")

    parser.add_argument("--progress_every", type=int, default=2000)

    parser.add_argument("--save_selected_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--save_best_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--plot_target_in_png", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--notch_threshold_db", type=float, default=-10.0)
    parser.add_argument("--min_target_notches", type=int, default=2)
    parser.add_argument("--overlap_expand_pts", type=int, default=0)

    parser.add_argument("--time_limit_sec", type=int, default=1800)

    parser.add_argument("--prefix_len", type=int, default=3)
    parser.add_argument("--warmup_pulls_per_arm", type=int, default=10)  # 0으로 줘도 잘 동작
    parser.add_argument("--mab_c", type=float, default=1.0)

    parser.add_argument("--pull_batch", type=int, default=500)
    parser.add_argument("--sample_micro_batch", type=int, default=500)

    parser.add_argument("--suffix_sampling", type=str, default="stochastic", choices=["stochastic", "greedy"])
    parser.add_argument("--overlap_fail_penalty", type=float, default=20.0)
    parser.add_argument("--loss_clip", type=float, default=50.0)

    args = parser.parse_args()
    main(args)
