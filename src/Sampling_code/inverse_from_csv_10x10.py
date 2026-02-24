# inverse_from_csv_10x10.py
# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

BOS_IDX = 2  # 0,1 bits + 2 BOS token

# ============================================================
# FIXED knobs (as requested)
# ============================================================
GEN_BATCH = 1024   # ✅ fixed
FWD_BATCH = 1024   # ✅ fixed


# ============================================================
# Ordering utilities (raster / snake / hilbert)
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
    N, num_bits = X_flat.shape
    assert num_bits == height * width, f"num_bits({num_bits}) != H*W({height*width})"
    X_restored = X_flat.reshape(N, 1, height, width)
    return X_restored.astype(np.float32)


def horizontal_flip_y_axis(X_flat, height, width):
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    X_flat = X_flat.copy()
    X_flip = horizontal_flip_y_axis(X_flat, height, width)
    N = X_flat.shape[0]
    X_can = np.empty_like(X_flat)
    for i in range(N):
        a = X_flat[i]
        b = X_flip[i]
        X_can[i] = a if list(a) <= list(b) else b
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
# Notch helpers
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


def batch_overlap_ok_from_segments(pred_db_BxP: np.ndarray,
                                  target_notch_segments,
                                  threshold_db: float = -10.0,
                                  expand_pts: int = 0) -> np.ndarray:
    """
    pred_db_BxP: (B,P)
    return: overlap_ok_bool (B,)
    """
    B, P = pred_db_BxP.shape
    if not target_notch_segments:
        return np.ones((B,), dtype=bool)

    thr = float(threshold_db)
    ok = np.ones((B,), dtype=bool)
    for (s, e) in target_notch_segments:
        s2, e2 = _expand_segment(s, e, P, expand_pts)
        seg_min = np.min(pred_db_BxP[:, s2:e2+1], axis=1)  # (B,)
        ok &= (seg_min <= thr)
        if not np.any(ok):
            break
    return ok


def compute_bw_below_threshold_ghz(freq_1d: np.ndarray,
                                  pred_db_1d: np.ndarray,
                                  threshold_db: float = -10.0) -> float:
    below = (pred_db_1d <= float(threshold_db))
    P = below.size
    bw = 0.0
    in_seg = False
    s = 0
    for i in range(P):
        if below[i] and not in_seg:
            in_seg = True
            s = i
        elif (not below[i]) and in_seg:
            e = i - 1
            if e > s:
                bw += float(freq_1d[e] - freq_1d[s])
            in_seg = False
    if in_seg:
        e = P - 1
        if e > s:
            bw += float(freq_1d[e] - freq_1d[s])
    return float(bw)


def loss_notch_mse_batch(pred_db_BxP: torch.Tensor,
                         target_1d: torch.Tensor,
                         notch_threshold_db: float = -10.0) -> torch.Tensor:
    """
    pred_db_BxP: (B,P)
    target_1d:   (P,)
    returns: (B,)
    """
    mask = (target_1d <= float(notch_threshold_db))
    if torch.count_nonzero(mask) == 0:
        return pred_db_BxP.new_zeros((pred_db_BxP.size(0),))
    diff = pred_db_BxP[:, mask] - target_1d[mask].unsqueeze(0)
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
# FAST batch AR sampling (GEN_BATCH at once)
# ============================================================
@torch.no_grad()
def ar_random_sample_batch(model: nn.Module,
                           y_1xP: torch.Tensor,
                           num_bits: int,
                           batch_size: int) -> np.ndarray:
    """
    return: bits_chain (B,num_bits) in {0,1} following chain order
    """
    device = next(model.parameters()).device
    model.eval()

    y_rep = y_1xP.to(device).repeat(batch_size, 1)   # (B,P)
    tokens = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)  # (B,1)

    eps = 1e-9
    for _ in range(num_bits):
        logits = model(y_rep, tokens)     # (B, L)
        next_logit = logits[:, -1]        # (B,)
        probs = torch.sigmoid(next_logit).clamp(eps, 1.0 - eps)
        next_bit = torch.bernoulli(probs).long()  # (B,)
        tokens = torch.cat([tokens, next_bit.unsqueeze(1)], dim=1)

    bits_chain = tokens[:, 1:1+num_bits].detach().cpu().numpy().astype(np.int64)  # (B,num_bits)
    return bits_chain


def _prompt_runtime_minutes(default_min=30) -> int:
    try:
        s = input("Enter runtime (minutes). e.g., 30 : ").strip()
        if s == "":
            return int(default_min)
        v = int(float(s))
        return max(1, v)
    except Exception:
        return int(default_min)


# ============================================================
# ✅ 추가: SKIP/DONE marker helper (최소 변경)
# ============================================================
def _write_marker(output_dir: str, fname: str, lines):
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(str(ln) + "\n")
        return path
    except Exception:
        return None


def _mark_skipped(output_dir: str, reason: str, extra: dict = None):
    lines = ["SKIPPED", f"reason={reason}"]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}={v}")
    p = _write_marker(output_dir, "SKIPPED.ok", lines)
    if p:
        print(f"⚠️ SKIPPED marker written: {p}", flush=True)


def _mark_done(output_dir: str, info: dict = None):
    lines = ["DONE"]
    if info:
        for k, v in info.items():
            lines.append(f"{k}={v}")
    p = _write_marker(output_dir, "DONE.ok", lines)
    if p:
        print(f"✅ DONE marker written: {p}", flush=True)


# ============================================================
# main
# ============================================================
def main(args):
    import csv as pycsv
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🟢 Using device: {device}", flush=True)

    # time limit 결정: runtime_min 우선, 없으면 prompt
    if args.runtime_min is not None:
        time_limit = float(int(args.runtime_min) * 60)
    else:
        m = _prompt_runtime_minutes(default_min=30)
        time_limit = float(m * 60)

    decode_method = str(args.decode_method).strip().lower()
    if decode_method != "sampling":
        _mark_skipped(args.output_dir, "decode_method_not_sampling", {"decode_method": decode_method})
        return

    print(f"⏱️ runtime = {time_limit:.1f} sec  (decode_method=sampling)", flush=True)
    print(f"⚙️ FIXED: GEN_BATCH={GEN_BATCH}, FWD_BATCH={FWD_BATCH}", flush=True)

    # ------------------------------------------------------------
    # CSV load (예외 시 SKIP)
    # ------------------------------------------------------------
    print(f"📄 Loading CSV: {args.csv_path}", flush=True)
    try:
        freq_raw, y_raw = load_freq_and_s11_from_csv(
            args.csv_path,
            freq_col_name=args.freq_col_name,
            s11_col_name=args.s11_col_name
        )
    except Exception as e:
        _mark_skipped(args.output_dir, "csv_load_failed", {"csv_path": args.csv_path, "err": repr(e)})
        return

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

    # ordering 예외 시 SKIP
    try:
        order_idx = get_order_indices(args.ordering, num_bits, H, W)
    except Exception as e:
        _mark_skipped(args.output_dir, "ordering_failed", {"ordering": args.ordering, "err": repr(e)})
        return

    print(f"🔁 Using ordering = {args.ordering}, order_idx[:20] = {order_idx[:20].tolist()}", flush=True)
    chain2spatial = torch.from_numpy(order_idx).long()

    # inverse model load 예외 시 SKIP
    try:
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
    except Exception as e:
        _mark_skipped(args.output_dir, "inverse_model_load_failed", {"model_path": args.model_path, "err": repr(e)})
        return

    target_s11_np = y_raw.astype(np.float32)
    target_s11_t = torch.tensor(target_s11_np, dtype=torch.float32, device=device)

    target_notch_mask = (target_s11_np <= float(args.notch_threshold_db))
    target_notch_segments = _find_true_segments(target_notch_mask.astype(bool))

    print(f"🎯 Target notch segments (thr={args.notch_threshold_db} dB): {target_notch_segments}", flush=True)
    print(f"🎯 Found {len(target_notch_segments)} notch segment(s). min_target_notches={args.min_target_notches}", flush=True)

    # ✅ 여기서 죽지 말고 SKIP
    if len(target_notch_segments) < int(args.min_target_notches):
        _mark_skipped(
            args.output_dir,
            "target_notch_too_few",
            {"found": len(target_notch_segments), "min_target_notches": int(args.min_target_notches)}
        )
        return

    # forward model pth pick/load 예외 시 SKIP
    try:
        if args.forward_model_path is None:
            forward_dir = os.path.join(os.path.dirname(__file__), "..", "forward_surrogate_models")
            args.forward_model_path = _find_default_forward_pth(forward_dir)
            print(f"📌 Auto-selected forward surrogate pth: {args.forward_model_path}", flush=True)

        print(f"📁 Loading forward surrogate from: {args.forward_model_path}", flush=True)
        fwd = DeepCNN10x10(num_points=num_points).to(device)
        _safe_load_state_dict(fwd, args.forward_model_path, device=device)
        fwd.eval()
    except Exception as e:
        _mark_skipped(args.output_dir, "forward_model_load_failed", {"forward_model_path": args.forward_model_path, "err": repr(e)})
        return

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

    max_candidates = int(args.num_candidates)  # upper bound

    best_idx_any = None
    best_loss_any = None
    best_pred_any = None
    best_img_any = None
    best_is_selected = False
    best_prefix_str = "NA"

    selected_count = 0
    selected_seen = set()

    t0 = time.perf_counter()
    deadline = t0 + time_limit
    cand_idx = 0

    def _fmt_time(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(sec - 60 * m)
        return f"{m:02d}m{s:02d}s"

    # ------------------------------------------------------------
    # ✅ sampling loop: 예외 발생해도 죽지 말고 SKIP 처리
    # ------------------------------------------------------------
    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                print(f"⏱️ Time limit reached ({time_limit:.1f}s). Stop sampling.", flush=True)
                break
            if max_candidates > 0 and cand_idx >= max_candidates:
                print(f"🧷 Reached max_candidates upper bound: {max_candidates}. Stop sampling.", flush=True)
                break

            remain = max_candidates - cand_idx if max_candidates > 0 else GEN_BATCH
            B = min(GEN_BATCH, remain)

            bits_chain_BxN = ar_random_sample_batch(
                model=model,
                y_1xP=y_in_t,
                num_bits=num_bits,
                batch_size=B
            )

            bits_flat = np.zeros((B, num_bits), dtype=np.int64)
            bits_flat[:, order_idx] = bits_chain_BxN

            if args.canonical_target:
                bits_flat = canonicalize_under_yflip(bits_flat, height=H, width=W)

            X_imgs = restore_pixel_order(bits_flat, H, W)

            pred_all = np.zeros((B, num_points), dtype=np.float32)
            loss_all = np.zeros((B,), dtype=np.float32)

            with torch.no_grad():
                for s in range(0, B, FWD_BATCH):
                    e = min(B, s + FWD_BATCH)
                    xb = torch.tensor(X_imgs[s:e], dtype=torch.float32, device=device)
                    pred_db = fwd(xb)
                    lv = loss_notch_mse_batch(pred_db, target_s11_t, float(args.notch_threshold_db))
                    pred_all[s:e] = pred_db.detach().cpu().numpy().astype(np.float32)
                    loss_all[s:e] = lv.detach().cpu().numpy().astype(np.float32)

            overlap_ok_bool = batch_overlap_ok_from_segments(
                pred_db_BxP=pred_all,
                target_notch_segments=target_notch_segments,
                threshold_db=float(args.notch_threshold_db),
                expand_pts=int(args.overlap_expand_pts),
            )

            for j in range(B):
                global_idx = cand_idx + j
                pred_np = pred_all[j]
                lv = float(loss_all[j])
                overlap_ok = bool(overlap_ok_bool[j])
                overlap_ok_int = 1 if overlap_ok else 0

                if overlap_ok:
                    bw_10db = compute_bw_below_threshold_ghz(freq_raw, pred_np, float(args.notch_threshold_db))
                else:
                    bw_10db = 0.0

                img_hw = X_imgs[j, 0]
                pixel_row_bt = _pixel_to_row_bottom_to_top(img_hw).reshape(-1).tolist()

                loss_writer.writerow([global_idx, f"{lv:.10f}", overlap_ok_int, f"{bw_10db:.2f}", best_prefix_str])
                all_writer.writerow([global_idx] + pixel_row_bt)

                if (best_loss_any is None) or (lv < best_loss_any):
                    best_loss_any = lv
                    best_idx_any = global_idx
                    best_pred_any = pred_np.copy()
                    best_img_any = img_hw.copy()
                    best_is_selected = False

                if overlap_ok:
                    key = tuple(pixel_row_bt)
                    if key not in selected_seen:
                        selected_seen.add(key)
                        selected_count += 1
                        sel_writer.writerow(pixel_row_bt)

                        out_spec = os.path.join(spectra_dir, f"candidate_{selected_count}.csv")
                        pd.DataFrame({
                            "Freq [GHz]": freq_raw,
                            "dB(S(1,1)) []": pred_np,
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
                            title_txt = f"sel#{selected_count} | cand_idx={global_idx} | method=sampling | loss={lv:.2f} | bw@thr={bw_10db:.2f}GHz"
                            save_pixel_plus_spectrum_png(
                                out_png=out_png,
                                pixel_hw=img_hw,
                                freq_1d=freq_raw,
                                pred_db_1d=pred_np,
                                target_db_1d=target_s11_np if args.plot_target_in_png else None,
                                title=title_txt,
                            )

                        if (best_loss_any is None) or (lv <= best_loss_any + 1e-18):
                            best_loss_any = lv
                            best_idx_any = global_idx
                            best_pred_any = pred_np.copy()
                            best_img_any = img_hw.copy()
                            best_is_selected = True

            cand_idx += B

    except Exception as e:
        # ✅ 어떤 예외든: 이 spec만 SKIP 처리하고 정상 종료
        _mark_skipped(args.output_dir, "runtime_exception", {"err": repr(e)})
        return

    finally:
        try:
            sel_f.close()
            all_f.close()
            loss_f.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    cps = (cand_idx / elapsed) if elapsed > 1e-9 else 0.0

    print("🎉 Done.", flush=True)
    print(f"⏱️ elapsed_sec = {elapsed:.2f} | elapsed={_fmt_time(elapsed)} | cps={cps:.2f}/s", flush=True)
    print(f"✅ generated candidates = {cand_idx}", flush=True)
    print(f"✅ selected_count(overlap_ok, unique) = {selected_count}", flush=True)
    print(f"💾 selected pixels CSV (NO HEADER): {selected_pixels_csv}", flush=True)
    print(f"💾 all candidates csv: {all_candidates_csv}", flush=True)
    print(f"💾 candidates loss csv: {candidates_loss_csv}", flush=True)

    # ✅ cand_idx==0도 죽지 말고 SKIP
    if cand_idx <= 0:
        _mark_skipped(args.output_dir, "no_candidates_generated", {"time_limit_sec": f"{time_limit:.1f}"})
        return

    # ✅ BEST 없음도 죽지 말고 SKIP
    if best_idx_any is None or best_pred_any is None or best_img_any is None:
        print("[WARN] No BEST candidate found. BEST artifacts skipped.", flush=True)
        _mark_skipped(args.output_dir, "no_best_candidate_found", {"generated_candidates": cand_idx})
        return

    best_bw = compute_bw_below_threshold_ghz(freq_raw, best_pred_any, float(args.notch_threshold_db))

    root_best_png = os.path.join(args.output_dir, "RUN_BEST_pixel_plus_spectrum.png")
    if args.save_best_png:
        save_pixel_plus_spectrum_png(
            out_png=root_best_png,
            pixel_hw=best_img_any,
            freq_1d=freq_raw,
            pred_db_1d=best_pred_any,
            target_db_1d=target_s11_np if args.plot_target_in_png else None,
            title=f"RUN BEST | cand_idx={best_idx_any} | method=sampling | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
        )

    best_folder = os.path.join(os.path.join(args.output_dir, "forward_loss"),
                               f"BEST_candidate_idx{best_idx_any}_loss{best_loss_any:.6f}")
    os.makedirs(best_folder, exist_ok=True)

    best_report_png = os.path.join(best_folder, "BEST_report_pixel_plus_spectrum.png")
    if args.save_best_png:
        save_pixel_plus_spectrum_png(
            out_png=best_report_png,
            pixel_hw=best_img_any,
            freq_1d=freq_raw,
            pred_db_1d=best_pred_any,
            target_db_1d=target_s11_np if args.plot_target_in_png else None,
            title=f"BEST REPORT | cand_idx={best_idx_any} | method=sampling | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
        )

    best_pixel_csv = os.path.join(best_folder, "BEST_pixel.csv")
    best_pixel_row = _pixel_to_row_bottom_to_top(best_img_any).reshape(-1)
    pd.DataFrame([best_pixel_row]).to_csv(best_pixel_csv, index=False, header=False)

    best_spectrum_csv = os.path.join(best_folder, "BEST_spectrum.csv")
    pd.DataFrame({"Freq [GHz]": freq_raw, "dB(S(1,1)) []": best_pred_any}).to_csv(best_spectrum_csv, index=False)

    print("\n[RUN BEST SAVED]", flush=True)
    print(f" - root_best_png: {root_best_png}", flush=True)
    print(f" - best_folder  : {best_folder}", flush=True)
    print(f" - best_report  : {best_report_png}", flush=True)
    print(f" - best_pixel   : {best_pixel_csv}", flush=True)
    print(f" - best_spectrum: {best_spectrum_csv}", flush=True)
    print(f" - best_is_selected(overlap_ok?) = {best_is_selected}", flush=True)

    # ✅ DONE 마커 (네가 원하던 방식 유지, helper로도 작성)
    _mark_done(args.output_dir, {
        "best_idx": best_idx_any,
        "best_loss": f"{best_loss_any:.10f}",
        "elapsed_sec": f"{elapsed:.4f}",
        "generated_candidates": cand_idx,
        "selected_count": selected_count,
        "cps": f"{cps:.4f}",
    })


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

    parser.add_argument("--save_selected_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--save_best_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--plot_target_in_png", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--notch_threshold_db", type=float, default=-10.0)
    parser.add_argument("--min_target_notches", type=int, default=2)
    parser.add_argument("--overlap_expand_pts", type=int, default=0)

    parser.add_argument("--runtime_min", type=int, default=None)
    parser.add_argument("--decode_method", type=str, default="sampling", choices=["sampling"])

    args = parser.parse_args()
    main(args)
