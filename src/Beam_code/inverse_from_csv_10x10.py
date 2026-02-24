# inverse_from_csv_10x10.py
# -*- coding: utf-8 -*-
import os
import time
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn

BOS_IDX = 2  # 0,1 bits + 2 BOS token


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

        assert len(coords) == num_bits, f"Hilbert coords length {len(coords)} != num_bits {num_bits}"
        order = np.array([y * width + x for (x, y) in coords], dtype=np.int64)

    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    return order


# ============================================================
# Canonicalization under left-right flip (y-axis symmetry)
# ============================================================
def restore_pixel_order(X_flat, height, width):
    N, num_bits = X_flat.shape
    assert num_bits == height * width
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
        x = self.act(self.input_bn(self.input_proj(y)))
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
            feat = feat + self.freq_pos_embed(pos_idx)
            h = self.cond_transformer(feat)
            return h.mean(dim=1)
        elif self.spectral_cond_type == "resnet1d":
            return self.resnet1d(y.unsqueeze(1))
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
            pos_emb = pos_emb + self.pos2d_embed(spatial_idx)

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
            raise ValueError(f"S11 column not found. Available columns: {list(df.columns)}. Specify --s11_col_name.")
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
    return nn.Sequential(*[ResidualBlock2D(in_ch if i == 0 else out_ch, out_ch) for i in range(n)])


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
# Auto find helper (csv/model/pth)
# ============================================================
def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root_guess():
    d = _script_dir()
    up1 = os.path.dirname(d)
    up2 = os.path.dirname(up1)
    return up2


def _pick_first(patterns):
    files = []
    for p in patterns:
        files += glob.glob(p)
    files = sorted(list(set(files)))
    return files[0] if files else None


def _interactive_pick_path(prompt, default_path):
    print(prompt, flush=True)
    if default_path:
        print(f"  [default] {default_path}", flush=True)
    inp = input("  Enter=use default, or type full path: ").strip()
    if inp == "":
        return default_path
    return inp


def _interactive_pick_int(prompt, default_val):
    print(prompt, flush=True)
    print(f"  [default] {default_val}", flush=True)
    inp = input("  Enter=use default, or type integer: ").strip()
    if inp == "":
        return int(default_val)
    try:
        return int(inp)
    except Exception:
        print("  ⚠️ invalid int. using default.", flush=True)
        return int(default_val)


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
# Time format + moving average
# ============================================================
def _fmt_time_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec - 3600 * h) // 60)
    s = int(sec - 3600 * h - 60 * m)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


class _MovingAverage:
    def __init__(self, window: int = 10):
        self.window = max(1, int(window))
        self.buf = []

    def add(self, x: float):
        self.buf.append(float(x))
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def mean(self):
        if not self.buf:
            return None
        return float(sum(self.buf) / len(self.buf))


# ============================================================
# ✅ Adaptive-K beam (soft target time, NEVER early-stop, MUST finish 100 bits)
# ============================================================
def _clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(x))))


def _k_controller(current_k: int, ratio: float, k_min: int, k_max: int):
    k = int(current_k)

    if ratio <= 0.35:
        factor = 0.25
    elif ratio <= 0.55:
        factor = 0.40
    elif ratio <= 0.75:
        factor = 0.60
    elif ratio <= 0.90:
        factor = 0.80
    elif ratio <= 1.10:
        factor = 1.0 + 0.5 * (ratio - 1.0)
        factor = max(0.92, min(1.08, factor))
    elif ratio <= 1.35:
        factor = 1.12
    elif ratio <= 1.70:
        factor = 1.25
    elif ratio <= 2.30:
        factor = 1.50
    else:
        factor = 2.0

    k_raw = int(round(k * factor))
    k_raw = _clamp_int(k_raw, k_min, k_max)

    if 0.90 < ratio < 1.10:
        alpha = 0.80
    else:
        alpha = 0.60
    k_smooth = int(round(alpha * k + (1 - alpha) * k_raw))
    k_smooth = _clamp_int(k_smooth, k_min, k_max)

    return k_smooth


def beam_search_adaptiveK_softtime(model: nn.Module,
                                  y_1xP: torch.Tensor,
                                  num_bits: int,
                                  beam_size_max: int,
                                  beam_chunk: int,
                                  verbose_every: int,
                                  eta_window: int,
                                  time_limit_sec: int,
                                  k_min: int):
    device = next(model.parameters()).device
    model.eval()

    assert y_1xP.ndim == 2 and y_1xP.size(0) == 1
    y = y_1xP.to(device)

    K_max = int(beam_size_max)
    K_min = max(1, int(k_min))
    K_max = max(K_min, K_max)

    beam_chunk = max(1, int(beam_chunk))
    verbose_every = max(1, int(verbose_every))
    ma_step = _MovingAverage(window=int(eta_window))
    ma_per_item = _MovingAverage(window=int(eta_window))
    eps = 1e-9

    t0 = time.perf_counter()

    bits = np.zeros((1, 0), dtype=np.uint8)
    logp = np.zeros((1,), dtype=np.float64)

    K_target = min(K_max, max(K_min, 1024))

    def _prune_to(bits_arr, logp_arr, K_keep):
        K_now = bits_arr.shape[0]
        if K_now <= K_keep:
            return bits_arr, logp_arr
        idx_keep = np.argpartition(logp_arr, -K_keep)[-K_keep:]
        idx_keep = idx_keep[np.argsort(logp_arr[idx_keep])[::-1]]
        return bits_arr[idx_keep], logp_arr[idx_keep]

    with torch.no_grad():
        for t in range(num_bits):
            step_start = time.perf_counter()

            bits, logp = _prune_to(bits, logp, K_target)
            K = bits.shape[0]

            logp0_all = np.empty((K,), dtype=np.float64)
            logp1_all = np.empty((K,), dtype=np.float64)

            for s in range(0, K, beam_chunk):
                e = min(K, s + beam_chunk)

                tok = np.zeros((e - s, t + 1), dtype=np.int64)
                tok[:, 0] = BOS_IDX
                if t > 0:
                    tok[:, 1:] = bits[s:e].astype(np.int64)

                tok_t = torch.tensor(tok, dtype=torch.long, device=device)
                y_rep = y.repeat(e - s, 1)

                logits = model(y_rep, tok_t)
                next_logit = logits[:, -1]
                probs = torch.sigmoid(next_logit).clamp(eps, 1.0 - eps)

                lp1 = torch.log(probs).detach().cpu().numpy().astype(np.float64)
                lp0 = torch.log(1.0 - probs).detach().cpu().numpy().astype(np.float64)

                logp0_all[s:e] = lp0
                logp1_all[s:e] = lp1

            score0 = logp + logp0_all
            score1 = logp + logp1_all
            scores = np.concatenate([score0, score1], axis=0)

            topk = min(K_target, scores.size)
            idx = np.argpartition(scores, -topk)[-topk:]
            idx = idx[np.argsort(scores[idx])[::-1]]

            new_bits = np.empty((topk, t + 1), dtype=np.uint8)
            new_logp = np.empty((topk,), dtype=np.float64)

            for ii, sel in enumerate(idx):
                if sel < K:
                    parent = sel
                    bit = 0
                    new_logp[ii] = score0[parent]
                else:
                    parent = sel - K
                    bit = 1
                    new_logp[ii] = score1[parent]

                if t > 0:
                    new_bits[ii, :t] = bits[parent]
                new_bits[ii, t] = bit

            bits = new_bits
            logp = new_logp

            step_sec = time.perf_counter() - step_start
            ma_step.add(step_sec)
            ma_per_item.add(step_sec / max(1, bits.shape[0]))

            if int(time_limit_sec) <= 0:
                K_target = K_max
            else:
                elapsed = time.perf_counter() - t0
                remain_steps = max(0, (num_bits - 1) - t)
                remain_time = float(time_limit_sec) - elapsed

                budget_per_step = (remain_time / (remain_steps + 1)) if (remain_steps >= 0) else remain_time
                budget_per_step = max(0.02, float(budget_per_step))

                per_item = ma_per_item.mean()
                if per_item is None:
                    per_item = step_sec / max(1, bits.shape[0])

                pred_step_time = max(1e-6, per_item * float(bits.shape[0]))
                ratio = budget_per_step / pred_step_time

                K_next = _k_controller(
                    current_k=int(bits.shape[0]),
                    ratio=float(ratio),
                    k_min=K_min,
                    k_max=K_max
                )
                K_target = int(K_next)

            if (t % verbose_every) == 0 or (t == num_bits - 1):
                elapsed2 = time.perf_counter() - t0
                sec_per = ma_step.mean()
                sec_per_str = "??" if sec_per is None else f"{sec_per:.3f}"
                if int(time_limit_sec) > 0:
                    diff = elapsed2 - float(time_limit_sec)
                    diff_str = (f"+{_fmt_time_hms(diff)}(over)" if diff > 0 else f"-{_fmt_time_hms(-diff)}(under)")
                    target_str = _fmt_time_hms(float(time_limit_sec))
                else:
                    diff_str = "OFF"
                    target_str = "OFF"

                print(
                    f"[BEAM] t={t:03d}/{num_bits-1:03d} | "
                    f"K={bits.shape[0]:,} -> nextK={int(K_target):,} | "
                    f"elapsed={_fmt_time_hms(elapsed2)} | target={target_str} | drift={diff_str} | "
                    f"sec/step~{sec_per_str}",
                    flush=True
                )

    total = time.perf_counter() - t0
    print(f"[BEAM] DONE | total_elapsed={_fmt_time_hms(total)} | K_final={bits.shape[0]:,}", flush=True)
    return bits.astype(np.int64), logp


# ============================================================
# main
# ============================================================
def main(args):
    import csv as pycsv
    import pandas as pd

    repo_root = _repo_root_guess()

    default_csv = _pick_first([
        os.path.join(repo_root, "specs", "desired", "step_mask", "*.csv"),
        os.path.join(repo_root, "specs", "desired", "step_mask", "**", "*.csv"),
        os.path.join(_script_dir(), "..", "..", "specs", "desired", "step_mask", "*.csv"),
    ])

    default_model = _pick_first([
        os.path.join(repo_root, "src", "transformer_ar_inverse_models_finetune_correct", "*.pth"),
        os.path.join(repo_root, "src", "transformer_ar_inverse_models*", "*.pth"),
        os.path.join(repo_root, "src", "**", "*.pth"),
    ])

    if not args.csv_path:
        args.csv_path = _interactive_pick_path("📄 CSV path (--csv_path) 입력", default_csv)
    if not args.model_path:
        args.model_path = _interactive_pick_path("📁 Inverse model path (--model_path) 입력", default_model)

    if (args.beam_size is None) or (int(args.beam_size) <= 0):
        args.beam_size = _interactive_pick_int("🧠 beam_size (K_max) 입력", 60000)

    if args.forward_model_path is None:
        forward_dir = os.path.join(repo_root, "forward_surrogate_models")
        if os.path.isdir(forward_dir):
            args.forward_model_path = _find_default_forward_pth(forward_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}", flush=True)

    if device.type == "cuda" and args.gpu_mem_frac < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_frac), device=0)
            print(f"🧩 Set GPU memory fraction = {args.gpu_mem_frac}", flush=True)
        except Exception as e:
            print(f"⚠️ set_per_process_memory_fraction failed: {e}", flush=True)

    print(f"📄 Loading CSV: {args.csv_path}", flush=True)
    freq_raw, y_raw = load_freq_and_s11_from_csv(args.csv_path, args.freq_col_name, args.s11_col_name)
    num_points = y_raw.shape[0]
    print(f"✅ Loaded S11 vector with length = {num_points}", flush=True)

    H, W = args.height, args.width
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
        raise FileNotFoundError(
            "forward_model_path not provided AND auto-pick failed. "
            "Provide --forward_model_path or ensure repo_root/forward_surrogate_models exists."
        )

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
    loss_writer.writerow(["candidate_idx", "loss", "overlap_ok", "bw_10dB"])

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

    beam_size_max = int(args.beam_size)
    beam_chunk = int(args.beam_chunk)

    max_candidates = int(args.num_candidates)
    if max_candidates > 0:
        beam_size_max = min(beam_size_max, max_candidates)

    print(f"🧠 Search mode = BEAM(adaptive-K soft-time) | K_max={beam_size_max:,} | beam_chunk={beam_chunk:,}", flush=True)
    print(f"🧠 ETA print: verbose_every={args.verbose_every}, eta_window={args.eta_window}", flush=True)
    print(f"⏱️ time_limit_sec(target) = {args.time_limit_sec} (0=OFF)", flush=True)
    print(f"🧩 k_min = {args.k_min}", flush=True)

    best_idx_any = None
    best_loss_any = None
    best_pred_any = None
    best_img_any = None
    best_is_selected = False

    selected_count = 0
    selected_seen = set()

    t0 = time.perf_counter()
    cand_idx = 0

    try:
        bits_chain, _logp = beam_search_adaptiveK_softtime(
            model=model,
            y_1xP=y_in_t,
            num_bits=num_bits,
            beam_size_max=beam_size_max,
            beam_chunk=beam_chunk,
            verbose_every=int(args.verbose_every),
            eta_window=int(args.eta_window),
            time_limit_sec=int(args.time_limit_sec),
            k_min=int(args.k_min),
        )

        if bits_chain.shape[0] == 0:
            # 이 케이스가 나오면 비정상이라 기존대로 에러 유지
            raise RuntimeError("Beam search generated 0 candidates (unexpected).")

        bits_flat = np.zeros_like(bits_chain, dtype=np.int64)
        bits_flat[:, order_idx] = bits_chain

        if args.canonical_target:
            bits_flat = canonicalize_under_yflip(bits_flat, height=H, width=W)

        X_imgs = restore_pixel_order(bits_flat, H, W)

        Btot = X_imgs.shape[0]
        pred_np_all = np.zeros((Btot, num_points), dtype=np.float32)
        loss_np_all = np.zeros((Btot,), dtype=np.float64)

        fwd_batch = int(args.fwd_batch)
        with torch.no_grad():
            for s in range(0, Btot, fwd_batch):
                e = min(Btot, s + fwd_batch)
                xb = torch.tensor(X_imgs[s:e], dtype=torch.float32, device=device)
                pred_db = fwd(xb)
                loss_b = loss_function_notch_mse(pred_db, target_s11_t, float(args.notch_threshold_db))
                pred_np_all[s:e] = pred_db.detach().cpu().numpy().astype(np.float32)
                loss_np_all[s:e] = loss_b.detach().cpu().numpy().astype(np.float64)
                if device.type == "cuda" and args.empty_cache_each_batch:
                    torch.cuda.empty_cache()

        for j in range(Btot):
            lv = float(loss_np_all[j])
            img_hw = X_imgs[j, 0]
            pixel_row_bt = _pixel_to_row_bottom_to_top(img_hw).reshape(-1).tolist()

            overlap_ok = check_overlap_ok_from_segments(
                pred_db_1d=pred_np_all[j],
                target_notch_segments=target_notch_segments,
                threshold_db=float(args.notch_threshold_db),
                expand_pts=int(args.overlap_expand_pts),
            )
            overlap_ok_int = 1 if overlap_ok else 0
            bw_10db = compute_bw_below_threshold_ghz(freq_raw, pred_np_all[j], float(args.notch_threshold_db))

            loss_writer.writerow([cand_idx, f"{lv:.10f}", overlap_ok_int, f"{bw_10db:.2f}"])
            all_writer.writerow([cand_idx] + pixel_row_bt)

            if (best_loss_any is None) or (lv < best_loss_any):
                best_loss_any = lv
                best_idx_any = cand_idx
                best_pred_any = pred_np_all[j].copy()
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
                        title_txt = f"sel#{selected_count} | cand_idx={cand_idx} | method=beam_adaptiveK_softtime | loss={lv:.2f} | bw@thr={bw_10db:.2f}GHz"
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

            cand_idx += 1

    finally:
        sel_f.close()
        all_f.close()
        loss_f.close()

    elapsed = time.perf_counter() - t0
    cps = (cand_idx / elapsed) if elapsed > 1e-9 else 0.0

    print("🎉 Done.", flush=True)
    print(f"⏱️ elapsed_sec = {elapsed:.2f} | elapsed={_fmt_time_hms(elapsed)} | cps={cps:.2f}/s", flush=True)
    if int(args.time_limit_sec) > 0:
        diff = elapsed - float(args.time_limit_sec)
        print(f"⏱️ target_sec = {int(args.time_limit_sec)} | drift_sec = {diff:+.2f}", flush=True)
    print(f"✅ evaluated candidates = {cand_idx}", flush=True)
    print(f"✅ selected_count(overlap_ok, unique) = {selected_count}", flush=True)
    print(f"💾 selected pixels CSV (NO HEADER): {selected_pixels_csv}", flush=True)
    print(f"💾 all candidates csv: {all_candidates_csv}", flush=True)
    print(f"💾 candidates loss csv: {candidates_loss_csv}", flush=True)

    if cand_idx <= 0:
        # cand_idx==0이면 뭔가 심각하므로 기존대로 에러 유지
        raise RuntimeError("No candidates evaluated (unexpected).")

    if best_idx_any is None or best_pred_any is None or best_img_any is None:
        # 이 케이스도 거의 없지만, 그래도 DONE 마커는 찍고 끝내자 (요구사항: 계속 진행 가능하게)
        print("[WARN] No BEST candidate found. BEST artifacts skipped.", flush=True)

        # ✅ DONE marker (even if no best)
        done_path = os.path.join(args.output_dir, "DONE.ok")
        try:
            with open(done_path, "w", encoding="utf-8") as f:
                f.write("DONE\n")
                f.write("best_idx=NA\n")
                f.write("best_loss=NA\n")
                f.write(f"elapsed_sec={elapsed:.4f}\n")
                f.write(f"evaluated_candidates={cand_idx}\n")
                f.write(f"selected_count={selected_count}\n")
            print(f"✅ DONE marker written: {done_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to write DONE marker: {done_path} | err={e}", flush=True)
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
            title=f"RUN BEST | cand_idx={best_idx_any} | method=beam_adaptiveK_softtime | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
        )

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
            title=f"BEST REPORT | cand_idx={best_idx_any} | method=beam_adaptiveK_softtime | loss={best_loss_any:.2f} | bw@thr={best_bw:.2f}GHz",
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

    # ============================================================
    # ✅ 추가: 완료 마커 생성 (run_repeat에서 skip 판단)
    # ============================================================
    done_path = os.path.join(args.output_dir, "DONE.ok")
    try:
        with open(done_path, "w", encoding="utf-8") as f:
            f.write("DONE\n")
            f.write(f"best_idx={best_idx_any}\n")
            f.write(f"best_loss={best_loss_any:.10f}\n")
            f.write(f"elapsed_sec={elapsed:.4f}\n")
            f.write(f"evaluated_candidates={cand_idx}\n")
            f.write(f"selected_count={selected_count}\n")
        print(f"✅ DONE marker written: {done_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to write DONE marker: {done_path} | err={e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)

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

    parser.add_argument("--beam_size", type=int, default=None,
                        help="K_max upper bound. If omitted, ask interactively.")

    parser.add_argument("--num_candidates", type=int, default=1000000,
                        help="upper bound; effective beam_size=min(beam_size, num_candidates) if >0")

    parser.add_argument("--beam_chunk", type=int, default=10000, help="beam forward chunk size for OOM safety")

    parser.add_argument("--verbose_every", type=int, default=1, help="print every N steps (1=every step)")
    parser.add_argument("--eta_window", type=int, default=10, help="moving average window for sec/step")

    parser.add_argument("--output_dir", type=str, default="./inverse_from_csv_outputs")
    parser.add_argument("--canonical_target", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--normalize_input", action="store_true")

    parser.add_argument("--forward_model_path", type=str, default=None)
    parser.add_argument("--fwd_batch", type=int, default=8)
    parser.add_argument("--gpu_mem_frac", type=float, default=1.0)
    parser.add_argument("--empty_cache_each_batch", action="store_true")

    parser.add_argument("--save_selected_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--save_best_png", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--plot_target_in_png", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--notch_threshold_db", type=float, default=-10.0)
    parser.add_argument("--min_target_notches", type=int, default=2)
    parser.add_argument("--overlap_expand_pts", type=int, default=0)

    parser.add_argument("--time_limit_sec", type=int, default=0,
                        help="soft target total time in seconds. 0=OFF")
    parser.add_argument("--k_min", type=int, default=64,
                        help="minimum K for adaptive beam (too small harms quality)")

    args = parser.parse_args()
    main(args)
