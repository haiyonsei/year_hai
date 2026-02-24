# run_repeat_inverse.py
# -*- coding: utf-8 -*-
import os
import sys
import glob
import subprocess
from datetime import datetime
import time
import re

# ============================================================
# ✅ [USER CONFIG] 경로/파일은 여기서 끝 (하드코딩)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEP_MASK_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "specs", "desired", "step_mask"))
INVERSE_SCRIPT = os.path.join(BASE_DIR, "inverse_from_csv_10x10.py")

MODEL_PATH = os.path.normpath(os.path.join(
    BASE_DIR,
    "..",
    "transformer_ar_inverse_models_finetune_correct",
    "ar10x10_finetune_correct_L15-L15-d512-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr1e-05-bs12",
    "best_model.pth"
))

# ============================================================
# ✅ [DECODE CONFIG] (훈련과 동일하게 맞춰야 함)
# ============================================================
HEIGHT = 10
WIDTH = 10
ORDERING = "hilbert"         # raster/snake/hilbert
USE_2D_POS = "false"         # "true"/"false"

D_MODEL = 512
NHEAD = 4
NUM_LAYERS = 15
DIM_FF = 768
DROPOUT = 0.1
SPECTRAL_COND = "resnet1d"   # linear/mlp/transformer/resnet1d

NORMALIZE_INPUT = False

# ============================================================
# ✅ [FORWARD EVAL KNOBS]
# ============================================================
FWD_BATCH = 8
GPU_MEM_FRAC = 0.90
EMPTY_CACHE_EACH_BATCH = False

MIN_TARGET_NOTCHES = 2
NOTCH_THRESHOLD_DB = -10.0
OVERLAP_EXPAND_PTS = 0

TAG = "beam_only"

# ============================================================
# helpers
# ============================================================
def _natural_key_general(path: str):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    parts = re.split(r"(\d+)", name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return tuple(key)


def _find_all_csvs(step_mask_dir: str, recursive: bool = False):
    if not os.path.isdir(step_mask_dir):
        raise FileNotFoundError(f"[FATAL] step_mask_dir not found: {step_mask_dir}")

    if recursive:
        pattern = os.path.join(step_mask_dir, "**", "*.csv")
        found = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(step_mask_dir, "*.csv")
        found = glob.glob(pattern)

    found = sorted(list(set(found)), key=_natural_key_general)

    if not found:
        raise FileNotFoundError(f"[FATAL] No *.csv found in: {step_mask_dir}")
    return found


def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec - 3600 * h) // 60)
    s = int(sec - 3600 * h - 60 * m)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _ask_beam_size(default_k=60000) -> int:
    while True:
        s = input(f"\nEnter BEAM SIZE K (Enter=default {default_k}): ").strip()
        if s == "":
            return int(default_k)
        try:
            v = int(float(s))
            if v <= 0:
                print("[WARN] K must be > 0")
                continue
            return v
        except Exception:
            print("[WARN] invalid input. Please enter an integer like 60000.")


def _ask_time_limit_minutes(default_min=30) -> int:
    while True:
        s = input(f"\nEnter TARGET TIME (minutes) (Enter=default {default_min}, 0=OFF): ").strip()
        if s == "":
            return int(default_min)
        try:
            v = int(float(s))
            if v < 0:
                print("[WARN] minutes must be >= 0")
                continue
            return v
        except Exception:
            print("[WARN] invalid input. Please enter an integer like 30 (or 0).")


def _run_and_tee(cmd, cwd, log_path, env=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        ret = proc.wait()
    return ret


def _auto_pick_forward_pth() -> str:
    candidates = []
    candidates.append(os.path.normpath(os.path.join(BASE_DIR, "..", "forward_surrogate_models")))
    candidates.append(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "forward_surrogate_models")))
    candidates.append(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "forward_surrogate_models")))
    candidates.append(os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "..", "forward_surrogate_models")))

    seen = set()
    cand_dirs = []
    for d in candidates:
        if d not in seen:
            seen.add(d)
            cand_dirs.append(d)

    for d in cand_dirs:
        if os.path.isdir(d):
            pths = sorted(glob.glob(os.path.join(d, "*.pth")))
            if pths:
                return os.path.normpath(pths[0])

    deep = sorted(glob.glob(os.path.join(BASE_DIR, "..", "..", "..", "**", "forward_surrogate_models", "*.pth"), recursive=True))
    if deep:
        return os.path.normpath(deep[0])

    raise FileNotFoundError(
        "[FATAL] forward surrogate .pth not found.\n"
        "=> Put your forward .pth under a folder named 'forward_surrogate_models' or set it explicitly in code."
    )


def _safe_dirname_from_csv(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    stem, _ = os.path.splitext(base)

    stem = stem.strip()
    stem = re.sub(r"[\\/:*?\"<>|]", "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    if stem == "":
        stem = "csv"
    return stem


def _is_done(out_dir: str) -> bool:
    return os.path.isfile(os.path.join(out_dir, "DONE.ok"))


def main():
    py = sys.executable

    if not os.path.isfile(INVERSE_SCRIPT):
        raise FileNotFoundError(f"[FATAL] INVERSE_SCRIPT not found: {INVERSE_SCRIPT}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"[FATAL] MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.isdir(STEP_MASK_DIR):
        raise FileNotFoundError(f"[FATAL] STEP_MASK_DIR not found: {STEP_MASK_DIR}")

    forward_pth = _auto_pick_forward_pth()
    print(f"[INFO] FORWARD_PTH   = {forward_pth}")

    mask_csvs = _find_all_csvs(STEP_MASK_DIR, recursive=False)

    K_max = _ask_beam_size(default_k=60000)

    time_limit_min = _ask_time_limit_minutes(default_min=30)
    time_limit_sec = int(time_limit_min * 60)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

    # ============================================================
    # ✅ 변경: result 폴더 "바로 아래"에 저장 (상위 폴더 없음)
    #   result/<csv_stem>/...
    # ============================================================
    result_root = os.path.join(BASE_DIR, "result")
    os.makedirs(result_root, exist_ok=True)

    print("\n" + "=" * 110)
    print("[INFO] STEP_MASK_DIR =", STEP_MASK_DIR)
    print("[INFO] MODEL_PATH    =", MODEL_PATH)
    print("[INFO] FORWARD_PTH   =", forward_pth)
    print("[INFO] INVERSE_SCRIPT=", INVERSE_SCRIPT)
    print("[INFO] RESULT_ROOT   =", result_root)
    print("[INFO] K_max         =", K_max)
    print("[INFO] TARGET_TIME   =", ("OFF" if time_limit_sec <= 0 else f"{time_limit_min} min"))
    print("[INFO] csv_files     =", len(mask_csvs))
    for s in mask_csvs:
        print("  -", os.path.basename(s))
    print("=" * 110)

    total_specs = len(mask_csvs)
    t_all0 = time.perf_counter()
    per_spec_times = []

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    # ============================================================
    # ✅ IMPORTANT:
    #   result 바로 아래 저장이면, 같은 stem 폴더에 덮어쓰게 됨.
    #   그래서 stem 충돌(동일 파일명)만 처리하면 충분.
    #   "이전 결과가 있으면 스킵"은 DONE.ok로 해결.
    # ============================================================
    used_dirnames = set()

    for i, csv_path in enumerate(mask_csvs, start=1):
        csv_base = os.path.basename(csv_path)

        stem = _safe_dirname_from_csv(csv_path)
        dir_name = stem

        # 동일 stem이 "이번 실행 리스트 내부"에 중복인 경우만 suffix
        # (이미 result/<stem>이 있다면 DONE.ok로 스킵되거나, 없으면 이어서 덮어쓰기)
        if dir_name in used_dirnames:
            dup_idx = 1
            while True:
                cand = f"{stem}__dup{dup_idx:04d}"
                if cand not in used_dirnames:
                    dir_name = cand
                    break
                dup_idx += 1
        used_dirnames.add(dir_name)

        out_dir = os.path.join(result_root, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        if _is_done(out_dir):
            print("\n" + "=" * 110)
            print(f"[SKIP {i}/{total_specs}] {csv_base}")
            print("[OUT] ", out_dir)
            print("=> DONE.ok exists. Skipping this spec.")
            print("=" * 110)
            skip_cnt += 1
            continue

        cmd = [
            py, "-u", INVERSE_SCRIPT,
            "--csv_path", csv_path,
            "--model_path", MODEL_PATH,
            "--forward_model_path", forward_pth,
            "--output_dir", out_dir,

            "--beam_size", str(int(K_max)),
            "--time_limit_sec", str(int(time_limit_sec)),

            "--height", str(HEIGHT),
            "--width", str(WIDTH),
            "--ordering", str(ORDERING),
            "--use_2d_pos", str(USE_2D_POS),

            "--d_model", str(D_MODEL),
            "--nhead", str(NHEAD),
            "--num_layers", str(NUM_LAYERS),
            "--dim_ff", str(DIM_FF),
            "--dropout", str(DROPOUT),
            "--spectral_cond", str(SPECTRAL_COND),

            "--min_target_notches", str(MIN_TARGET_NOTCHES),
            "--notch_threshold_db", str(NOTCH_THRESHOLD_DB),
            "--overlap_expand_pts", str(OVERLAP_EXPAND_PTS),

            "--fwd_batch", str(FWD_BATCH),
            "--gpu_mem_frac", str(GPU_MEM_FRAC),
        ]

        if NORMALIZE_INPUT:
            cmd += ["--normalize_input"]
        if EMPTY_CACHE_EACH_BATCH:
            cmd += ["--empty_cache_each_batch"]

        log_path = os.path.join(out_dir, "run_log.txt")

        elapsed_all = time.perf_counter() - t_all0
        if per_spec_times:
            avg = sum(per_spec_times) / len(per_spec_times)
            remain_specs = total_specs - (i - 1)
            eta_all = avg * remain_specs
            eta_str = _fmt_hms(eta_all)
            avg_str = _fmt_hms(avg)
        else:
            eta_str = "??"
            avg_str = "??"

        print("\n" + "=" * 110)
        print(f"[RUN {i}/{total_specs}] {csv_base}")
        print("[OUT]    ", out_dir)
        print("[LOG]    ", log_path)
        print("[K_max]  ", K_max)
        print("[TIME]   ", ("OFF" if time_limit_sec <= 0 else f"{time_limit_min} min"))
        print("[ELAPSED]", _fmt_hms(elapsed_all), "| [AVG/spec]", avg_str, "| [ETA(all)]", eta_str)
        print("[CMD]    ", " ".join(cmd))
        print("=" * 110)

        t0 = time.perf_counter()
        ret = _run_and_tee(cmd, cwd=BASE_DIR, log_path=log_path, env=env)
        dt = time.perf_counter() - t0
        per_spec_times.append(dt)

        if ret != 0:
            print(f"[ERROR] {csv_base} failed (code={ret}). Continue to next spec.")
            fail_cnt += 1
            continue

        # inverse가 정상 종료하면 DONE.ok가 생겨야 함
        if _is_done(out_dir):
            ok_cnt += 1
        else:
            ok_cnt += 1
            print(f"[WARN] {csv_base} returned 0 but DONE.ok not found. (Still continue)")

        print(f"[OK] {csv_base} done. spec_time={_fmt_hms(dt)}")

    total_elapsed = time.perf_counter() - t_all0
    print("\n" + "=" * 110)
    print("[DONE] All csv specs processed.")
    print("[DONE] Results saved under:", result_root)
    print("[DONE] Total elapsed:", _fmt_hms(total_elapsed))
    print(f"[DONE] summary: ok={ok_cnt}, skip={skip_cnt}, fail={fail_cnt}")
    print("=" * 110)


if __name__ == "__main__":
    main()
