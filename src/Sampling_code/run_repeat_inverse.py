# run_repeat_inverse.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import subprocess
import glob
import csv
import shutil


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def _natural_key(path: str):
    """
    파일명 전체에 대해 숫자 기준 자연정렬 key.
    """
    import re
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)

    tokens = re.split(r"(\d+)", name)
    key = []
    for t in tokens:
        if not t:
            continue
        if t.isdigit():
            key.append(int(t))
        else:
            key.append(t.lower())
    return tuple(key)


def _find_step_dual_csvs(step_mask_dir: str):
    if not os.path.isdir(step_mask_dir):
        raise FileNotFoundError(f"[FATAL] step_mask_dir not found: {step_mask_dir}")

    found = glob.glob(os.path.join(step_mask_dir, "*.csv"))
    found = sorted(found, key=_natural_key)

    if not found:
        raise FileNotFoundError(f"[FATAL] No *.csv found in: {step_mask_dir}")

    return found


def _read_min_loss_from_csv(loss_csv_path):
    best_loss = None
    best_idx = None

    with open(loss_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        if not fieldnames:
            raise ValueError(f"Invalid CSV header: {loss_csv_path}")

        cand_col = None
        loss_col = None
        overlap_col = None

        # exact match 우선
        for c in fieldnames:
            lc = c.lower().strip()
            if lc in ["candidate_idx", "candidate", "idx", "index", "candidate_index"]:
                cand_col = c
            if lc in ["loss", "mse", "score", "value"]:
                if loss_col is None:
                    loss_col = c
            if lc in ["overlap_ok"]:
                overlap_col = c

        # fallback: contains
        if loss_col is None:
            for c in fieldnames:
                lc = c.lower().strip()
                if ("loss" in lc) or ("mse" in lc) or ("score" in lc):
                    loss_col = c
                    break

        if loss_col is None:
            raise ValueError(f"'loss' column not found in: {loss_csv_path} (cols={fieldnames})")

        def _to_bool(v):
            v = str(v).strip().lower()
            if v in ["1", "true", "yes", "y", "t"]:
                return True
            if v in ["0", "false", "no", "n", "f", ""]:
                return False
            try:
                return float(v) != 0.0
            except Exception:
                return False

        row_i = -1
        for row in reader:
            row_i += 1

            if overlap_col is not None:
                if not _to_bool(row.get(overlap_col, "")):
                    continue

            try:
                loss = float(row[loss_col])
            except Exception:
                continue

            if cand_col is None:
                idx = row_i
            else:
                try:
                    idx = int(float(row[cand_col]))
                except Exception:
                    idx = row_i

            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_idx = idx

    if best_loss is None:
        raise ValueError(f"No valid loss values in: {loss_csv_path} (maybe all overlap_ok==0)")

    return best_idx, best_loss


def _find_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            hits.sort()
            return hits[0]
    return None


def _safe_copy(src, dst):
    if src is None or (not os.path.isfile(src)):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


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

        for line in (proc.stdout or []):
            print(line, end="")
            lf.write(line)

        ret = proc.wait()

    return ret


def _replace_arg(cmd_list, key, value):
    if key in cmd_list:
        i = cmd_list.index(key)
        if i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("--"):
            cmd_list[i + 1] = str(value)
        else:
            cmd_list.insert(i + 1, str(value))
    else:
        cmd_list += [key, str(value)]
    return cmd_list


def _prompt_runtime_minutes(default_min=30) -> int:
    try:
        s = input("Enter runtime (minutes). e.g., 30 : ").strip()
        if s == "":
            return int(default_min)
        v = int(float(s))
        return max(1, v)
    except Exception:
        return int(default_min)


def _is_spec_done_or_skipped(out_dir: str) -> bool:
    """
    ✅ 완료/스킵 판단:
      - DONE.ok 있으면 done
      - SKIPPED.ok 있으면 skipped
      - 또는 forward_loss/candidates_loss.csv 존재 + 파일 크기 > 0 이면 done으로 간주
    """
    if not os.path.isdir(out_dir):
        return False

    if os.path.isfile(os.path.join(out_dir, "DONE.ok")):
        return True
    if os.path.isfile(os.path.join(out_dir, "SKIPPED.ok")):
        return True

    loss_csv = os.path.join(out_dir, "forward_loss", "candidates_loss.csv")
    if os.path.isfile(loss_csv) and os.path.getsize(loss_csv) > 0:
        return True

    return False


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--tag", type=str, default="multi_specs", help="(호환성 유지) 사용 안함. result 고정 저장.")
    p.add_argument("--step_mask_dir", type=str, default=None,
                   help="*.csv가 들어있는 폴더. 미지정 시 ../specs/desired/step_mask")
    p.add_argument("--runtime_min", type=int, default=None,
                   help="각 CSV당 random sampling runtime(분). 미지정이면 실행 시 입력받음")
    p.add_argument("--num_candidates", type=int, default=1000000, help="(upper bound) max candidates")

    p.add_argument("--min_target_notches", type=int, default=2, help="dual band=2, single=1")
    p.add_argument("--notch_threshold_db", type=float, default=-10.0, help="notch threshold, default -10 dB")
    p.add_argument("--overlap_expand_pts", type=int, default=0, help="expand pts for overlap check")

    p.add_argument("--model_path", type=str, default=None,
                   help="inverse model pth. 미지정 시 기존 하드코딩 경로 구조로 자동 설정")

    args = p.parse_args()

    if args.runtime_min is None:
        args.runtime_min = _prompt_runtime_minutes(default_min=30)
    args.runtime_min = max(1, int(args.runtime_min))

    base_env = os.environ.copy()
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base_dir, "inverse_from_csv_10x10.py")
    py = sys.executable

    if args.step_mask_dir is None:
        step_mask_dir = os.path.normpath(os.path.join(base_dir, "..", "..", "specs", "desired", "step_mask"))
    else:
        step_mask_dir = os.path.normpath(args.step_mask_dir)

    mask_csvs = _find_step_dual_csvs(step_mask_dir)

    if args.model_path is None:
        model_path = os.path.normpath(os.path.join(
            base_dir,
            "..",
            "transformer_ar_inverse_models_finetune_correct",
            "ar10x10_finetune_correct_L15-L15-d512-h4-ff768-dr0.1-ordhilbert-specresnet1d-2dpos0-canon1-lr1e-05-bs12",
            "best_model.pth"
        ))
    else:
        model_path = os.path.normpath(args.model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"[FATAL] Model not found: {model_path}")

    # ✅ 고정 output root
    out_root = os.path.join(base_dir, "result")
    _safe_makedirs(out_root)

    print("\n" + "=" * 110)
    print(f"[INFO] STEP_MASK_DIR = {step_mask_dir}")
    print(f"[INFO] MODEL_PATH    = {model_path}")
    print(f"[INFO] OUT_ROOT      = {out_root}   (FIXED)")
    print(f"[INFO] runtime_min   = {args.runtime_min}")
    print(f"[INFO] specs(total)  = {len(mask_csvs)}")
    print("=" * 110 + "\n")

    per_spec_best = []
    failed_specs = []

    base_cmd = [
        py, "-u", script,
        "--model_path", model_path,
        "--num_layers", "15",
        "--d_model", "512",

        "--min_target_notches", str(args.min_target_notches),
        "--notch_threshold_db", str(args.notch_threshold_db),
        "--overlap_expand_pts", str(args.overlap_expand_pts),

        "--runtime_min", str(args.runtime_min),
        "--num_candidates", str(args.num_candidates),

        "--decode_method", "sampling",
    ]

    for spec_i, csv_path in enumerate(mask_csvs, start=1):
        csv_base = os.path.basename(csv_path)
        stem = os.path.splitext(csv_base)[0]
        out_dir = os.path.join(out_root, stem)
        _safe_makedirs(out_dir)

        if _is_spec_done_or_skipped(out_dir):
            print("\n" + "-" * 110)
            print(f"[SKIP {spec_i}/{len(mask_csvs)}] {csv_base}")
            print(f"[SKIP] DONE/SKIPPED already. out_dir={out_dir}")
            print("-" * 110)

            loss_csv = _find_first([
                os.path.join(out_dir, "forward_loss", "*candidates_loss*.csv"),
                os.path.join(out_dir, "*candidates_loss*.csv"),
                os.path.join(out_dir, "**", "*candidates_loss*.csv"),
            ])
            if loss_csv is not None:
                try:
                    best_idx, best_loss = _read_min_loss_from_csv(loss_csv)
                    per_spec_best.append({
                        "spec_name": stem,
                        "csv_path": csv_path,
                        "out_dir": out_dir,
                        "loss_csv": loss_csv,
                        "best_candidate_idx": best_idx,
                        "best_loss": best_loss,
                    })
                except Exception as e:
                    print(f"[WARN] Skip-summary read failed: {loss_csv} | err={e}")
            continue

        print("\n" + "=" * 110)
        print(f"[RUN {spec_i}/{len(mask_csvs)}] {csv_base}")
        print(f"[OUT] {out_dir}")
        print("=" * 110)

        cmd = list(base_cmd)
        cmd = _replace_arg(cmd, "--csv_path", csv_path)
        cmd = _replace_arg(cmd, "--output_dir", out_dir)

        log_path = os.path.join(out_dir, "run_log.txt")
        print("[CMD]", " ".join(cmd))
        print(f"[LOG] {log_path}")

        ret = _run_and_tee(cmd, cwd=base_dir, log_path=log_path, env=base_env)

        # ✅ 핵심: 실패해도 STOP 하지 않고 다음 spec으로 넘어감
        if ret != 0:
            print(f"[ERROR] {csv_base} failed with code {ret}. Continue next spec.")
            failed_specs.append((stem, ret))
            continue

        loss_csv = _find_first([
            os.path.join(out_dir, "forward_loss", "*candidates_loss*.csv"),
            os.path.join(out_dir, "*candidates_loss*.csv"),
            os.path.join(out_dir, "**", "*candidates_loss*.csv"),
        ])

        if loss_csv is None:
            print(f"[WARN] No candidates_loss CSV found for {csv_base}. Continue next spec.")
            continue

        try:
            best_idx, best_loss = _read_min_loss_from_csv(loss_csv)
            print(f"[SPEC BEST] {csv_base}: best_candidate_idx={best_idx}, best_loss={best_loss:.6f}")

            per_spec_best.append({
                "spec_name": stem,
                "csv_path": csv_path,
                "out_dir": out_dir,
                "loss_csv": loss_csv,
                "best_candidate_idx": best_idx,
                "best_loss": best_loss,
            })
        except Exception as e:
            print(f"[WARN] BEST read failed for {csv_base}: {e}. Continue next spec.")
            continue

    if not per_spec_best:
        print("\n[WARN] No spec produced readable candidates_loss CSV (maybe all skipped or failed).")
        if failed_specs:
            print("[WARN] failed specs:")
            for s, rc in failed_specs:
                print(f"  - {s}: rc={rc}")
        # FINAL_BEST는 만들 수 없지만, 전체 프로세스는 정상 종료
        return

    final = min(per_spec_best, key=lambda d: d["best_loss"])
    final_dir = os.path.join(out_root, "FINAL_BEST_ACROSS_SPECS")
    _safe_makedirs(final_dir)

    final_out_dir = final["out_dir"]
    final_best_idx = final["best_candidate_idx"]
    final_best_loss = final["best_loss"]

    root_best_png = _find_first([
        os.path.join(final_out_dir, "RUN_BEST_pixel_plus_spectrum.png"),
        os.path.join(final_out_dir, "*_BEST_pixel_plus_spectrum.png"),
        os.path.join(final_out_dir, "**", "*BEST*pixel_plus_spectrum*.png"),
    ])

    best_folder = _find_first([
        os.path.join(final_out_dir, "forward_loss", f"BEST_candidate_idx{final_best_idx}_loss*"),
        os.path.join(final_out_dir, "**", f"BEST_candidate_idx{final_best_idx}_loss*"),
    ])

    best_report_png = None
    best_pixel_csv = None
    best_spectrum_csv = None
    best_candidates_csv_all = _find_first([
        os.path.join(final_out_dir, "all_candidates.csv"),
        os.path.join(final_out_dir, "**", "all_candidates.csv"),
    ])

    if best_folder and os.path.isdir(best_folder):
        best_report_png = _find_first([
            os.path.join(best_folder, "BEST_report_pixel_plus_spectrum.png"),
            os.path.join(best_folder, "*report*.png"),
            os.path.join(best_folder, "*.png"),
        ])
        best_pixel_csv = _find_first([
            os.path.join(best_folder, "BEST_pixel.csv"),
            os.path.join(best_folder, "*pixel*.csv"),
        ])
        best_spectrum_csv = _find_first([
            os.path.join(best_folder, "BEST_spectrum.csv"),
            os.path.join(best_folder, "*spectrum*.csv"),
        ])

    _safe_copy(root_best_png, os.path.join(final_dir, "FINAL_root_best_pixel_plus_spectrum.png"))
    _safe_copy(best_report_png, os.path.join(final_dir, "FINAL_report_pixel_plus_spectrum.png"))
    _safe_copy(best_pixel_csv, os.path.join(final_dir, "FINAL_pixel.csv"))
    _safe_copy(best_spectrum_csv, os.path.join(final_dir, "FINAL_spectrum.csv"))
    _safe_copy(final["loss_csv"], os.path.join(final_dir, "FINAL_candidates_loss.csv"))
    _safe_copy(best_candidates_csv_all, os.path.join(final_dir, "FINAL_all_candidates.csv"))

    summary_txt = os.path.join(final_dir, "FINAL_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"out_root: {out_root}\n")
        f.write(f"final_spec: {final['spec_name']}\n")
        f.write(f"final_csv_path: {final['csv_path']}\n")
        f.write(f"final_out_dir: {final_out_dir}\n")
        f.write(f"best_candidate_idx: {final_best_idx}\n")
        f.write(f"best_loss: {final_best_loss:.10f}\n")
        f.write(f"loss_csv: {final['loss_csv']}\n")
        f.write(f"best_folder: {best_folder}\n")
        f.write(f"root_best_png(src): {root_best_png}\n")
        f.write(f"best_report_png(src): {best_report_png}\n")
        f.write(f"best_pixel_csv(src): {best_pixel_csv}\n")
        f.write(f"best_spectrum_csv(src): {best_spectrum_csv}\n")
        f.write(f"all_candidates_csv(src): {best_candidates_csv_all}\n")

    per_spec_csv = os.path.join(out_root, "per_spec_best_summary.csv")
    with open(per_spec_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["spec_name", "best_candidate_idx", "best_loss", "out_dir", "csv_path", "loss_csv"])
        for d in per_spec_best:
            w.writerow([d["spec_name"], d["best_candidate_idx"], f"{d['best_loss']:.10f}",
                        d["out_dir"], d["csv_path"], d["loss_csv"]])

    print("\n[DONE] Specs processing finished (continue-on-failure enabled).")
    print(f"[DONE] Results saved under: {out_root}")
    print(f"[DONE] Per-spec summary: {per_spec_csv}")
    print(f"[DONE] FINAL BEST across specs: {final_dir}")

    if failed_specs:
        print("\n[WARN] Some specs failed but were skipped and continued:")
        for s, rc in failed_specs:
            print(f"  - {s}: rc={rc}")


if __name__ == "__main__":
    main()
