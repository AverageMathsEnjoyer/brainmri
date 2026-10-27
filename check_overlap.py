"""
Near-duplicate check: does Kaggle Testing contain images that (nearly) also appear in Mendeley train?

Uses the cached preprocessed 224x224 arrays (no images re-read). Each image is shrunk to 32x32,
mean-centred and L2-normalised, so the dot product is a Pearson correlation of the downsampled pixels.
Exact / near-exact copies (even re-encoded or resized) score ~0.99+.

Because *any* two brain slices correlate fairly strongly, always compare against the baseline row
"Mendeley val -> Mendeley train" (that dataset claims zero overlap between splits). If Kaggle's numbers
sit clearly above that baseline, you have cross-dataset overlap.

Run from the folder containing train_noaug.py, after the caches exist:
    python check_overlap.py
    python check_overlap.py --seed 123        # which seed's preds to use for the accuracy-by-similarity table
Writes outputs/noaug/overlap_kaggle_vs_mendeley_train.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

THRESHOLDS = [0.90, 0.95, 0.98, 0.99]


def load(cache_dir: Path, name: str):
    d = np.load(cache_dir / f"{name}.npz")
    return d["x"], d["y"], d["paths"]


def embed(x: np.ndarray, size: int = 32) -> torch.Tensor:
    t = torch.from_numpy(x).unsqueeze(1)
    t = F.adaptive_avg_pool2d(t, size).flatten(1)
    t = t - t.mean(1, keepdim=True)
    return t / (t.norm(dim=1, keepdim=True) + 1e-8)


def best_match(q: torch.Tensor, ref: torch.Tensor, chunk: int = 512):
    sims, idx = [], []
    for i in range(0, len(q), chunk):
        m, j = (q[i:i + chunk] @ ref.T).max(1)
        sims.append(m)
        idx.append(j)
    return torch.cat(sims).numpy(), torch.cat(idx).numpy()


def summarise(label: str, sims: np.ndarray):
    pct = np.percentile(sims, [50, 90, 99])
    over = "  ".join(f">{t}: {(sims > t).sum():4d} ({(sims > t).mean() * 100:4.1f}%)" for t in THRESHOLDS)
    print(f"{label:34s} median {pct[0]:.3f}  p90 {pct[1]:.3f}  p99 {pct[2]:.3f}  max {sims.max():.3f} | {over}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--out", type=Path, default=here / "outputs" / "noaug")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    x_tr, y_tr, p_tr = load(args.cache_dir, "mendeley_train")
    x_va, _, p_va = load(args.cache_dir, "mendeley_val")
    x_te, _, p_te = load(args.cache_dir, "mendeley_test")
    x_kg, y_kg, p_kg = load(args.cache_dir, "kaggle_test_orig")

    e_tr, e_va, e_te, e_kg = (embed(a) for a in (x_tr, x_va, x_te, x_kg))

    print("\nMax similarity of each image to its closest Mendeley TRAIN image\n" + "-" * 110)
    s_va, _ = best_match(e_va, e_tr)
    s_te, _ = best_match(e_te, e_tr)
    s_kg, j_kg = best_match(e_kg, e_tr)
    summarise("Mendeley val  -> train (baseline)", s_va)
    summarise("Mendeley test -> train (baseline)", s_te)
    summarise("Kaggle test   -> train", s_kg)

    def rel_key(p):
        # split on both / and \ so it's immune to which OS originally wrote the path
        parts = [x for x in re.split(r"[\\/]+", str(p).strip()) if x]
        return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(p))
        # e.g. "glioma/Te-gl_12.jpg" - stable across machines/sessions/path styles

    df = pd.DataFrame({"kaggle_path": p_kg, "kaggle_key": [rel_key(p) for p in p_kg],
                       "true": y_kg, "max_sim": s_kg, "closest_mendeley_train": p_tr[j_kg]})

    pred_file = args.out / f"seed_{args.seed}" / "preds_kaggle_test.csv"
    if pred_file.exists():
        pr = pd.read_csv(pred_file)[["path", "pred"]].rename(columns={"path": "kaggle_path"})
        df = df.merge(pr, on="kaggle_path", how="left")
        df["correct"] = (df["pred"] == df["true"]).astype(float)
        bins = pd.cut(df["max_sim"], [-1, 0.90, 0.95, 0.98, 1.01],
                      labels=["<0.90", "0.90-0.95", "0.95-0.98", ">=0.98"])
        print(f"\nKaggle accuracy by similarity to Mendeley train (seed {args.seed}):")
        print(df.groupby(bins, observed=True)["correct"].agg(n="size", accuracy="mean").round(4).to_string())
    else:
        print(f"\n[note] {pred_file} not found, skipping accuracy-by-similarity table")

    out_csv = args.out / "overlap_kaggle_vs_mendeley_train.csv"
    df.sort_values("max_sim", ascending=False).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  (sorted by similarity; open the top rows side by side to eyeball them)")


if __name__ == "__main__":
    main()