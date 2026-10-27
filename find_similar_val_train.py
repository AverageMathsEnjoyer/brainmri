"""
For every Mendeley VALIDATION image, finds its single most similar Mendeley TRAINING image
(same embed/best_match method used throughout this project for the Kaggle/SARTAJ overlap
checks: each image downsampled to 32x32, mean-centred, L2-normalised, so the dot product is a
Pearson correlation of the downsampled pixel pattern).

This is primarily a sanity check on Mendeley's own claimed leakage-free 70/15/15 split (the
val-vs-train comparison was already used as a similarity BASELINE in check_overlap.py, to show
what "no leakage" looks like; this script makes that same comparison the main event, with a
full per-image CSV rather than just summary percentiles), and secondarily a look at which
training image the model's "nearest neighbour" for each val image actually is, including
whether that nearest neighbour is even the same class.

Usage:
    python find_similar_val_train.py
    python find_similar_val_train.py --top-n 20       # print the 20 most similar pairs
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_overlap import best_match, embed
from train_noaug import CLASSES


def class_of(path: str) -> str:
    """Class name is the image's parent folder, matching this project's data layout."""
    return Path(path).parent.name


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--out", type=Path, default=here / "outputs" / "val_train_similarity.csv")
    ap.add_argument("--top-n", type=int, default=15, help="print this many of the most similar pairs")
    args = ap.parse_args()

    tr = np.load(args.cache_dir / "mendeley_train.npz")
    va = np.load(args.cache_dir / "mendeley_val.npz")
    x_tr, p_tr = tr["x"], tr["paths"]
    x_va, p_va = va["x"], va["paths"]

    e_tr = embed(x_tr)
    e_va = embed(x_va)
    sim, idx = best_match(e_va, e_tr)  # for each val image: best score + index into train set

    df = pd.DataFrame({
        "val_path": p_va,
        "val_class": [class_of(p) for p in p_va],
        "max_sim": sim,
        "closest_train_path": p_tr[idx],
        "closest_train_class": [class_of(p) for p in p_tr[idx]],
    })
    df["same_class"] = df["val_class"] == df["closest_train_class"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("max_sim", ascending=False).to_csv(args.out, index=False)

    print(f"{len(df)} validation images checked against {len(p_tr)} training images.\n")

    pct = np.percentile(sim, [50, 90, 99])
    print(f"Similarity to nearest train image: median {pct[0]:.3f} | p90 {pct[1]:.3f} | "
          f"p99 {pct[2]:.3f} | max {sim.max():.3f}")
    for t in [0.90, 0.95, 0.98, 0.99]:
        n = (sim >= t).sum()
        print(f"  >= {t}: {n:4d} ({n / len(df) * 100:4.1f}%)")

    cross_class = (~df["same_class"]).sum()
    print(f"\nNearest train image is a DIFFERENT class for {cross_class}/{len(df)} "
          f"val images ({cross_class / len(df) * 100:.1f}%).")
    if cross_class > 0:
        print("(expected at low similarity - two unrelated images can share a class by chance; "
              "only worth investigating if this co-occurs with a HIGH max_sim below)")

    print(f"\nTop {args.top_n} most similar val/train pairs:")
    top = df.sort_values("max_sim", ascending=False).head(args.top_n)
    print(top[["val_path", "val_class", "max_sim", "closest_train_path", "same_class"]]
          .to_string(index=False))

    print(f"\nWrote {args.out} (all {len(df)} pairs, sorted by similarity - open the top rows "
          f"and view the two images side by side to judge whether any are true near-duplicates)")


if __name__ == "__main__":
    main()