"""
For every Mendeley TEST image, finds its single most similar
Mendeley TRAINING image using the same embed/best_match method
as the project's overlap checks.

Usage:
    python find_similar_test_train.py
    python find_similar_test_train.py --top-n 20
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_overlap import best_match, embed


def class_of(path: str) -> str:
    """Class name is the image's parent folder."""
    return Path(path).parent.name


def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=here / "cache"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "outputs" / "test_train_similarity.csv"
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Print this many of the most similar pairs"
    )
    args = ap.parse_args()

    # Load training and test image caches
    tr = np.load(args.cache_dir / "mendeley_train.npz")
    te = np.load(args.cache_dir / "mendeley_test.npz")

    x_tr, p_tr = tr["x"], tr["paths"]
    x_te, p_te = te["x"], te["paths"]

    # Embed images using the same method as check_overlap.py
    e_tr = embed(x_tr)
    e_te = embed(x_te)

    # Find the closest training image for each test image
    sim, idx = best_match(e_te, e_tr)

    df = pd.DataFrame({
        "test_path": p_te,
        "test_class": [class_of(p) for p in p_te],
        "max_sim": sim,
        "closest_train_path": p_tr[idx],
        "closest_train_class": [
            class_of(p) for p in p_tr[idx]
        ],
    })

    df["same_class"] = (
        df["test_class"] == df["closest_train_class"]
    )

    # Save every test-to-train match, sorted by similarity
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(
        "max_sim", ascending=False
    ).to_csv(args.out, index=False)

    print(
        f"{len(df)} test images checked against "
        f"{len(p_tr)} training images.\n"
    )

    pct = np.percentile(sim, [50, 90, 99])

    print(
        f"Similarity to nearest train image: "
        f"median {pct[0]:.3f} | "
        f"p90 {pct[1]:.3f} | "
        f"p99 {pct[2]:.3f} | "
        f"max {sim.max():.3f}"
    )

    for t in [0.90, 0.95, 0.98, 0.99]:
        n = (sim >= t).sum()
        print(
            f"  >= {t}: {n:4d} "
            f"({n / len(df) * 100:4.1f}%)"
        )

    cross_class = (~df["same_class"]).sum()

    print(
        f"\nNearest train image is a DIFFERENT class for "
        f"{cross_class}/{len(df)} test images "
        f"({cross_class / len(df) * 100:.1f}%)."
    )

    if cross_class > 0:
        print(
            "Inspect high-similarity cross-class pairs carefully "
            "for potential overlap or labeling issues."
        )

    print(f"\nTop {args.top_n} most similar test/train pairs:")

    top = df.sort_values(
        "max_sim", ascending=False
    ).head(args.top_n)

    print(
        top[
            [
                "test_path",
                "test_class",
                "max_sim",
                "closest_train_path",
                "closest_train_class",
                "same_class",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nWrote {args.out} "
        f"({len(df)} pairs, sorted by similarity)."
    )


if __name__ == "__main__":
    main()
