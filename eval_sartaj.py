"""
Evaluates no_aug, baseline, and combined (whichever seeds of each are trained) on SARTAJ, a
third dataset not used anywhere in training. SARTAJ's Testing and Training folders are combined
into ONE evaluation set (training was never done on SARTAJ, so "Training" here is just more
unseen data, same treatment as Kaggle's Training folder elsewhere in this project).

Folder layout expected:
    sartaj/Testing/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}
    sartaj/Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}

This script computes its own SARTAJ-vs-Mendeley-train similarity (same embed/best_match
approach as check_overlap.py) rather than relying on a pre-made CSV of unknown schema/location.
Images with similarity >= --sim-threshold (default 0.90) to their nearest Mendeley train image
are dropped as near-duplicates, same convention as the Kaggle evaluation elsewhere.

SARTAJ's glioma_tumor folder is a known source of label-quality issues in this dataset (widely
reported mislabelling), so every result is reported twice: the full 4-class evaluation, and a
3-class evaluation with glioma excluded (its true-label rows dropped; any stray glioma
predictions on the remaining images still count as plain errors, via sklearn's `labels` filter
rather than dropping those rows too - see compute_metrics_excluding()).

Usage:
    python eval_sartaj.py
    python eval_sartaj.py --sim-threshold 0.90 --arms no_aug baseline combined --seeds 123 456 789
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from check_overlap import best_match, embed
from train_noaug import (CLASSES, IMG_SIZE, NUM_CLASSES, build_model, is_augmented,
                         list_split, predict_logits, preprocess)

SARTAJ_DIRS = {"glioma": "glioma_tumor", "meningioma": "meningioma_tumor",
               "no_tumor": "no_tumor", "pituitary": "pituitary_tumor"}
GLIOMA_IDX = CLASSES.index("glioma")
ARM_DIR = {"no_aug": "noaug"}  # everything else uses its own name as the folder name
SCALAR = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]


def arm_dir(out_root: Path, arm: str) -> Path:
    return out_root / ARM_DIR.get(arm, arm)


# ----------------------------------------------------------------------------
# Load + cache SARTAJ, Testing and Training combined into one set.
# ----------------------------------------------------------------------------
def load_sartaj_combined(data_root: Path, cache_dir: Path, rebuild: bool = False):
    cache_file = cache_dir / "sartaj_all.npz"
    if cache_file.exists() and not rebuild:
        d = np.load(cache_file)
        x, y, paths = d["x"], d["y"], d["paths"]
    else:
        items = []
        for split in ["Testing", "Training"]:
            split_items, _ = list_split(data_root / "sartaj" / split, SARTAJ_DIRS)
            items += split_items
        print(f"[cache] preprocessing sartaj (Testing+Training): {len(items)} images ...", flush=True)
        x = np.empty((len(items), IMG_SIZE, IMG_SIZE), dtype=np.float32)
        y = np.empty(len(items), dtype=np.int64)
        paths = []
        for i, (p, ci) in enumerate(items):
            x[i] = preprocess(p)
            y[i] = ci
            paths.append(str(p))
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(items)}", flush=True)
        paths = np.array(paths)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, x=x, y=y, paths=paths)

    counts = np.bincount(y, minlength=NUM_CLASSES).tolist()
    print(f"[data] sartaj combined: {len(y)} images, per-class {dict(zip(CLASSES, counts))}")
    return torch.from_numpy(x), torch.from_numpy(y), paths


# ----------------------------------------------------------------------------
def compute_metrics_full(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "macro_f2": fbeta_score(y_true, y_pred, beta=2, labels=labels, average="macro", zero_division=0),
        "n": int(len(y_true)),
    }


def compute_metrics_excluding(y_true, y_pred, exclude_idx):
    """Drops rows whose TRUE label is exclude_idx, then scores the remaining classes only
    (via sklearn's `labels` filter). A stray prediction of the excluded class on a kept image
    is NOT dropped - it still counts as a plain misclassification against whatever the true
    class actually was, which is the standard convention for "exclude class X from evaluation"."""
    mask = y_true != exclude_idx
    yt, yp = y_true[mask], y_pred[mask]
    labels = [c for c in range(NUM_CLASSES) if c != exclude_idx]
    return {
        "macro_f1": f1_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(yt, yp),
        "macro_precision": precision_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(yt, yp, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, labels=labels, average="weighted", zero_division=0),
        "macro_f2": fbeta_score(yt, yp, beta=2, labels=labels, average="macro", zero_division=0),
        "n": int(mask.sum()),
    }


def summarize(rows, label):
    df = pd.DataFrame(rows)
    print(f"\n--- {label} ---")
    print(df[["arm", "seed", "n"] + SCALAR].round(4).to_string(index=False))
    agg = df.groupby("arm")[SCALAR].agg(["mean", "std"]).reindex(
        [a for a in ["no_aug", "baseline", "combined"] if a in df["arm"].unique()])
    print(f"\nMean +/- SD across seeds ({label}):")
    for arm in agg.index:
        parts = [f"{m}={agg.loc[arm, (m, 'mean')]:.4f}+/-{agg.loc[arm, (m, 'std')]:.4f}" for m in SCALAR]
        print(f"  {arm:10s} " + "  ".join(parts))
    return df


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out-root", type=Path, default=here / "outputs")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sim-threshold", type=float, default=0.90)
    ap.add_argument("--arms", nargs="+", default=["no_aug", "baseline", "combined"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--csv-out", type=Path, default=here / "outputs" / "sartaj_eval")
    args = ap.parse_args()

    x, y, paths = load_sartaj_combined(args.data_root, args.cache_dir, args.rebuild_cache)

    mtrain_cache = args.cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(f"{mtrain_cache} not found - run train_noaug.py at least once first.")
    e_tr = embed(np.load(mtrain_cache)["x"])
    sim, _ = best_match(embed(x.numpy()), e_tr)
    aug = np.array([is_augmented(p) for p in paths])
    near_dup = sim >= args.sim_threshold
    kept = ~(aug | near_dup)
    print(f"\n[kept] sartaj: {kept.sum()}/{len(paths)} images kept "
          f"(dropped aug={aug.sum()}, near-dup={int((near_dup & ~aug).sum())}, "
          f"sim_threshold={args.sim_threshold})")

    x_kept, y_kept = x[torch.from_numpy(kept)], y[torch.from_numpy(kept)].numpy()
    counts_kept = np.bincount(y_kept, minlength=NUM_CLASSES).tolist()
    print(f"[kept] per-class: {dict(zip(CLASSES, counts_kept))}")

    device = torch.device(args.device)
    rows_full, rows_excl = [], []
    for arm in args.arms:
        for seed in args.seeds:
            ckpt = arm_dir(args.out_root, arm) / f"seed_{seed}" / "best.pt"
            if not ckpt.exists():
                print(f"[skip] {arm}/seed {seed}: no best.pt yet")
                continue
            model = build_model().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            logits = predict_logits(model, x_kept, device)
            pred = logits.argmax(1).numpy()

            m_full = compute_metrics_full(y_kept, pred)
            m_excl = compute_metrics_excluding(y_kept, pred, GLIOMA_IDX)
            rows_full.append({"arm": arm, "seed": seed, **m_full})
            rows_excl.append({"arm": arm, "seed": seed, **m_excl})

    if not rows_full:
        print("\nNo trained checkpoints found for any requested arm/seed - nothing to evaluate.")
        return

    df_full = summarize(rows_full, "SARTAJ, all 4 classes")
    df_excl = summarize(rows_excl, "SARTAJ, glioma excluded (3 classes)")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(args.csv_out.with_name(args.csv_out.name + "_full.csv"), index=False)
    df_excl.to_csv(args.csv_out.with_name(args.csv_out.name + "_no_glioma.csv"), index=False)
    print(f"\nWrote {args.csv_out.with_name(args.csv_out.name + '_full.csv')}")
    print(f"Wrote {args.csv_out.with_name(args.csv_out.name + '_no_glioma.csv')}")


if __name__ == "__main__":
    main()