"""
Detailed evaluation of the two-stage pipeline (binary ResNet-18 Tumor/No-Tumor -> baseline
ResNet-34 seed 123 sub-classification, falling back to 2nd-highest confidence if stage 2
contradicts stage 1 by predicting "no_tumor" on an image stage 1 called Tumor), against the
ResNet-34-alone model, on both Mendeley test and Kaggle.

Unlike eval_twostage.py (the draft this is based on), Kaggle evaluation here uses the same
methodology already established and debugged elsewhere in this project (train_resnet34.py,
eval_sartaj.py, final_analysis.py): cache_dir="cache" (not ".cache"), aug-named duplicates
excluded at cache-build time, Testing+Training combined, and anything near-duplicating a
Mendeley train image (>= --sim-threshold, default 0.90) dropped. The draft script's Kaggle
loading skipped all of this, which would silently reintroduce the contamination this project
spent real effort removing earlier - so that part of it is not reused here. The two-stage
DECISION LOGIC itself (gate -> sub-classify -> fallback) is unchanged from the draft.

Reports, for BOTH models, on BOTH eval sets:
  - a full 4x4 confusion matrix
  - every metric from train_noaug.compute_metrics (macro F1, accuracy, macro precision/recall,
    weighted F1, macro F2, per-class F1)
  - the "tumor predicted as no_tumor" false-negative count/rate specifically - the exact
    failure mode this two-stage pipeline was built to reduce, so it's reported as its own line,
    not left implicit in the confusion matrix

Usage:
    python eval_twostage_detailed.py --device cuda
    python eval_twostage_detailed.py --sim-threshold 0.90
    python eval_twostage_detailed.py --bin-ckpt outputs/binary_resnet18/seed_123/best.pt \
        --r34-ckpt outputs/baseline_resnet34/seed_123/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from check_overlap import best_match, embed
from train_noaug import (CLASSES, KAGGLE_DIRS, MENDELEY_DIRS, NUM_CLASSES, compute_metrics,
                         is_augmented, load_split, to_input)
from train_resnet34 import build_model_r34

NO_TUMOR_IDX = CLASSES.index("no_tumor")
SCALAR = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]


def build_model_binary():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# ----------------------------------------------------------------------------
def load_mendeley_test(data_root: Path, cache_dir: Path):
    x, y, paths = load_split("mendeley_test", data_root / "mendeley" / "test", MENDELEY_DIRS,
                             cache_dir, False)
    return x, y.numpy(), paths


def load_kaggle_combined(data_root: Path, cache_dir: Path, sim_threshold: float):
    """Testing + Training combined, aug-name + near-duplicate (vs Mendeley train) filtered -
    same methodology as train_resnet34.py / eval_sartaj.py / final_analysis.py."""
    x_te, y_te, p_te = load_split("kaggle_test_orig", data_root / "kaggle" / "Testing", KAGGLE_DIRS,
                                  cache_dir, False, exclude_aug=True, cache_name="kaggle_test_orig")
    x_tr, y_tr, p_tr = load_split("kaggle_train_orig", data_root / "kaggle" / "Training", KAGGLE_DIRS,
                                  cache_dir, False, exclude_aug=True, cache_name="kaggle_train_orig")

    mtrain_cache = cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(f"{mtrain_cache} not found - run train_noaug.py at least once first.")
    e_tr = embed(np.load(mtrain_cache)["x"])

    x_all = torch.cat([x_te, x_tr])
    y_all = torch.cat([y_te, y_tr]).numpy()
    p_all = np.concatenate([p_te, p_tr])

    sim, _ = best_match(embed(x_all.numpy()), e_tr)
    aug = np.array([is_augmented(p) for p in p_all])
    near_dup = sim >= sim_threshold
    kept = ~(aug | near_dup)
    print(f"[kaggle] kept {kept.sum()}/{len(p_all)} images "
          f"(dropped aug={aug.sum()}, near-dup={int((near_dup & ~aug).sum())}, "
          f"sim_threshold={sim_threshold})")
    return x_all[torch.from_numpy(kept)], y_all[kept], p_all[kept]


# ----------------------------------------------------------------------------
def predict_twostage(bin_model, r34_model, x, device, bs=64):
    """Same decision logic as the draft eval_twostage.py: returns (r34_only_preds,
    twostage_preds, fallback_count)."""
    r34_only, twostage = [], []
    fallback_count = 0
    for i in range(0, len(x), bs):
        xb = to_input(x[i:i + bs], device)
        with torch.no_grad():
            bin_logits = bin_model(xb)
            r34_logits = r34_model(xb)
        bin_preds = bin_logits.argmax(dim=1)  # 0 = Tumor, 1 = No Tumor
        r34_preds = r34_logits.argmax(dim=1)
        r34_only.extend(r34_preds.cpu().tolist())

        for j in range(len(xb)):
            if bin_preds[j].item() == 1:
                final_pred = NO_TUMOR_IDX
            else:
                pred = r34_preds[j].item()
                if pred == NO_TUMOR_IDX:
                    fallback_count += 1
                    logits = r34_logits[j].clone()
                    logits[NO_TUMOR_IDX] = -float("inf")
                    final_pred = logits.argmax().item()
                else:
                    final_pred = pred
            twostage.append(final_pred)
    return np.array(r34_only), np.array(twostage), fallback_count


def false_negative_rate(y_true, y_pred):
    """Fraction of true-TUMOR images (true != no_tumor) predicted as no_tumor - the exact
    failure mode the two-stage pipeline was built to reduce."""
    tumor_mask = y_true != NO_TUMOR_IDX
    if tumor_mask.sum() == 0:
        return 0.0, 0, 0
    fn = int(((y_pred == NO_TUMOR_IDX) & tumor_mask).sum())
    return fn / tumor_mask.sum(), fn, int(tumor_mask.sum())


def print_confusion_matrix(y_true, y_pred, title):
    cm = pd.DataFrame(
        [[int(((y_true == t) & (y_pred == p)).sum()) for p in range(NUM_CLASSES)]
         for t in range(NUM_CLASSES)],
        index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
    print(f"\n{title} - confusion matrix:")
    print(cm.to_string())


def print_metrics(y_true, y_pred, title):
    m = compute_metrics(y_true, y_pred)
    print(f"\n{title} - metrics:")
    for k in SCALAR:
        print(f"  {k:18s} {m[k]:.4f}")
    print("  per_class_f1       " + ", ".join(f"{c}={m['per_class_f1'][c]:.4f}" for c in CLASSES))
    fnr, fn, n_tumor = false_negative_rate(y_true, y_pred)
    print(f"  tumor->no_tumor FN  {fn}/{n_tumor} ({fnr * 100:.2f}%)")
    return m


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--sim-threshold", type=float, default=0.90)
    ap.add_argument("--bin-ckpt", type=Path, default=None)
    ap.add_argument("--r34-ckpt", type=Path, default=None)
    args = ap.parse_args()
    if args.bin_ckpt is None:
        args.bin_ckpt = args.data_root / "outputs" / "binary_resnet18" / "seed_123" / "best.pt"
    if args.r34_ckpt is None:
        args.r34_ckpt = args.data_root / "outputs" / "baseline_resnet34" / "seed_123" / "best.pt"

    device = torch.device(args.device)
    print("Loading Binary ResNet-18...")
    bin_model = build_model_binary().to(device)
    bin_model.load_state_dict(torch.load(args.bin_ckpt, map_location=device))
    bin_model.eval()

    print("Loading Baseline ResNet-34 (seed 123)...")
    r34_model = build_model_r34().to(device)
    r34_model.load_state_dict(torch.load(args.r34_ckpt, map_location=device))
    r34_model.eval()

    splits = [
        ("Mendeley test", lambda: load_mendeley_test(args.data_root, args.cache_dir)),
        ("Kaggle (filtered)", lambda: load_kaggle_combined(args.data_root, args.cache_dir, args.sim_threshold)),
    ]

    for split_name, loader_fn in splits:
        print(f"\n{'=' * 70}\n{split_name}\n{'=' * 70}")
        x, y_true, paths = loader_fn()
        print(f"n = {len(y_true)}")

        r34_only, twostage, fallback_count = predict_twostage(bin_model, r34_model, x, device)
        print(f"Fallback triggered (stage 1 said Tumor, stage 2 said No Tumor): {fallback_count} times")

        m_base = print_metrics(y_true, r34_only, f"{split_name} - ResNet-34 only")
        print_confusion_matrix(y_true, r34_only, f"{split_name} - ResNet-34 only")

        m_two = print_metrics(y_true, twostage, f"{split_name} - Two-stage pipeline")
        print_confusion_matrix(y_true, twostage, f"{split_name} - Two-stage pipeline")

        print(f"\n{split_name} - summary comparison:")
        header = f"{'Metric':<20} {'R34-only':>12} {'Two-stage':>12}"
        print(header)
        print("-" * len(header))
        for k in SCALAR:
            print(f"{k:<20} {m_base[k]:>12.4f} {m_two[k]:>12.4f}")
        fnr_base, fn_base, n_tumor = false_negative_rate(y_true, r34_only)
        fnr_two, fn_two, _ = false_negative_rate(y_true, twostage)
        print(f"{'tumor->no_tumor FN':<20} {fn_base:>8d}/{n_tumor:<3d}  {fn_two:>8d}/{n_tumor:<3d}")


if __name__ == "__main__":
    main()