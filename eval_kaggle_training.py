"""
Run the (already-trained) reference model(s) on the Kaggle "Training" folder — used only as
extra, unseen cross-dataset evaluation data (never for training). For each seed with a saved
best.pt, this writes outputs/noaug/seed_X/preds_kaggle_train.csv with the same columns as
preds_kaggle_test.csv, plus a max_sim column: each image's max cosine similarity to its closest
Mendeley TRAIN image (same measure as check_overlap.py), so reeval_kaggle_filtered.py can apply
the same aug-name + similarity filtering.

"aug"-named files (e.g. "Tr-aug-me_1.jpg") are excluded up front, same rule as the Testing folder.

Run from the same folder as train_noaug.py, after training has produced at least one seed_*/best.pt:
    python eval_kaggle_training.py
    python eval_kaggle_training.py --seeds 123 456
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from check_overlap import embed, best_match
from train_noaug import (CLASSES, KAGGLE_DIRS, build_model, load_split,
                         predict_logits)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out", type=Path, default=here / "outputs" / "noaug")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="which seed_* dirs to predict with (default: all with a best.pt)")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    # 1. load / cache Kaggle Training (aug-named files excluded, same rule as Testing)
    x_kt, y_kt, p_kt = load_split(
        "kaggle_train_orig", args.data_root / "kaggle" / "Training", KAGGLE_DIRS,
        args.cache_dir, args.rebuild_cache, exclude_aug=True, cache_name="kaggle_train_orig")
    print(f"[data] kaggle_train (unseen eval only): {len(y_kt)} images, "
          f"per class {dict(zip(CLASSES, np.bincount(y_kt.numpy(), minlength=len(CLASSES)).tolist()))}")

    # 2. similarity to closest Mendeley TRAIN image (model-independent, computed once)
    d = np.load(args.cache_dir / "mendeley_train.npz")
    e_tr = embed(d["x"])
    e_kt = embed(x_kt.numpy())
    max_sim, _ = best_match(e_kt, e_tr)
    print(f"[sim] kaggle_train vs mendeley_train: median {np.median(max_sim):.3f} "
          f"| >=0.95: {(max_sim >= 0.95).sum()} ({(max_sim >= 0.95).mean() * 100:.1f}%)")

    # 3. predict with each seed's best model
    device = torch.device(args.device)
    seed_dirs = sorted(args.out.glob("seed_*"))
    if args.seeds is not None:
        wanted = {f"seed_{s}" for s in args.seeds}
        seed_dirs = [d for d in seed_dirs if d.name in wanted]

    for seed_dir in seed_dirs:
        ckpt = seed_dir / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {seed_dir.name}: no best.pt")
            continue
        model = build_model().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logits = predict_logits(model, x_kt, device)
        probs = F.softmax(logits, dim=1).numpy()
        pred = probs.argmax(1)

        df = pd.DataFrame({"path": p_kt, "true": y_kt.numpy(), "pred": pred, "max_sim": max_sim})
        for ci, c in enumerate(CLASSES):
            df[f"prob_{c}"] = probs[:, ci]
        out_csv = seed_dir / "preds_kaggle_train.csv"
        df.to_csv(out_csv, index=False)
        acc = (pred == y_kt.numpy()).mean()
        print(f"[{seed_dir.name}] wrote {out_csv}  (raw accuracy on all {len(df)} images: {acc:.4f})")


if __name__ == "__main__":
    main()