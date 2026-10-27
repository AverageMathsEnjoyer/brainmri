"""
Trains one augmentation arm (baseline table 4.2, optionally + one additive augmentation from
table 4.3), across seeds 123/456/789, and evaluates on Mendeley test + Kaggle (same as
train_noaug.py). Val/Mendeley-test/Kaggle are NOT augmented, so they reuse the exact .npz
caches train_noaug.py already built - only the training images are read fresh and augmented
on the fly, every epoch, since augmentation must differ across epochs (so caching them defeats
the point).

Augmentation itself runs on CPU (albumentations/torchio/numpy - no GPU path for these). To stop
CPU-side augmentation from bottlenecking a GPU's fast forward/backward pass, training uses a
DataLoader with num_workers>0 so the next batch is augmented in parallel while the GPU is busy.

Usage (mirrors train_noaug.py):
    python train_aug.py --arm baseline
    python train_aug.py --arm rician_noise
    python train_aug.py --arm rician_noise --seeds 123
    python train_aug.py --arm baseline --num-workers 4 --device cuda
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from augmentations import (ALL_ARM_NAMES, ARMS, ARMS_POST_ZSCORE, _combined_brightness_contrast,
                          build_baseline, build_final_pipeline, build_final_pipeline2,
                          needs_post_zscore)
from check_overlap import best_match, embed
from train_noaug import (BATCH_SIZE, CLASSES, ES_MIN_DELTA, ES_PATIENCE, IMG_SIZE, KAGGLE_DIRS,
                         LR, MAX_EPOCHS, MENDELEY_DIRS, NUM_CLASSES, build_model,
                         build_optimizer, compute_metrics, is_augmented, list_split,
                         load_split, predict_logits, set_seed, to_input)

MENDELEY_TRAIN_EXPECTED = [1534, 1035, 551, 1040]


# ----------------------------------------------------------------------------
# On-the-fly augmented training dataset. Only used for the TRAIN split.
# ----------------------------------------------------------------------------
def percentile_clip_otsu_zscore(x: np.ndarray) -> np.ndarray:
    """Same normalisation core as train_noaug.preprocess(), factored out so both scripts
    apply it identically. Operates on a float array of any augmented values."""
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    try:
        mask = x > threshold_otsu(x)
    except ValueError:
        mask = np.zeros_like(x, dtype=bool)
    if mask.sum() < 2:
        mask = np.ones_like(x, dtype=bool)
    vals = x[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if std < 1e-6:
        std = 1.0
    return (x - mean) / std


def resize_224(x: np.ndarray) -> np.ndarray:
    # antialias=False: cheaper (roughly halves this op's cost on CPU), minor quality trade-off
    # at these up/downscale ratios. Flip back to True if CPU is no longer the bottleneck.
    t = torch.from_numpy(x.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bicubic", align_corners=False, antialias=False)
    return t[0, 0].numpy()


class AugmentedMendeleyTrain(Dataset):
    """Reads each Mendeley train image fresh and re-augments it every __getitem__ call, so
    every epoch sees different augmented versions (this is why train images are never cached)."""

    def __init__(self, items, arm: str):
        self.items = items  # list of (Path, class_index)
        self.arm = arm
        if arm == "combined":
            # The settled pipeline: baseline's geometric/digital steps interleaved with elastic
            # deform/ghosting/blur in a specific order (see augmentations.build_final_pipeline),
            # not simply "baseline then append" like every other arm. Brightness/contrast is
            # still applied separately, after the z-score, same as for every other arm.
            self.pipeline = build_final_pipeline()
            self.baseline = self.extra = None
            self.post = _combined_brightness_contrast  # p=0.3, not the standalone arm's p=0.5
        elif arm == "combined2":
            # Lighter follow-up to combined (halved elastic/ghosting/blur magnitudes, lower
            # blur probability) with brightness/contrast removed entirely - no post-zscore step.
            self.pipeline = build_final_pipeline2()
            self.baseline = self.extra = None
            self.post = None
        else:
            self.pipeline = None
            self.baseline = build_baseline()
            self.extra = ARMS.get(arm)                 # pre-clip extra augmentation, or None
            self.post = ARMS_POST_ZSCORE.get(arm)       # post-z-score extra augmentation, or None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)

        if self.pipeline is not None:
            img = self.pipeline(img)
        else:
            # baseline table 4.2 (skipped entirely for arm == "no_aug"; use train_noaug.py for that)
            img = self.baseline(image=img[..., None])["image"][..., 0]
            # additive arm, applied after baseline, before the percentile clip
            if self.extra is not None:
                img = self.extra(img)

        img = percentile_clip_otsu_zscore(img.astype(np.float32))

        # brightness/contrast is the one arm applied AFTER the z-score instead
        if self.post is not None:
            img = self.post(img)

        img = resize_224(img)
        return torch.from_numpy(img), label


def _profile_dataset(ds, n=40):
    """One-off timing breakdown of the four stages inside __getitem__, averaged over n random
    samples, run single-process in the main thread (bypasses DataLoader workers entirely) so the
    numbers reflect pure per-sample CPU cost. Prints once per run when --profile is passed."""
    import random as _r
    idxs = [_r.randrange(len(ds)) for _ in range(n)]
    t_read = t_base = t_extra = t_norm = t_resize = 0.0
    for i in idxs:
        path, _ = ds.items[i]
        t0 = time.time()
        img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        t_read += time.time() - t0

        t0 = time.time()
        img = ds.baseline(image=img[..., None])["image"][..., 0]
        t_base += time.time() - t0

        t0 = time.time()
        if ds.extra is not None:
            img = ds.extra(img)
        t_extra += time.time() - t0

        t0 = time.time()
        img = percentile_clip_otsu_zscore(img.astype(np.float32))
        if ds.post is not None:
            img = ds.post(img)
        t_norm += time.time() - t0

        t0 = time.time()
        resize_224(img)
        t_resize += time.time() - t0

    total = t_read + t_base + t_extra + t_norm + t_resize
    print(f"\n[profile] per-sample cost over {n} samples (single-process, ms):")
    print(f"    read+grayscale : {1000 * t_read / n:6.2f}")
    print(f"    baseline (4.2) : {1000 * t_base / n:6.2f}")
    print(f"    arm extra      : {1000 * t_extra / n:6.2f}")
    print(f"    clip+Otsu+z    : {1000 * t_norm / n:6.2f}")
    print(f"    resize 224     : {1000 * t_resize / n:6.2f}")
    per_sample = total / n
    print(f"    TOTAL          : {1000 * per_sample:6.2f}  "
          f"(-> ~{per_sample * len(ds):.0f}s/epoch on 1 core, single-process; "
          f"a real run adds DataLoader/collate/GPU-transfer overhead on top)\n", flush=True)


def worker_init_fn(worker_id):
    """Derive each DataLoader worker's seed from torch.initial_seed(), per spec section 7, so
    augmentation RNG doesn't decorrelate/repeat across worker processes."""
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
def run_seed(seed, arm, train_items, eval_data, args, out_root: Path):
    out = out_root / f"seed_{seed}"
    if (out / "results.json").exists() and not args.force:
        print(f"[{arm}/seed {seed}] results.json exists, skipping (use --force to rerun)")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"\n{'=' * 70}\n[{arm}/seed {seed}] starting on {device}\n{'=' * 70}", flush=True)

    set_seed(seed)
    x_va, y_va, _ = eval_data["mendeley_val"]

    labels = [c for _, c in train_items]
    counts = torch.bincount(torch.tensor(labels), minlength=NUM_CLASSES).float()
    class_w = counts.sum() / (NUM_CLASSES * counts)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = AugmentedMendeleyTrain(train_items, arm)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                        persistent_workers=args.num_workers > 0, pin_memory=(device.type == "cuda"))

    if args.profile:
        _profile_dataset(train_ds, n=40)

    best_f1, best_epoch = -1.0, 0
    es_ref, es_bad = -1.0, 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = to_input(xb, device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(yb)
            n_seen += len(yb)
        train_loss = run_loss / n_seen

        val_logits = predict_logits(model, x_va, device)
        val_loss = criterion(val_logits.to(device), y_va.to(device)).item()
        val_pred = val_logits.argmax(1).numpy()
        val_f1 = f1_score(y_va.numpy(), val_pred, average="macro", zero_division=0)
        val_acc = accuracy_score(y_va.numpy(), val_pred)
        scheduler.step(val_f1)
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), out / "best.pt")
        if val_f1 > es_ref + ES_MIN_DELTA:
            es_ref, es_bad = val_f1, 0
        else:
            es_bad += 1

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_macro_f1=val_f1, val_acc=val_acc, lr_fc=lrs[-1],
                            lr_layer4=lrs[-3], epoch_time_s=time.time() - t0))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{arm}/seed {seed}] ep {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_F1 {val_f1:.4f} | val_acc {val_acc:.4f} | best {best_f1:.4f}@{best_epoch} "
              f"| es {es_bad}/{ES_PATIENCE} | lr_fc {lrs[-1]:.1e} | {time.time() - t0:.0f}s", flush=True)

        if es_bad >= ES_PATIENCE:
            print(f"[{arm}/seed {seed}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    results = {"arm": arm, "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
               "best_val_macro_f1": best_f1, "train_time_s": time.time() - t_start,
               "class_weights": dict(zip(CLASSES, class_w.tolist())),
               "sim_threshold": args.sim_threshold, "metrics": {}}

    # Mendeley val/test: no filtering needed, straightforward prediction + metrics.
    for name in ["mendeley_val", "mendeley_test"]:
        x, y, paths = eval_data[name]
        logits = predict_logits(model, x, device)
        probs = F.softmax(logits, dim=1).numpy()
        pred = probs.argmax(1)
        results["metrics"][name] = compute_metrics(y.numpy(), pred)
        df = pd.DataFrame({"path": paths, "true": y.numpy(), "pred": pred})
        for ci, c in enumerate(CLASSES):
            df[f"prob_{c}"] = probs[:, ci]
        df.to_csv(out / f"preds_{name}.csv", index=False)

    # Kaggle: Testing + Training combined, then dropped if aug-named OR near-duplicate of a
    # Mendeley train image (max_sim >= sim_threshold). This is the corrected cross-dataset
    # evaluation (see check_overlap.py); it replaces the old raw/uncorrected kaggle_test.
    kaggle_parts = []
    for name, sim_key, source in [("kaggle_test", "kaggle_test_sim", "kaggle_test"),
                                  ("kaggle_train", "kaggle_train_sim", "kaggle_train")]:
        x, y, paths = eval_data[name]
        logits = predict_logits(model, x, device)
        probs = F.softmax(logits, dim=1).numpy()
        pred = probs.argmax(1)
        df = pd.DataFrame({"path": paths, "true": y.numpy(), "pred": pred,
                           "max_sim": eval_data[sim_key], "source": source})
        for ci, c in enumerate(CLASSES):
            df[f"prob_{c}"] = probs[:, ci]
        df.to_csv(out / f"preds_{name}.csv", index=False)  # includes max_sim, unlike train_noaug.py's
        kaggle_parts.append(df)

    kaggle_all = pd.concat(kaggle_parts, ignore_index=True)
    aug_mask = kaggle_all["path"].map(is_augmented)
    near_dup_mask = kaggle_all["max_sim"] >= args.sim_threshold
    kept = kaggle_all[~(aug_mask | near_dup_mask)]

    results["kaggle_eval_detail"] = {
        "n_total": int(len(kaggle_all)), "n_kept": int(len(kept)),
        "n_kept_by_source": kept["source"].value_counts().to_dict(),
        "n_dropped_aug": int(aug_mask.sum()),
        "n_dropped_near_dup": int((near_dup_mask & ~aug_mask).sum()),
    }
    results["metrics"]["kaggle_test"] = compute_metrics(kept["true"].to_numpy(), kept["pred"].to_numpy())

    scalar = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]
    results["generalisation_gap"] = {
        k: results["metrics"]["mendeley_test"][k] - results["metrics"]["kaggle_test"][k] for k in scalar}

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    d = results["kaggle_eval_detail"]
    print(f"\n[{arm}/seed {seed}] FINAL (best epoch {best_epoch})")
    print(f"   Kaggle eval: kept {d['n_kept']}/{d['n_total']} by source {d['n_kept_by_source']} "
          f"| dropped aug={d['n_dropped_aug']}, near-dup={d['n_dropped_near_dup']} "
          f"(sim_threshold={args.sim_threshold})")
    for name in ["mendeley_test", "kaggle_test"]:
        m = results["metrics"][name]
        print(f"   {name:14s} macroF1 {m['macro_f1']:.4f} | acc {m['accuracy']:.4f}")
    print(f"   gap (macro F1): {results['generalisation_gap']['macro_f1']:.4f}", flush=True)


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ALL_ARM_NAMES,
                    help="'baseline' = table 4.2 only; any other name = baseline + that arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: outputs/<arm>/")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache",
                    help="reused for the (unaugmented) eval sets, same cache as train_noaug.py")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="CPU workers augmenting in parallel with GPU training; 0 = single process. "
                         "On a 2-vCPU host (check with `!nproc`), 1 usually beats 2: with 2 workers "
                         "plus the main process all competing for 2 cores, there is no free core left "
                         "for genuine parallelism.")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="print a one-off per-stage timing breakdown before training starts")
    ap.add_argument("--sim-threshold", type=float, default=0.90,
                    help="drop Kaggle images with max_sim >= this value to their nearest "
                         "Mendeley train image, as near-duplicates (default 0.90)")
    args = ap.parse_args()
    if args.out is None:
        args.out = here / "outputs" / args.arm

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"arm={args.arm} | torch {torch.__version__} | threads={torch.get_num_threads()} "
          f"| num_workers={args.num_workers}")

    m, k = args.data_root / "mendeley", args.data_root / "kaggle"

    # training items are listed fresh each run (paths + labels only - no preprocessing/caching,
    # since every epoch re-augments them from the raw file)
    train_items, _ = list_split(m / "train", MENDELEY_DIRS)
    counts = np.bincount([c for _, c in train_items], minlength=NUM_CLASSES).tolist()
    flag = "" if counts == MENDELEY_TRAIN_EXPECTED else f"   <-- WARNING: expected {MENDELEY_TRAIN_EXPECTED}"
    print(f"[data] mendeley_train (raw, for on-the-fly aug): {len(train_items)} images, "
          f"per-class {counts}{flag}")

    # eval sets: identical to train_noaug.py, so reuse its caches untouched (no augmentation here)
    eval_data = {
        "mendeley_val": load_split("mendeley_val", m / "val", MENDELEY_DIRS, args.cache_dir, False),
        "mendeley_test": load_split("mendeley_test", m / "test", MENDELEY_DIRS, args.cache_dir, False),
        "kaggle_test": load_split("kaggle_test_orig", k / "Testing", KAGGLE_DIRS, args.cache_dir,
                                  False, exclude_aug=True, cache_name="kaggle_test_orig"),
        "kaggle_train": load_split("kaggle_train_orig", k / "Training", KAGGLE_DIRS, args.cache_dir,
                                   False, exclude_aug=True, cache_name="kaggle_train_orig"),
    }

    # Similarity to nearest Mendeley TRAIN image, for both Kaggle sets - model-independent, so
    # computed once here rather than per seed. Requires train_noaug.py to have already built
    # cache/mendeley_train.npz (it's the same cache train_noaug.py's own runs use).
    mtrain_cache = args.cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(
            f"{mtrain_cache} not found. Run train_noaug.py at least once first (even just to build "
            f"caches) so the Mendeley train cache this similarity check needs already exists.")
    e_tr = embed(np.load(mtrain_cache)["x"])
    for name in ["kaggle_test", "kaggle_train"]:
        x, _, _ = eval_data[name]
        sim, _ = best_match(embed(x.numpy()), e_tr)
        eval_data[f"{name}_sim"] = sim
        print(f"[sim] {name} vs mendeley_train: median {np.median(sim):.3f} | "
              f">={args.sim_threshold}: {(sim >= args.sim_threshold).sum()} "
              f"({(sim >= args.sim_threshold).mean() * 100:.1f}%)")

    for seed in args.seeds:
        run_seed(seed, args.arm, train_items, eval_data, args, args.out)


if __name__ == "__main__":
    main()