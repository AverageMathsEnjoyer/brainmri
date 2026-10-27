"""
No-augmentation reference model: ResNet-18 trained on Mendeley, evaluated on
Mendeley test (in-domain) and Kaggle Testing (cross-dataset).

Folder layout (script sits next to both folders):
    mendeley/{train,val,test}/{glioma,meningioma,no_tumor,pituitary}
    kaggle/{Training,Testing}/{glioma,meningioma,notumor,pituitary}   (only Testing is used)

Pipeline (no augmentation, so it is fully deterministic and cached to disk once):
    grayscale (uint8) -> percentile clip 0.5-99.5 -> Otsu-masked z-score -> bicubic resize 224x224

Usage:
    python train_noaug.py                    # seeds 123 456 789, sequentially
    python train_noaug.py --seeds 123        # one seed only
    python train_noaug.py --rebuild-cache    # redo preprocessing
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_otsu
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]  # fixed label order
NUM_CLASSES = len(CLASSES)
MENDELEY_DIRS = {c: c for c in CLASSES}
KAGGLE_DIRS = {"glioma": "glioma", "meningioma": "meningioma",
               "no_tumor": "notumor", "pituitary": "pituitary"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
# Kaggle test set contains augmented copies named e.g. "Te-aug-me_1.jpg" (originals: "Te-me_1.jpg")
AUG_RE = re.compile(r"(^|[-_.])aug([-_.]|$)", re.IGNORECASE)


def is_augmented(path) -> bool:
    return bool(AUG_RE.search(Path(str(path)).stem))
IMG_SIZE = 224

EXPECTED_COUNTS = {
    "mendeley_train": [1534, 1035, 551, 1040],
    "mendeley_val": [329, 222, 118, 223],
    "mendeley_test": [328, 221, 118, 222],
}

# Per-module learning rates
LR = {"conv1": 1e-5, "bn1": 1e-5, "layer1": 1e-5, "layer2": 3e-5,
      "layer3": 1e-4, "layer4": 3e-4, "fc": 1e-3}
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
ES_PATIENCE = 15
ES_MIN_DELTA = 0.003  # absolute macro-F1 (fraction, i.e. 0.3 percentage points)
MAX_EPOCHS = 150


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def preprocess(path: Path) -> np.ndarray:
    """Returns a (224, 224) float32 array."""
    # 1. grayscale collapse -> 2D uint8
    img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    x = img.astype(np.float32)

    # 3. percentile clipping (0.5 - 99.5, per image)
    lo, hi = np.percentile(x, [0.5, 99.5])
    x = np.clip(x, lo, hi)

    # 4. Otsu-masked z-score (mask = x > threshold)
    try:
        mask = x > threshold_otsu(x)
    except ValueError:  # degenerate (constant) image
        mask = np.zeros_like(x, dtype=bool)
    if mask.sum() < 2:  # fall back to whole image if mask is empty/degenerate
        mask = np.ones_like(x, dtype=bool)
    vals = x[mask]
    mean, std = float(vals.mean()), float(vals.std())
    if std < 1e-6:
        std = 1.0
    z = (x - mean) / std  # applied to every pixel

    # 5. bicubic resize to 224x224
    t = torch.from_numpy(z.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bicubic",
                      align_corners=False, antialias=True)
    return t[0, 0].numpy()


def list_split(split_dir: Path, dirmap: dict, exclude_aug: bool = False):
    items, n_excluded = [], []
    for ci, c in enumerate(CLASSES):
        d = split_dir / dirmap[c]
        if not d.is_dir():
            raise FileNotFoundError(f"Missing folder: {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        n_aug = 0
        if exclude_aug:
            kept = [p for p in files if not is_augmented(p)]
            n_aug = len(files) - len(kept)
            files = kept
        n_excluded.append(n_aug)
        items += [(p, ci) for p in files]
    return items, n_excluded


def load_split(name: str, split_dir: Path, dirmap: dict, cache_dir: Path, rebuild: bool,
               exclude_aug: bool = False, cache_name: str = None):
    """Preprocess a split once and cache it as an uncompressed .npz."""
    cache_file = cache_dir / f"{cache_name or name}.npz"
    n_excluded = None
    if cache_file.exists() and not rebuild:
        d = np.load(cache_file)
        x, y, paths = d["x"], d["y"], d["paths"]
        if "n_excluded" in d.files:
            n_excluded = d["n_excluded"].tolist()
    else:
        items, n_excluded = list_split(split_dir, dirmap, exclude_aug)
        print(f"[cache] preprocessing {name}: {len(items)} images ...", flush=True)
        t0 = time.time()
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
        np.savez(cache_file, x=x, y=y, paths=paths, n_excluded=np.array(n_excluded))
        print(f"[cache] {name} done in {time.time() - t0:.0f}s", flush=True)

    counts = np.bincount(y, minlength=NUM_CLASSES).tolist()
    exp = EXPECTED_COUNTS.get(name)
    flag = "" if exp is None or counts == exp else f"   <-- WARNING: expected {exp}"
    print(f"[data] {name}: {len(y)} images, per-class {counts}{flag}")
    if exclude_aug and n_excluded is not None:
        print(f"[data] {name}: excluded augmented files per class {dict(zip(CLASSES, n_excluded))}")
    if any(is_augmented(p) for p in paths) and exclude_aug:
        raise RuntimeError(f"{name} cache still contains augmented files; rerun with --rebuild-cache")
    return torch.from_numpy(x), torch.from_numpy(y), paths


# ----------------------------------------------------------------------------
# Model / optimiser
# ----------------------------------------------------------------------------
def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_optimizer(model):
    groups, n_params = [], 0
    for name, lr in LR.items():
        module = getattr(model, name)
        params = list(module.parameters())
        # conv / fc weights (ndim > 1) get decay; BN weight/bias and all biases (ndim == 1) get none
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        if decay:
            groups.append(dict(params=decay, lr=lr, weight_decay=WEIGHT_DECAY))
        if no_decay:
            groups.append(dict(params=no_decay, lr=lr, weight_decay=0.0))
        n_params += len(params)
    assert n_params == len(list(model.parameters())), "some parameters are not in the optimiser"
    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


# ----------------------------------------------------------------------------
# Eval helpers
# ----------------------------------------------------------------------------
def to_input(xb, device):
    # (B,224,224) -> (B,3,224,224) by replicating the grayscale channel
    return xb.unsqueeze(1).repeat(1, 3, 1, 1).to(device)


@torch.no_grad()
def predict_logits(model, x, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(to_input(x[i:i + bs], device)).cpu())
    return torch.cat(out)


def compute_metrics(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f2": fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0),
        "per_class_f1": dict(zip(CLASSES, f1_score(y_true, y_pred, average=None,
                                                    labels=labels, zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN deterministic intentionally left off (per spec)


# ----------------------------------------------------------------------------
# One full run (one seed)
# ----------------------------------------------------------------------------
def run_seed(seed, data, args, out_root: Path):
    out = out_root / f"seed_{seed}"
    if (out / "results.json").exists() and not args.force:
        print(f"[seed {seed}] results.json exists, skipping (use --force to rerun)")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"\n{'=' * 70}\n[seed {seed}] starting on {device}\n{'=' * 70}", flush=True)

    set_seed(seed)
    x_tr, y_tr, _ = data["mendeley_train"]
    x_va, y_va, _ = data["mendeley_val"]

    # class-weighted CE, inverse class frequency (normalised so weights average ~1)
    counts = torch.bincount(y_tr, minlength=NUM_CLASSES).float()
    class_w = counts.sum() / (NUM_CLASSES * counts)
    print(f"[seed {seed}] class weights: {dict(zip(CLASSES, [round(w, 4) for w in class_w.tolist()]))}")
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    g = torch.Generator()
    g.manual_seed(seed)  # shuffling generator seeded to the run seed
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH_SIZE,
                        shuffle=True, generator=g, num_workers=0)

    best_f1, best_epoch = -1.0, 0        # best model checkpoint (any strict improvement)
    es_ref, es_bad = -1.0, 0             # early-stopping reference (needs >= +0.005 to reset)
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
                            val_macro_f1=val_f1, val_acc=val_acc,
                            lr_fc=lrs[-1],   # last group is fc (no-decay)
                            lr_layer4=lrs[-3],  # layer4 (no-decay) group
                            epoch_time_s=time.time() - t0))
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[seed {seed}] ep {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_F1 {val_f1:.4f} | val_acc {val_acc:.4f} | best {best_f1:.4f}@{best_epoch} "
              f"| es {es_bad}/{ES_PATIENCE} | lr_fc {lrs[-1]:.1e} | {time.time() - t0:.0f}s", flush=True)

        if es_bad >= ES_PATIENCE:
            print(f"[seed {seed}] early stopping at epoch {epoch}")
            break

    # ---- final evaluation with the best (by val macro F1) checkpoint ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    results = {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
               "best_val_macro_f1": best_f1, "train_time_s": time.time() - t_start,
               "class_weights": dict(zip(CLASSES, class_w.tolist())), "metrics": {}}

    for name in ["mendeley_val", "mendeley_test", "kaggle_test"]:
        x, y, paths = data[name]
        logits = predict_logits(model, x, device)
        probs = F.softmax(logits, dim=1).numpy()
        pred = probs.argmax(1)
        results["metrics"][name] = compute_metrics(y.numpy(), pred)
        df = pd.DataFrame({"path": paths, "true": y.numpy(), "pred": pred})
        for ci, c in enumerate(CLASSES):
            df[f"prob_{c}"] = probs[:, ci]
        df.to_csv(out / f"preds_{name}.csv", index=False)  # per-image preds for paired bootstrap

    scalar = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]
    results["generalisation_gap"] = {
        k: results["metrics"]["mendeley_test"][k] - results["metrics"]["kaggle_test"][k] for k in scalar}

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[seed {seed}] FINAL (best epoch {best_epoch})")
    for name in ["mendeley_test", "kaggle_test"]:
        m = results["metrics"][name]
        print(f"   {name:14s} macroF1 {m['macro_f1']:.4f} | acc {m['accuracy']:.4f} | "
              f"prec {m['macro_precision']:.4f} | rec {m['macro_recall']:.4f} | "
              f"wF1 {m['weighted_f1']:.4f} | F2 {m['macro_f2']:.4f}")
    print(f"   gap (macro F1): {results['generalisation_gap']['macro_f1']:.4f}", flush=True)


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out", type=Path, default=here / "outputs" / "noaug")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun seeds that already have results.json")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")

    m, k = args.data_root / "mendeley", args.data_root / "kaggle"
    rb = args.rebuild_cache
    data = {
        "mendeley_train": load_split("mendeley_train", m / "train", MENDELEY_DIRS, args.cache_dir, rb),
        "mendeley_val": load_split("mendeley_val", m / "val", MENDELEY_DIRS, args.cache_dir, rb),
        "mendeley_test": load_split("mendeley_test", m / "test", MENDELEY_DIRS, args.cache_dir, rb),
        "kaggle_test": load_split("kaggle_test", k / "Testing", KAGGLE_DIRS, args.cache_dir, rb,
                                  exclude_aug=True, cache_name="kaggle_test_orig"),  # Training/ unused; augmented copies dropped
    }

    for seed in args.seeds:
        run_seed(seed, data, args, args.out)


if __name__ == "__main__":
    main()