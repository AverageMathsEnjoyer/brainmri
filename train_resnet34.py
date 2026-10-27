"""
Any augmentation arm from augmentations.py/train_aug.py (baseline, combined, combined2,
elastic_deform2, etc. - pass --arm), but on ResNet-34 instead of ResNet-18, across seeds
123/456/789.

ResNet-34 shares ResNet-18's top-level module names (conv1, bn1, layer1..layer4, fc) - it just
has more BasicBlocks per layer (~21M params vs ~11M). Given the training set is unchanged
(~4160 images), the larger capacity raises overfitting risk relative to ResNet-18, so this
script makes one architectural change beyond swapping the backbone:

    conv1, bn1, and layer1 are FROZEN: both their weights (requires_grad=False, set in
    build_model_r34) AND their BatchNorm running statistics (running_mean/running_var - these
    are buffers, not parameters, and update via a running average on every forward pass whenever
    a module is in .train() mode, independent of requires_grad; see run_seed()'s epoch loop,
    which re-applies .eval() to this trunk every epoch since model.train() would otherwise reset
    it). Leaving running stats to drift while gamma/beta stay pinned at their ImageNet-calibrated
    values would decouple two things that were jointly trained together on ImageNet, which can
    corrupt the features this trunk produces - so both halves are frozen together.

    These layers capture generic low-level features (edges, local contrast) that transfer well
    regardless of domain and are the least likely to encode anything tumor-specific, so there is
    little to gain from fine-tuning them - freezing directly reduces the larger model's
    overfitting risk on a dataset of unchanged size.

layer2/layer3/layer4/fc keep the exact same per-tier learning rates as the ResNet-18 scheme
(3e-5 / 1e-4 / 3e-4 / 1e-3) - same discriminative schedule, just missing its bottom (now frozen)
tier, with fc's rate/decay specifically kept unchanged as requested.

Deliberately a separate script rather than a flag added to train_aug.py/train_noaug.py: those
assume a fixed conv1/bn1/layer1/layer2/layer3/layer4/fc LR-group structure with no frozen tier.
This script now takes --arm so every ResNet-34 run shares one copy of the freeze/BatchNorm-fix/
training-loop logic instead of drifting across several near-duplicate files, one per arm.

Reuses, unmodified, via import: AugmentedMendeleyTrain and worker_init_fn from train_aug.py
(architecture-agnostic - doesn't know or care what model consumes its output, and already
handles any arm name correctly, including the special-cased combined/combined2 pipelines), and
data loading / metrics / eval-set caching from train_noaug.py and check_overlap.py.

Usage:
    python train_resnet34.py --arm elastic_deform2 --device cuda
    python train_resnet34.py --arm combined2 --seeds 123 --num-workers 4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights, resnet34

from augmentations import ALL_ARM_NAMES
from check_overlap import best_match, embed
from train_aug import AugmentedMendeleyTrain, worker_init_fn
from train_noaug import (BATCH_SIZE, CLASSES, ES_MIN_DELTA, ES_PATIENCE, KAGGLE_DIRS,
                         MAX_EPOCHS, MENDELEY_DIRS, NUM_CLASSES, WEIGHT_DECAY, compute_metrics,
                         is_augmented, list_split, load_split, predict_logits, set_seed, to_input)

R34_LR = {"layer2": 3e-5, "layer3": 1e-4, "layer4": 3e-4, "fc": 1e-3}  # conv1/bn1/layer1: frozen


# ----------------------------------------------------------------------------
def build_model_r34():
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    for name in ["conv1", "bn1", "layer1"]:
        for p in getattr(model, name).parameters():
            p.requires_grad = False
    return model


def build_optimizer_r34(model):
    groups, n_params = [], 0
    for name, lr in R34_LR.items():
        module = getattr(model, name)
        params = list(module.parameters())
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        if decay:
            groups.append(dict(params=decay, lr=lr, weight_decay=WEIGHT_DECAY))
        if no_decay:
            groups.append(dict(params=no_decay, lr=lr, weight_decay=0.0))
        n_params += len(params)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert n_params == len(trainable), (
        f"optimizer covers {n_params} params but {len(trainable)} are trainable - "
        f"a module's parameters are missing from R34_LR or the freeze list")
    return AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


# ----------------------------------------------------------------------------
def run_seed(seed, arm, train_items, eval_data, args, out_root: Path):
    out = out_root / f"seed_{seed}"
    if (out / "results.json").exists() and not args.force:
        print(f"[r34-{arm}/seed {seed}] results.json exists, skipping (use --force to rerun)")
        return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"\n{'=' * 70}\n[r34-{arm}/seed {seed}] starting on {device}\n{'=' * 70}", flush=True)

    set_seed(seed)
    x_va, y_va, _ = eval_data["mendeley_val"]

    labels = [c for _, c in train_items]
    counts = torch.bincount(torch.tensor(labels), minlength=NUM_CLASSES).float()
    class_w = counts.sum() / (NUM_CLASSES * counts)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))

    model = build_model_r34().to(device)
    optimizer = build_optimizer_r34(model)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = AugmentedMendeleyTrain(train_items, arm)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                        num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                        persistent_workers=args.num_workers > 0, pin_memory=(device.type == "cuda"))

    best_f1, best_epoch = -1.0, 0
    es_ref, es_bad = -1.0, 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        # model.train() resets EVERY submodule to training-mode behaviour, including the frozen
        # trunk. requires_grad=False (set in build_model_r34) stops conv1/bn1/layer1's weights
        # from updating via backprop, but does NOT stop bn1/layer1's BatchNorm running_mean/
        # running_var buffers from drifting - those update via a running average computed every
        # forward pass, independent of gradients, whenever the module is in .train() mode. Left
        # alone, this would let the running stats adapt to MRI data while gamma/beta stay pinned
        # at their original ImageNet-calibrated values - a mismatched pair that was never jointly
        # trained together, which can corrupt the features this "frozen" trunk produces. So the
        # frozen submodules are explicitly re-set to eval() every epoch, right after model.train().
        # .eval() propagates recursively, so layer1.eval() also reaches every BatchNorm2d nested
        # inside its internal blocks, not just a single top-level layer.
        for frozen_module in (model.conv1, model.bn1, model.layer1):
            frozen_module.eval()
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
        print(f"[r34-{arm}/seed {seed}] ep {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_F1 {val_f1:.4f} | val_acc {val_acc:.4f} | "
              f"best {best_f1:.4f}@{best_epoch} | es {es_bad}/{ES_PATIENCE} | "
              f"lr_fc {lrs[-1]:.1e} | {time.time() - t0:.0f}s", flush=True)

        if es_bad >= ES_PATIENCE:
            print(f"[r34-{arm}/seed {seed}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    results = {"arm": arm, "architecture": "resnet34", "frozen": ["conv1", "bn1", "layer1"],
               "seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
               "best_val_macro_f1": best_f1, "train_time_s": time.time() - t_start,
               "class_weights": dict(zip(CLASSES, class_w.tolist())),
               "sim_threshold": args.sim_threshold, "metrics": {}}

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

    # Kaggle: Testing + Training combined, aug-name + near-duplicate filtered - same correction
    # as train_aug.py's run_seed(), kept identical here so results are directly comparable.
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
        df.to_csv(out / f"preds_{name}.csv", index=False)
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
    print(f"\n[r34-{arm}/seed {seed}] FINAL (best epoch {best_epoch})")
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
                    help="any arm name from augmentations.py (baseline, combined, combined2, "
                         "elastic_deform2, etc.) - same augmentation pipeline as train_aug.py, "
                         "just trained on ResNet-34 with conv1/bn1/layer1 frozen")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/<arm>_resnet34/")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sim-threshold", type=float, default=0.90)
    args = ap.parse_args()
    if args.out is None:
        args.out = here / "outputs" / f"{args.arm}_resnet34"

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"arm={args.arm} | architecture=resnet34 (conv1/bn1/layer1 frozen) | "
          f"torch {torch.__version__} | threads={torch.get_num_threads()} "
          f"| num_workers={args.num_workers}")

    m, k = args.data_root / "mendeley", args.data_root / "kaggle"

    train_items, _ = list_split(m / "train", MENDELEY_DIRS)
    print(f"[data] mendeley_train (raw, for on-the-fly aug): {len(train_items)} images")

    eval_data = {
        "mendeley_val": load_split("mendeley_val", m / "val", MENDELEY_DIRS, args.cache_dir, False),
        "mendeley_test": load_split("mendeley_test", m / "test", MENDELEY_DIRS, args.cache_dir, False),
        "kaggle_test": load_split("kaggle_test_orig", k / "Testing", KAGGLE_DIRS, args.cache_dir,
                                  False, exclude_aug=True, cache_name="kaggle_test_orig"),
        "kaggle_train": load_split("kaggle_train_orig", k / "Training", KAGGLE_DIRS, args.cache_dir,
                                   False, exclude_aug=True, cache_name="kaggle_train_orig"),
    }

    mtrain_cache = args.cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(f"{mtrain_cache} not found - run train_noaug.py at least once first.")
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