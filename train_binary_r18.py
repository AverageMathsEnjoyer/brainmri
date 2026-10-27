"""
Trains a binary ResNet-18 model to classify images as "Tumour" (0) vs "No Tumour" (1).
Uses class weighting to handle the severe imbalance, and matches the exact logging, 
early stopping, and output saving conventions of train_noaug.py.

Uses baseline augmentations via AugmentedMendeleyTrain for the training split.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from train_noaug import (BATCH_SIZE, CLASSES, MENDELEY_DIRS, KAGGLE_DIRS, MAX_EPOCHS,
                         ES_PATIENCE, ES_MIN_DELTA, build_optimizer, load_split, list_split,
                         predict_logits, set_seed, to_input)
from train_aug import AugmentedMendeleyTrain, worker_init_fn

NO_TUMOR_IDX = CLASSES.index("no_tumor")


class BinaryAugmentedTrain(Dataset):
    """
    Wraps AugmentedMendeleyTrain to convert its 4-class labels to binary labels 
    (1 for no_tumor, 0 for tumor) on the fly without modifying train_aug.py.
    """
    def __init__(self, augmented_dataset: AugmentedMendeleyTrain):
        self.aug_ds = augmented_dataset
        
    def __len__(self):
        return len(self.aug_ds)
        
    def __getitem__(self, idx):
        img_tensor, label_4class = self.aug_ds[idx]
        label_binary = 1 if label_4class == NO_TUMOR_IDX else 0
        return img_tensor, label_binary


def build_model_binary():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "binary_resnet18")
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    ap.add_argument("--num-workers", type=int, default=4, help="workers for augmentation")
    args = ap.parse_args()

    out = args.out_dir / f"seed_{args.seed}"
    if (out / "results.json").exists() and not args.force:
        print(f"[seed {args.seed}] results.json exists, skipping (use --force to rerun)")
        return
        
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"\n{'=' * 70}\n[seed {args.seed}] starting on {device}\n{'=' * 70}", flush=True)

    set_seed(args.seed)

    cache_dir = here / ".cache"
    
    # Validation and Test sets remain statically cached without augmentation
    x_va, y_va_4class, paths_va = load_split("mendeley_val", here / "mendeley" / "val", MENDELEY_DIRS, cache_dir, False)
    x_te_mend, y_te_mend_4class, paths_te_mend = load_split("mendeley_test", here / "mendeley" / "test", MENDELEY_DIRS, cache_dir, False)
    x_te_kag, y_te_kag_4class, paths_te_kag = load_split("kaggle_test", here / "kaggle" / "Testing", KAGGLE_DIRS, cache_dir, False)

    # For the training set, we need raw paths/labels for the Augmented dataset
    train_items, n_excluded = list_split(here / "mendeley" / "train", MENDELEY_DIRS, exclude_aug=True)

    # Convert static 4-class labels to binary
    y_va = (y_va_4class == NO_TUMOR_IDX).long()
    y_te_mend = (y_te_mend_4class == NO_TUMOR_IDX).long()
    y_te_kag = (y_te_kag_4class == NO_TUMOR_IDX).long()
    
    data_dict = {
        "mendeley_val": (x_va, y_va, paths_va),
        "mendeley_test": (x_te_mend, y_te_mend, paths_te_mend),
        "kaggle_test": (x_te_kag, y_te_kag, paths_te_kag)
    }

    # Calculate class weights from raw train_items
    train_labels_binary = [1 if c == NO_TUMOR_IDX else 0 for _, c in train_items]
    counts = torch.bincount(torch.tensor(train_labels_binary), minlength=2).float()
    class_w = counts.sum() / (2.0 * counts)
    print(f"[seed {args.seed}] class weights: {{'Tumor': {class_w[0]:.4f}, 'No_Tumor': {class_w[1]:.4f}}}")

    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    model = build_model_binary().to(device)
    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    # Wrap AugmentedMendeleyTrain in our binary wrapper
    aug_train_4class = AugmentedMendeleyTrain(train_items, arm="baseline")
    train_ds = BinaryAugmentedTrain(aug_train_4class)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        generator=g, 
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device.type == "cuda")
    )

    best_f1, best_epoch = -1.0, 0 
    es_ref, es_bad = -1.0, 0 
    history = []
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
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
                            lr_fc=lrs[-1], lr_layer4=lrs[-3], 
                            epoch_time_s=time.time() - t0))
                            
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[seed {args.seed}] ep {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_F1 {val_f1:.4f} | val_acc {val_acc:.4f} | best {best_f1:.4f}@{best_epoch} "
              f"| es {es_bad}/{ES_PATIENCE} | lr_fc {lrs[-1]:.1e} | {time.time() - t0:.0f}s", flush=True)

        if es_bad >= ES_PATIENCE:
            print(f"[seed {args.seed}] early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    
    def compute_metrics_binary(y_true, y_pred):
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "per_class_f1": dict(zip(["Tumor", "No_Tumor"], f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0).tolist())),
        }
        
    results = {"seed": args.seed, "best_epoch": best_epoch, "epochs_run": epoch,
               "best_val_macro_f1": best_f1, "train_time_s": time.time() - t_start,
               "class_weights": {"Tumor": float(class_w[0]), "No_Tumor": float(class_w[1])}, "metrics": {}}

    for name in ["mendeley_val", "mendeley_test", "kaggle_test"]:
        x, y, paths = data_dict[name]
        logits = predict_logits(model, x, device)
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()
        pred = probs.argmax(1)
        results["metrics"][name] = compute_metrics_binary(y.numpy(), pred)
        
        df = pd.DataFrame({"path": paths, "true": y.numpy(), "pred": pred})
        df["prob_Tumor"] = probs[:, 0]
        df["prob_No_Tumor"] = probs[:, 1]
        df.to_csv(out / f"preds_{name}.csv", index=False)

    results["generalisation_gap"] = {
        "macro_f1": results["metrics"]["mendeley_test"]["macro_f1"] - results["metrics"]["kaggle_test"]["macro_f1"],
        "accuracy": results["metrics"]["mendeley_test"]["accuracy"] - results["metrics"]["kaggle_test"]["accuracy"]
    }

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[seed {args.seed}] FINAL (best epoch {best_epoch}): mendeley_test F1 "
          f"{results['metrics']['mendeley_test']['macro_f1']:.4f}, kaggle_test F1 "
          f"{results['metrics']['kaggle_test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()