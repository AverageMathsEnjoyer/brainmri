"""
Compares the performance (Accuracy and F1) of 3 individual baseline ResNet-34 models
(seeds 123, 456, 789) against an ensemble of the three using majority vote.

Usage:
    python compare_ensemble.py --split mendeley_test
    python compare_ensemble.py --split kaggle_test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from train_noaug import BATCH_SIZE, CLASSES, KAGGLE_DIRS, MENDELEY_DIRS, load_split, to_input
from train_resnet34 import build_model_r34


# ----------------------------------------------------------------------------
def ensemble_predict(logits1: torch.Tensor, logits2: torch.Tensor, logits3: torch.Tensor) -> torch.Tensor:
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)
    probs3 = F.softmax(logits3, dim=1)

    preds1 = probs1.argmax(dim=1)
    preds2 = probs2.argmax(dim=1)
    preds3 = probs3.argmax(dim=1)

    final_preds = []
    for i in range(len(preds1)):
        p1, p2, p3 = preds1[i].item(), preds2[i].item(), preds3[i].item()
        
        if p1 == p2 or p1 == p3:
            final_preds.append(p1)
        elif p2 == p3:
            final_preds.append(p2)
        else:
            c1 = probs1[i, p1].item()
            c2 = probs2[i, p2].item()
            c3 = probs3[i, p3].item()
            
            best_idx = np.argmax([c1, c2, c3])
            final_preds.append([p1, p2, p3][best_idx])
            
    return torch.tensor(final_preds, device=logits1.device)


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", choices=["mendeley_test", "kaggle_test"], default="mendeley_test")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    
    ckpt_paths = [
        here / "outputs" / "baseline_resnet34" / "seed_123" / "best.pt",
        here / "outputs" / "baseline_resnet34" / "seed_456" / "best.pt",
        here / "outputs" / "baseline_resnet34" / "seed_789" / "best.pt"
    ]
    
    models = []
    for i, ckpt_path in enumerate(ckpt_paths, start=1):
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        print(f"Loading Model {i} from {ckpt_path.parent.name}...")
        model = build_model_r34().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
        
    print(f"\nLoading data for split: {args.split}...")
    
    if args.split == "mendeley_test":
        split_dir = here / "mendeley" / "test"
        dirmap = MENDELEY_DIRS
    elif args.split == "kaggle_test":
        split_dir = here / "kaggle" / "Testing"
        dirmap = KAGGLE_DIRS
    else:
        raise ValueError(f"Unknown split: {args.split}")

    # Saves/loads the cache exactly where train_noaug does
    cache_dir = here / ".cache"
    x, y, _ = load_split(
        name=args.split,
        split_dir=split_dir,
        dirmap=dirmap,
        cache_dir=cache_dir,
        rebuild=False
    )
    
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    
    all_targets = []
    all_preds_m1 = []
    all_preds_m2 = []
    all_preds_m3 = []
    all_preds_ens = []
    
    print("Evaluating models...")
    with torch.no_grad():
        for xb, yb in loader:
            xb = to_input(xb, device)
            
            logits1 = models[0](xb)
            logits2 = models[1](xb)
            logits3 = models[2](xb)
            
            ens_preds = ensemble_predict(logits1, logits2, logits3)
            
            all_preds_m1.extend(logits1.argmax(dim=1).cpu().tolist())
            all_preds_m2.extend(logits2.argmax(dim=1).cpu().tolist())
            all_preds_m3.extend(logits3.argmax(dim=1).cpu().tolist())
            all_preds_ens.extend(ens_preds.cpu().tolist())
            all_targets.extend(yb.tolist())
            
    # Calculate and Display Metrics
    results = [
        ("Model 1 (Seed 123)", all_preds_m1),
        ("Model 2 (Seed 456)", all_preds_m2),
        ("Model 3 (Seed 789)", all_preds_m3),
        ("Ensemble (Majority)", all_preds_ens)
    ]
    
    print("\n" + "="*50)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 50)
    for name, preds in results:
        acc = accuracy_score(all_targets, preds)
        f1 = f1_score(all_targets, preds, average="macro")
        print(f"{name:<25} | {acc:.4f}     | {f1:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()