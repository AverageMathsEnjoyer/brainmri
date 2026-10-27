"""
Evaluates an ensemble of 3 baseline ResNet-34 models (seeds 123, 456, 789) on a given
dataset split.

The prediction is determined by majority vote across the 3 models. In the event of a
3-way tie (all three models predict a different class), the prediction with the highest
individual softmax confidence is chosen.

Usage:
    # Evaluate the ensemble on Mendeley test split
    python ensemble_resnet34.py --split mendeley_test
    
    # Evaluate on Kaggle test split
    python ensemble_resnet34.py --split kaggle_test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from train_noaug import BATCH_SIZE, CLASSES, load_split, to_input
from train_resnet34 import build_model_r34
from torch.utils.data import DataLoader, TensorDataset
from train_noaug import BATCH_SIZE, CLASSES, KAGGLE_DIRS, MENDELEY_DIRS, load_split, to_input

# ----------------------------------------------------------------------------
def ensemble_predict(logits1: torch.Tensor, logits2: torch.Tensor, logits3: torch.Tensor) -> torch.Tensor:
    """
    Takes 3 sets of logits (batch_size, num_classes) and returns the ensembled 
    class indices (batch_size,) using majority vote + max confidence tie-breaker.
    """
    probs1 = F.softmax(logits1, dim=1)
    probs2 = F.softmax(logits2, dim=1)
    probs3 = F.softmax(logits3, dim=1)

    preds1 = probs1.argmax(dim=1)
    preds2 = probs2.argmax(dim=1)
    preds3 = probs3.argmax(dim=1)

    final_preds = []
    for i in range(len(preds1)):
        p1, p2, p3 = preds1[i].item(), preds2[i].item(), preds3[i].item()
        
        # Majority vote
        if p1 == p2 or p1 == p3:
            final_preds.append(p1)
        elif p2 == p3:
            final_preds.append(p2)
        else:
            # 3-way tie: Compare the max confidence of their respective predicted classes
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
    
    # Defaulting to the baseline outputs for the 3 seeds
    ap.add_argument("--ckpt1", type=Path, default=here / "outputs" / "baseline_resnet34" / "seed_123" / "best.pt")
    ap.add_argument("--ckpt2", type=Path, default=here / "outputs" / "baseline_resnet34" / "seed_456" / "best.pt")
    ap.add_argument("--ckpt3", type=Path, default=here / "outputs" / "baseline_resnet34" / "seed_789" / "best.pt")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # Load the 3 models
    models = []
    for i, ckpt_path in enumerate([args.ckpt1, args.ckpt2, args.ckpt3], start=1):
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        print(f"Loading Model {i} from {ckpt_path.parent.name}...")
        model = build_model_r34().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()  # Set to evaluation mode
        models.append(model)
        
    print(f"\nLoading data for split: {args.split}...")
    
    # 1. Map the split name to its physical directory and class-folder mapping
    if args.split == "mendeley_test":
        split_dir = here / "mendeley" / "test"
        dirmap = MENDELEY_DIRS
    elif args.split == "kaggle_test":
        split_dir = here / "kaggle" / "Testing"
        dirmap = KAGGLE_DIRS
    else:
        raise ValueError(f"Unknown split: {args.split}")

    cache_dir = here / ".cache"
    
    # 2. Call load_split with the required 5 positional arguments
    x, y, _ = load_split(
        name=args.split,
        split_dir=split_dir,
        dirmap=dirmap,
        cache_dir=cache_dir,
        rebuild=False
    )
    
    # 3. Wrap the returned tensors in a TensorDataset
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers)
    all_preds = []
    all_targets = []
    
    print("Evaluating ensemble...")
    with torch.no_grad():
        for x, y in loader:
            x = to_input(x, device)
            
            # Forward passes
            logits1 = models[0](x)
            logits2 = models[1](x)
            logits3 = models[2](x)
            
            # Ensemble predictions
            preds = ensemble_predict(logits1, logits2, logits3)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.tolist())
            
    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    
    print("\n" + "-"*40)
    print(f"Ensemble Results on {args.split}")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()