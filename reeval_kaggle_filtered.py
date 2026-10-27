"""
Recompute Kaggle cross-dataset metrics from saved per-image predictions, restricted to Kaggle
images whose similarity to their nearest Mendeley TRAIN image is below SIM_THRESHOLD (default
0.95). Combines Kaggle "Testing" (preds_kaggle_test.csv) with Kaggle "Training" used purely as
extra unseen evaluation data (preds_kaggle_train.csv, from eval_kaggle_training.py), if present.
Both get the same aug-name + similarity filtering before being combined into one "Kaggle" set.

Requires check_overlap.py to have been run first (outputs/noaug/overlap_kaggle_vs_mendeley_train.csv)
for the Testing-folder similarity; eval_kaggle_training.py already writes max_sim for the
Training-folder file directly into preds_kaggle_train.csv.

Reads   outputs/noaug/seed_*/preds_kaggle_test.csv
        outputs/noaug/seed_*/preds_kaggle_train.csv        (optional, extra eval data)
        outputs/noaug/seed_*/results.json                 (for Mendeley test metrics)
        outputs/noaug/overlap_kaggle_vs_mendeley_train.csv (for max_sim per Kaggle Testing image)
Writes  outputs/noaug/seed_*/results_kaggle_filtered.json
        outputs/noaug/summary_filtered.csv

Run from the same folder as train_noaug.py:
    python reeval_kaggle_filtered.py
    python reeval_kaggle_filtered.py --sim-threshold 0.95
    python reeval_kaggle_filtered.py --no-kaggle-train      # Testing folder only, old behaviour
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from train_noaug import CLASSES, compute_metrics, is_augmented

SCALAR = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]
HEADERS = {"macro_f1": "Macro F1", "accuracy": "Accuracy", "macro_precision": "Macro Prec",
           "macro_recall": "Macro Recall", "weighted_f1": "Weighted F1", "macro_f2": "Macro F2"}
ROW_NAMES = ["Mendeley test", "Kaggle test"]


def two_row_table(mend: dict, kag: dict) -> str:
    """Rows = Mendeley / Kaggle, columns = metrics."""
    df = pd.DataFrame([[mend[k] for k in SCALAR], [kag[k] for k in SCALAR]],
                      index=ROW_NAMES, columns=[HEADERS[k] for k in SCALAR])
    return df.map(lambda v: f"{v:.4f}").to_string()


def two_row_mean_sd(table: pd.DataFrame) -> str:
    """Same layout, each cell is mean +/- SD across seeds."""
    def cell(prefix, k):
        col = table[f"{prefix}_{k}"]
        return f"{col.mean():.4f} +/- {col.std(ddof=1):.4f}"
    df = pd.DataFrame([[cell("mendeley", k) for k in SCALAR], [cell("kaggle", k) for k in SCALAR]],
                      index=ROW_NAMES, columns=[HEADERS[k] for k in SCALAR])
    return df.to_string()


def rel_key(p) -> str:
    # split on both / and \\ so it's immune to which OS originally wrote the path
    # (matches check_overlap.py's kaggle_key)
    parts = [x for x in re.split(r"[\\/]+", str(p).strip()) if x]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(p))


def load_and_tag(path: Path, source: str, overlap: pd.DataFrame = None) -> pd.DataFrame:
    """Load a preds_*.csv, attach max_sim (from its own column or via merge), tag its source."""
    df = pd.read_csv(path)
    if "max_sim" not in df.columns:
        if overlap is None:
            raise ValueError(f"{path} has no max_sim column and no overlap table was given")
        df["kaggle_key"] = df["path"].map(rel_key)
        df = df.merge(overlap, on="kaggle_key", how="left")
        if df["max_sim"].isna().any():
            missing = int(df["max_sim"].isna().sum())
            print(f"[warn] {path.name}: {missing} images missing from overlap CSV "
                  f"(not filtered by similarity, only by aug-name)")
            df["max_sim"] = df["max_sim"].fillna(-1)
        df = df.drop(columns=["kaggle_key"])
    df["source"] = source
    return df


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=here / "outputs" / "noaug")
    ap.add_argument("--overlap-csv", type=Path, default=here / "outputs" / "noaug" /
                    "overlap_kaggle_vs_mendeley_train.csv")
    ap.add_argument("--sim-threshold", type=float, default=0.95,
                    help="drop Kaggle images with max_sim >= this value (default 0.95)")
    ap.add_argument("--no-kaggle-train", action="store_true",
                    help="ignore preds_kaggle_train.csv even if present (Testing folder only)")
    args = ap.parse_args()

    overlap = None
    if args.overlap_csv.exists():
        overlap = pd.read_csv(args.overlap_csv)
        if "kaggle_key" not in overlap.columns:
            raise ValueError(
                f"{args.overlap_csv} has no kaggle_key column - rerun check_overlap.py "
                f"with the updated script to regenerate it.")
        overlap = overlap[["kaggle_key", "max_sim"]]
    else:
        print(f"[warn] {args.overlap_csv} not found; Kaggle Testing images will only be filtered "
              f"by aug-name, not similarity. Run check_overlap.py to fix this.")

    rows = []
    for seed_dir in sorted(args.out.glob("seed_*")):
        pred_file, res_file = seed_dir / "preds_kaggle_test.csv", seed_dir / "results.json"
        if not (pred_file.exists() and res_file.exists()):
            print(f"[skip] {seed_dir.name}: preds or results.json missing (run not finished?)")
            continue

        parts = [load_and_tag(pred_file, "kaggle_test", overlap)]
        train_file = seed_dir / "preds_kaggle_train.csv"
        if train_file.exists() and not args.no_kaggle_train:
            parts.append(load_and_tag(train_file, "kaggle_train", overlap))
        elif not train_file.exists():
            print(f"[note] {seed_dir.name}: no preds_kaggle_train.csv "
                  f"(run eval_kaggle_training.py to add it)")
        df = pd.concat(parts, ignore_index=True)

        aug = df["path"].map(is_augmented)
        near_dup = df["max_sim"] >= args.sim_threshold
        drop = aug | near_dup
        kept = df[~drop]

        per_source = kept["source"].value_counts().to_dict()
        per_class_kept = kept["true"].value_counts().sort_index().reindex(range(len(CLASSES)), fill_value=0)
        per_class_drop = df[drop]["true"].value_counts().sort_index().reindex(range(len(CLASSES)), fill_value=0)

        kaggle = compute_metrics(kept["true"].to_numpy(), kept["pred"].to_numpy())
        with open(res_file) as f:
            res = json.load(f)
        mend = res["metrics"]["mendeley_test"]
        gap = {k: mend[k] - kaggle[k] for k in SCALAR}

        with open(seed_dir / "results_kaggle_filtered.json", "w") as f:
            json.dump({"seed": res["seed"], "sim_threshold": args.sim_threshold,
                       "n_kept": len(kept), "n_kept_by_source": per_source,
                       "n_dropped_aug": int(aug.sum()),
                       "n_dropped_near_dup": int((near_dup & ~aug).sum()),
                       "kaggle_test_filtered": kaggle, "generalisation_gap": gap}, f, indent=2)

        rows.append({"seed": res["seed"], "n_kaggle": len(kept),
                     **{f"mendeley_{k}": mend[k] for k in SCALAR},
                     **{f"kaggle_{k}": kaggle[k] for k in SCALAR},
                     **{f"gap_{k}": gap[k] for k in SCALAR}})

        print(f"\n=== Seed {res['seed']} (sim threshold {args.sim_threshold}) ===")
        print(f"Kaggle: kept {len(kept)}/{len(df)} | by source {per_source} "
              f"| dropped aug={int(aug.sum())}, near-dup={int((near_dup & ~aug).sum())}")
        print(f"   kept per class   : {dict(zip(CLASSES, per_class_kept.tolist()))}")
        print(f"   dropped per class: {dict(zip(CLASSES, per_class_drop.tolist()))}")
        print(two_row_table(mend, kaggle))
        print(f"Gap (Mendeley - Kaggle), macro F1: {gap['macro_f1']:+.4f}")

    if not rows:
        print("Nothing to summarise.")
        return
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary_filtered.csv", index=False)

    if len(table) > 1:
        print(f"\n=== Mean +/- SD across {len(table)} seeds ===")
        print(two_row_mean_sd(table))
        g = table["gap_macro_f1"]
        print(f"Gap (Mendeley - Kaggle), macro F1: {g.mean():+.4f} +/- {g.std(ddof=1):.4f}")
    print(f"\nFull per-seed numbers (including all gaps): {args.out / 'summary_filtered.csv'}")


if __name__ == "__main__":
    main()