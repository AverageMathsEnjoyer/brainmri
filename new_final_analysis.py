"""
Final cross-arm comparison: evaluates every trained arm on Mendeley test and on Kaggle
(Testing+Training combined, aug-name + near-duplicate filtered at --sim-threshold, default 0.90),
reports seed mean +/- SD, the generalisation gap, and a paired-bootstrap 95% CI on each arm's
Kaggle macro-F1 delta vs the `baseline` arm (spec section 6). Produces one summary CSV and one
comparison figure, with arms in a fixed order: no_aug, baseline, then every additive arm.

IMPORTANT: this script does NOT trust each arm's own saved results.json/results_kaggle_filtered.json
numbers for the Kaggle side. Those came from two different pipelines (train_noaug.py's separate
reeval_kaggle_filtered.py step for the no_aug arm, vs train_aug.py's built-in filtering for every
other arm) and could in principle drift out of sync (e.g. if run with different --sim-threshold
values at different times). Instead it recomputes the aug-name + similarity filter itself, ONCE,
directly from the Kaggle Testing/Training caches, and applies that one canonical "kept" image set
to every arm's raw preds_kaggle_test.csv / preds_kaggle_train.csv. This guarantees every arm is
scored on the identical set of Kaggle images, which paired bootstrap requires.

Expects, for every arm, exactly the files train_noaug.py/train_aug.py/eval_kaggle_training.py
already produce:
    outputs/<arm_dir>/seed_{123,456,789}/preds_mendeley_test.csv  (path, true, pred, prob_*)
    outputs/<arm_dir>/seed_{123,456,789}/preds_kaggle_test.csv    (path, true, pred, prob_*[, max_sim, source])
    outputs/<arm_dir>/seed_{123,456,789}/preds_kaggle_train.csv   (same columns)
(max_sim/source columns, if present, are ignored - the canonical filter below is used instead.)

Usage:
    python final_analysis.py
    python final_analysis.py --sim-threshold 0.90 --n-resamples 5000
"""
import argparse
import json
import re
from pathlib import Path


def rel_key(p) -> str:
    """Class-folder/filename key, immune to which OS/environment wrote the path (forward vs
    backslash separators, different absolute prefixes). Different arms here were trained in a
    mix of Colab and local Windows sessions, so raw absolute-path matching is not reliable -
    this is the same fix applied earlier in check_overlap.py/reeval_kaggle_filtered.py."""
    parts = [x for x in re.split(r"[\\/]+", str(p).strip()) if x]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else str(p))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from check_overlap import best_match, embed
from train_noaug import CLASSES, compute_metrics, is_augmented

SEEDS = [123, 456, 789]
SCALAR = ["macro_f1", "accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f2"]

# Fixed display order. Left side maps the display name to the folder under outputs/; anything not
# listed here uses its own name as the folder name directly (true for every additive arm).
ARM_ORDER = ["no_aug", "baseline", "elastic_deform", "elastic_deform2", "gamma",
            "gaussian_blur", "gaussian_blur2", "rician_noise", "gaussian_noise",
            "gaussian_noise2", "kspace_motion_blur", "bias_field", "bias_field2", "spike",
            "ghosting", "ghosting2", "brightness_contrast"]
ARM_DIR = {"no_aug": "noaug"}


def arm_dir(out_root: Path, arm: str) -> Path:
    return out_root / ARM_DIR.get(arm, arm)


# ----------------------------------------------------------------------------
# Step 1: the ONE canonical Kaggle "kept" image set (model-independent - computed once, shared
# by every arm/seed below).
# ----------------------------------------------------------------------------
def build_canonical_kept_sets(cache_dir: Path, sim_threshold: float):
    mtrain_cache = cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(f"{mtrain_cache} not found - run train_noaug.py at least once first.")
    e_tr = embed(np.load(mtrain_cache)["x"])

    kept = {}
    for split, cache_name in [("kaggle_test", "kaggle_test_orig"), ("kaggle_train", "kaggle_train_orig")]:
        d = np.load(cache_dir / f"{cache_name}.npz")
        paths = d["paths"]
        sim, _ = best_match(embed(d["x"]), e_tr)
        aug = np.array([is_augmented(p) for p in paths])
        near_dup = sim >= sim_threshold
        kept_keys = set(rel_key(p) for p in paths[~(aug | near_dup)])
        kept[split] = kept_keys
        print(f"[kept] {split}: {len(kept_keys)}/{len(paths)} images kept "
              f"(dropped aug={aug.sum()}, near-dup={int((near_dup & ~aug).sum())}, "
              f"sim_threshold={sim_threshold})")
    return kept


# ----------------------------------------------------------------------------
# Step 2: per arm/seed, load raw predictions and apply the canonical filter.
# ----------------------------------------------------------------------------
def load_seed(out_root: Path, arm: str, seed: int, kept: dict):
    sd = arm_dir(out_root, arm) / f"seed_{seed}"
    mend = pd.read_csv(sd / "preds_mendeley_test.csv")

    kte = pd.read_csv(sd / "preds_kaggle_test.csv")
    kte["rel_key"] = kte["path"].map(rel_key)
    kte = kte[kte["rel_key"].isin(kept["kaggle_test"])]
    ktr = pd.read_csv(sd / "preds_kaggle_train.csv")
    ktr["rel_key"] = ktr["path"].map(rel_key)
    ktr = ktr[ktr["rel_key"].isin(kept["kaggle_train"])]
    kag = pd.concat([kte, ktr], ignore_index=True)
    dupes = kag["rel_key"].duplicated()
    if dupes.any():
        print(f"[warn] {arm}/seed {seed}: {dupes.sum()} duplicate rel_key(s) across "
              f"kaggle_test/kaggle_train - keeping first occurrence only")
        kag = kag[~dupes]
    kag = kag.sort_values("rel_key").reset_index(drop=True)

    return {"mendeley_true": mend["true"].to_numpy(), "mendeley_pred": mend["pred"].to_numpy(),
           "kaggle_key": kag["rel_key"].to_numpy(), "kaggle_true": kag["true"].to_numpy(),
           "kaggle_pred": kag["pred"].to_numpy()}


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ----------------------------------------------------------------------------
# Step 3: paired bootstrap, arm vs baseline, on Kaggle macro F1 (spec section 6).
# Each resample draws image indices (stratified by class) with replacement from the N canonical
# Kaggle images; for every resampled image, ALL 3 seeds' predictions for that image are pulled in
# (this is the "pool all 3 seeds" step), so each resample scores 3xN_resampled predictions.
# ----------------------------------------------------------------------------
def paired_bootstrap_delta(true_ref, arm_pred_stack, base_pred_stack, n_resamples, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(true_ref)
    class_idx = {c: np.where(true_ref == c)[0] for c in classes}
    deltas = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = np.concatenate([rng.choice(class_idx[c], size=len(class_idx[c]), replace=True)
                              for c in classes])
        yt = np.tile(true_ref[idx], arm_pred_stack.shape[0])
        ya = arm_pred_stack[:, idx].ravel()
        yb = base_pred_stack[:, idx].ravel()
        deltas[r] = macro_f1(yt, ya) - macro_f1(yt, yb)
    return deltas


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=here / "outputs")
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--sim-threshold", type=float, default=0.90)
    ap.add_argument("--n-resamples", type=int, default=5000)
    ap.add_argument("--figure-out", type=Path, default=here / "outputs" / "final_analysis.png")
    ap.add_argument("--csv-out", type=Path, default=here / "outputs" / "final_analysis_summary.csv")
    args = ap.parse_args()

    kept = build_canonical_kept_sets(args.cache_dir, args.sim_threshold)

    # Load every arm/seed, skipping arms with missing data (so a partially-finished run still
    # produces a figure for whatever is ready) rather than crashing the whole script.
    data = {}
    for arm in ARM_ORDER:
        seed_data = {}
        for seed in SEEDS:
            try:
                seed_data[seed] = load_seed(args.out_root, arm, seed, kept)
            except FileNotFoundError as e:
                print(f"[skip] {arm}/seed {seed}: {e}")
        if seed_data:
            data[arm] = seed_data
    present_arms = [a for a in ARM_ORDER if a in data]
    missing = [a for a in ARM_ORDER if a not in data]
    if missing:
        print(f"\n[warn] no data at all for: {missing} - omitted from the figure/table.\n")
    if "baseline" not in data:
        raise RuntimeError("baseline arm has no data - it's required as the bootstrap reference.")

    # ---- per-arm, per-seed metrics + seed mean/SD ----
    rows = []
    for arm in present_arms:
        for seed, d in data[arm].items():
            mend = compute_metrics(d["mendeley_true"], d["mendeley_pred"])
            kag = compute_metrics(d["kaggle_true"], d["kaggle_pred"])
            row = {"arm": arm, "seed": seed, "n_kaggle": len(d["kaggle_true"])}
            row.update({f"mendeley_{k}": mend[k] for k in SCALAR})
            row.update({f"kaggle_{k}": kag[k] for k in SCALAR})
            row["gap_macro_f1"] = mend["macro_f1"] - kag["macro_f1"]
            rows.append(row)
    per_seed = pd.DataFrame(rows)

    summary = per_seed.groupby("arm").agg(
        n_seeds=("seed", "count"),
        mendeley_macro_f1_mean=("mendeley_macro_f1", "mean"), mendeley_macro_f1_sd=("mendeley_macro_f1", "std"),
        kaggle_macro_f1_mean=("kaggle_macro_f1", "mean"), kaggle_macro_f1_sd=("kaggle_macro_f1", "std"),
        kaggle_accuracy_mean=("kaggle_accuracy", "mean"), kaggle_accuracy_sd=("kaggle_accuracy", "std"),
        gap_macro_f1_mean=("gap_macro_f1", "mean"), gap_macro_f1_sd=("gap_macro_f1", "std"),
    ).reindex(present_arms)

    # ---- paired bootstrap vs baseline, per arm (skip baseline itself: delta is 0 by definition) ----
    # Requires the SAME canonical Kaggle images, in the SAME order, across every seed of an arm,
    # and between the arm and baseline - guaranteed by filtering on the single canonical `kept`
    # path sets above and sorting by path in load_seed().
    def seeds_agree(arm):
        """All 3 seeds of this arm must have identical, identically-ordered kaggle_key arrays
        for np.stack() below to be a valid per-image alignment across seeds."""
        present = [s for s in SEEDS if s in data[arm]]
        if len(present) < 3:
            return False
        ref = data[arm][present[0]]["kaggle_key"]
        return all(np.array_equal(data[arm][s]["kaggle_key"], ref) for s in present[1:])

    for arm in present_arms:
        if not seeds_agree(arm):
            print(f"[warn] {arm}: seeds disagree on the kept Kaggle image set/order "
                  f"(missing seed, or a genuinely different image set between seeds) - "
                  f"excluding this arm from the bootstrap entirely.")
    bootstrap_eligible = [a for a in present_arms if seeds_agree(a)]

    if "baseline" not in bootstrap_eligible:
        raise RuntimeError("baseline arm's 3 seeds don't agree on the kept Kaggle image set - "
                           "cannot run any bootstrap comparison. Check n_kaggle per seed in the "
                           "per-seed summary CSV for baseline.")

    base_keys = data["baseline"][SEEDS[0]]["kaggle_key"]
    base_true = data["baseline"][SEEDS[0]]["kaggle_true"]
    base_stack = np.stack([data["baseline"][s]["kaggle_pred"] for s in SEEDS])

    boot_rows = []
    for arm in bootstrap_eligible:
        if arm == "baseline":
            continue
        arm_keys = data[arm][SEEDS[0]]["kaggle_key"]
        if not np.array_equal(arm_keys, base_keys):
            print(f"[warn] {arm}: Kaggle image set/order differs from baseline even after "
                  f"rel_key matching - skipping bootstrap for this arm. Likely cause: this arm's "
                  f"kept image COUNT differs from baseline's (check the per-seed n_kaggle in the "
                  f"summary CSV), not just path formatting.")
            continue
        arm_stack = np.stack([data[arm][s]["kaggle_pred"] for s in SEEDS])
        deltas = paired_bootstrap_delta(base_true, arm_stack, base_stack, args.n_resamples)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        boot_rows.append({"arm": arm, "delta_mean": deltas.mean(), "ci_lo": lo, "ci_hi": hi,
                          "significant": bool(lo > 0 or hi < 0)})
        print(f"[bootstrap] {arm:20s} delta vs baseline: {deltas.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]{'  *' if (lo > 0 or hi < 0) else ''}")
    bootstrap = pd.DataFrame(boot_rows).set_index("arm") if boot_rows else pd.DataFrame()

    summary = summary.join(bootstrap, how="left")
    summary.to_csv(args.csv_out)
    per_seed.to_csv(args.csv_out.with_name(args.csv_out.stem + "_per_seed.csv"), index=False)
    print(f"\nWrote {args.csv_out}\nWrote {args.csv_out.with_name(args.csv_out.stem + '_per_seed.csv')}")

    # ---- figure: 3 panels, arms in ARM_ORDER (only those with data) ----
    x = np.arange(len(present_arms))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(present_arms) * 0.9), 12), sharex=True)

    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, summary["mendeley_macro_f1_mean"], w, yerr=summary["mendeley_macro_f1_sd"],
          capsize=3, label="Mendeley test", color="#4C72B0")
    ax.bar(x + w / 2, summary["kaggle_macro_f1_mean"], w, yerr=summary["kaggle_macro_f1_sd"],
          capsize=3, label="Kaggle (filtered)", color="#DD8452")
    ax.set_ylabel("Macro F1")
    ax.set_title("In-domain (Mendeley) vs cross-dataset (Kaggle) macro F1, by arm (mean +/- SD across 3 seeds)")
    ax.set_ylim(0.9, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in summary["gap_macro_f1_mean"]]
    ax.bar(x, summary["gap_macro_f1_mean"], yerr=summary["gap_macro_f1_sd"], capsize=3, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mendeley - Kaggle\n(macro F1)")
    ax.set_title("Generalisation gap, by arm (mean +/- SD across 3 seeds)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    delta_mean = summary["delta_mean"].reindex(present_arms)
    err_lo = (delta_mean - summary["ci_lo"]).reindex(present_arms)
    err_hi = (summary["ci_hi"] - delta_mean).reindex(present_arms)
    has_delta = delta_mean.notna()
    colors = ["#937860" if not sig else ("#55A868" if d >= 0 else "#C44E52")
             for d, sig in zip(delta_mean, summary["significant"].reindex(present_arms).fillna(False))]
    ax.bar(x[has_delta.to_numpy()], delta_mean[has_delta], color=np.array(colors)[has_delta.to_numpy()],
          yerr=[err_lo[has_delta], err_hi[has_delta]], capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta vs baseline\n(Kaggle macro F1)")
    ax.set_title("Paired bootstrap 95% CI, Kaggle macro F1 delta vs baseline "
                 f"({args.n_resamples} resamples; red = CI excludes 0)")
    ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(present_arms, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.figure_out, dpi=150)
    print(f"Wrote {args.figure_out}")


if __name__ == "__main__":
    main()