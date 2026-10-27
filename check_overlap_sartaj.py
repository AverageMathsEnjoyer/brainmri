"""
Near-duplicate check: does the separately-downloaded SARTAJ dataset (sartaj/Testing and
sartaj/Training) contain images that are (near-)identical to Mendeley TRAIN images? Same method
as check_overlap.py (used for the Nickparvar Kaggle dataset), applied here to a different
external source. SARTAJ is itself one of the three source datasets the Nickparvar Kaggle set was
built from (figshare + SARTAJ + Br35H), so overlap here is at least plausible and worth checking
before treating this as an independent second cross-dataset eval set.

Note: SARTAJ's glioma class is widely reported to contain mislabelled images - it's why the
Nickparvar dataset's own authors replaced SARTAJ's glioma folder with figshare's instead of using
it directly. Worth keeping in mind if glioma results from this dataset look unusual later.
e
Folder structure expected (script sits next to it, same as mendeley/ and kaggle/):
    sartaj/Testing/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}
    sartaj/Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}
Both splits are checked and combined - "Training" here is just another folder of unseen images
to check for overlap, not something this project ever trains on.

Run from the folder containing train_noaug.py, after its caches exist (needs cache/mendeley_train.npz):
    python check_overlap_sartaj.py
Writes outputs/noaug/overlap_sartaj_vs_mendeley_train.csv (sorted by similarity, both splits combined)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from check_overlap import THRESHOLDS, best_match, embed
from train_noaug import CLASSES, load_split

SARTAJ_DIRS = {"glioma": "glioma_tumor", "meningioma": "meningioma_tumor",
              "no_tumor": "no_tumor", "pituitary": "pituitary_tumor"}


def summarise(label: str, sims: np.ndarray):
    pct = np.percentile(sims, [50, 90, 99])
    over = "  ".join(f">{t}: {(sims > t).sum():4d} ({(sims > t).mean() * 100:4.1f}%)" for t in THRESHOLDS)
    print(f"{label:34s} median {pct[0]:.3f}  p90 {pct[1]:.3f}  p99 {pct[2]:.3f}  max {sims.max():.3f} | {over}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--cache-dir", type=Path, default=here / "cache")
    ap.add_argument("--out", type=Path, default=here / "outputs" / "noaug")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    s = args.data_root / "sartaj"

    # cache + preprocess both splits with the exact same pipeline used everywhere else in this
    # project (grayscale -> percentile clip -> Otsu z-score -> bicubic resize), so the embedding
    # is comparing like with like. exclude_aug=True is a harmless no-op here if SARTAJ has no
    # "aug"-named files (unlike the Nickparvar Kaggle set), but costs nothing to leave on.
    x_te, y_te, p_te = load_split("sartaj_test_orig", s / "Testing", SARTAJ_DIRS, args.cache_dir,
                                  args.rebuild_cache, exclude_aug=True, cache_name="sartaj_test_orig")
    x_tr, y_tr, p_tr = load_split("sartaj_train_orig", s / "Training", SARTAJ_DIRS, args.cache_dir,
                                  args.rebuild_cache, exclude_aug=True, cache_name="sartaj_train_orig")

    mtrain_cache = args.cache_dir / "mendeley_train.npz"
    if not mtrain_cache.exists():
        raise FileNotFoundError(
            f"{mtrain_cache} not found. Run train_noaug.py at least once first so this cache exists.")
    d = np.load(mtrain_cache)
    e_ref, p_ref = embed(d["x"]), d["paths"]

    s_te, j_te = best_match(embed(x_te.numpy()), e_ref)
    s_tr, j_tr = best_match(embed(x_tr.numpy()), e_ref)

    print("\nMax similarity of each SARTAJ image to its closest Mendeley TRAIN image\n" + "-" * 110)
    summarise("SARTAJ Testing  -> mendeley train", s_te)
    summarise("SARTAJ Training -> mendeley train", s_tr)

    df = pd.concat([
        pd.DataFrame({"sartaj_path": p_te, "split": "Testing", "true": y_te.numpy(),
                     "max_sim": s_te, "closest_mendeley_train": p_ref[j_te]}),
        pd.DataFrame({"sartaj_path": p_tr, "split": "Training", "true": y_tr.numpy(),
                     "max_sim": s_tr, "closest_mendeley_train": p_ref[j_tr]}),
    ], ignore_index=True)

    df["class"] = df["true"].map(dict(enumerate(CLASSES)))
    print("\nPer-class breakdown (both splits combined):")
    for c in CLASSES:
        sub = df.loc[df["class"] == c, "max_sim"]
        if len(sub):
            print(f"  {c:12s} n={len(sub):4d}  median {sub.median():.3f}  "
                  f">=0.90: {(sub >= 0.90).sum():4d} ({(sub >= 0.90).mean() * 100:4.1f}%)")

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "overlap_sartaj_vs_mendeley_train.csv"
    df.sort_values("max_sim", ascending=False).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  (sorted by similarity; open the top rows side by side to eyeball them)")


if __name__ == "__main__":
    main()