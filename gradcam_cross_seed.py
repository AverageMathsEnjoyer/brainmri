"""
Extends gradcam_misclassified.py from one seed to all three: scans every Mendeley test image
against all 3 trained seeds of a given arm, finds the UNION of everything any seed gets wrong,
draws a 3-set Venn diagram of which seeds got which images wrong, and runs Grad-CAM on every
image in that union.

For each image in the union, all 3 seeds' Grad-CAMs are stitched side by side - EACH seed's own
CAM for its OWN predicted class (not the true class), since each seed is a separately-trained
model making its own independent decision. Every seed is shown for every image in the union,
even a seed that happened to get that particular image right, so you can directly compare what
each of the 3 models looked at. Each panel's caption gives that seed's number, its predicted
class and confidence, and whether that prediction was correct. The filename still records every
seed that got the image wrong, as a quick-glance summary without opening it.

Needs matplotlib-venn for the 3-set diagram (not a core matplotlib package):
    pip install matplotlib-venn

Output goes in a dedicated subfolder: <out-dir>/cross_seed_misclassified_mendeley_test/
  - venn_diagram.png
  - cross_seed_misclassified_summary.csv  (every seed's prediction+confidence per image)
  - one stitched (3-panel) Grad-CAM PNG per uniquely-wrong image, filename encoding wrong seeds:
        {name}__true-{label}__wrongseeds-123-456.png

Usage:
    python gradcam_cross_seed.py --arm-dir outputs/baseline_resnet34
    python gradcam_cross_seed.py --arm-dir outputs/baseline_resnet34 --seeds 123 456 789
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from matplotlib_venn import venn3
except ImportError as e:
    raise ImportError(
        "matplotlib-venn is required for the 3-set Venn diagram and isn't a core matplotlib "
        "package. Install it with: pip install matplotlib-venn") from e

from PIL import Image, ImageDraw, ImageFont

from gradcam_resnet34 import GradCAM, overlay
from train_noaug import CLASSES, MENDELEY_DIRS, list_split, preprocess, to_input
from train_resnet34 import build_model_r34


def stitch_n_with_labels(images: list, labels: list, label_h: int = 32, gap: int = 4,
                         bg=(255, 255, 255)) -> Image.Image:
    """Combines any number of same-size PIL Images side by side, each with its own text caption
    underneath, entirely in memory - nothing is written to disk until the single returned,
    combined image is saved. (A generalised, N-panel version of the 2-panel stitch_with_labels
    in gradcam_misclassified.py; kept separate here rather than modifying that one, since its
    existing 2-panel callers shouldn't need to change.)"""
    assert len(images) == len(labels)
    w, h = images[0].size
    n = len(images)
    canvas = Image.new("RGB", (w * n + gap * (n - 1), h + label_h), bg)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for img, label in zip(images, labels):
        canvas.paste(img, (x, 0))
        draw.text((x + w // 2, h + label_h // 2), label, fill=(0, 0, 0), font=font, anchor="mm")
        x += w + gap
    return canvas


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=Path, default=here / "outputs" / "baseline_resnet34",
                    help="folder containing seed_<N>/best.pt for each seed (e.g. outputs/baseline_resnet34)")
    ap.add_argument("--seeds", type=int, nargs=3, default=[123, 456, 789],
                    help="exactly 3 seeds - venn3 is a 3-set diagram")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "gradcam")
    args = ap.parse_args()

    out_dir = args.out_dir / "cross_seed_misclassified_mendeley_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load all 3 models + one GradCAM hook engine per model up front (each wraps that specific
    # model's layer4, so they can't be shared across models).
    models, cam_engines = {}, {}
    for seed in args.seeds:
        ckpt = args.arm_dir / f"seed_{seed}" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found")
        m = build_model_r34().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models[seed] = m
        cam_engines[seed] = GradCAM(m, m.layer4)

    items, _ = list_split(args.data_root / "mendeley" / "test", MENDELEY_DIRS)
    print(f"Scanning {len(items)} Mendeley test images across seeds {args.seeds} ...")

    # Cheap no-grad forward pass per image per seed, to find every seed's prediction first.
    per_seed_pred = {seed: {} for seed in args.seeds}  # seed -> {path: (pred_idx, confidence)}
    true_labels = {}
    for path, true_idx in items:
        true_labels[path] = true_idx
        img = preprocess(path)
        x = to_input(torch.from_numpy(img)[None], device)
        for seed in args.seeds:
            with torch.no_grad():
                logits = models[seed](x)
                probs = F.softmax(logits, dim=1)[0]
                pred_idx = logits.argmax(1).item()
            per_seed_pred[seed][path] = (pred_idx, probs[pred_idx].item())

    wrong_sets = {}
    for seed in args.seeds:
        wrong_sets[seed] = {p for p, (pred_idx, _) in per_seed_pred[seed].items()
                            if pred_idx != true_labels[p]}
        print(f"  seed {seed}: {len(wrong_sets[seed])}/{len(items)} wrong")

    union_wrong = set.union(*wrong_sets.values())
    print(f"Union across all {len(args.seeds)} seeds: {len(union_wrong)} uniquely-wrong images")

    # --- Venn diagram ---
    fig, ax = plt.subplots(figsize=(7, 7))
    venn3([wrong_sets[s] for s in args.seeds],
         set_labels=[f"seed {s}" for s in args.seeds], ax=ax)
    ax.set_title("Mendeley test images misclassified, by seed")
    fig.tight_layout()
    fig.savefig(out_dir / "venn_diagram.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'venn_diagram.png'}")

    # --- Grad-CAM (all 3 seeds, each its own predicted class) + summary CSV ---
    rows = []
    for path in sorted(union_wrong, key=lambda p: p.name):
        true_idx = true_labels[path]
        true_label = CLASSES[true_idx]
        wrong_seeds = [s for s in args.seeds if path in wrong_sets[s]]

        row = {"path": str(path), "true": true_label, "n_seeds_wrong": len(wrong_seeds),
              "wrong_seeds": ",".join(str(s) for s in wrong_seeds)}
        for seed in args.seeds:
            pi, pc = per_seed_pred[seed][path]
            row[f"seed_{seed}_pred"] = CLASSES[pi]
            row[f"seed_{seed}_confidence"] = round(pc, 4)
        rows.append(row)

        img = preprocess(path)
        x = to_input(torch.from_numpy(img)[None], device)

        panels, captions = [], []
        for seed in args.seeds:
            pred_idx, pred_conf = per_seed_pred[seed][path]
            pred_label = CLASSES[pred_idx]
            correct = pred_idx == true_idx
            cam, _ = cam_engines[seed](x, pred_idx)  # each seed's CAM for its OWN prediction
            panels.append(overlay(img, cam))
            captions.append(f"Seed {seed}: {pred_label} ({pred_conf:.2f}) "
                            f"- {'Correct' if correct else 'Wrong'}")

        combined = stitch_n_with_labels(panels, captions)
        seeds_tag = "-".join(str(s) for s in wrong_seeds)
        fname = f"{path.stem}__true-{true_label}__wrongseeds-{seeds_tag}.png"
        combined.save(out_dir / fname)
        print(f"  {path.name}: true={true_label} wrong_seeds={wrong_seeds}")

    for seed in args.seeds:
        cam_engines[seed].remove()

    pd.DataFrame(rows).to_csv(out_dir / "cross_seed_misclassified_summary.csv", index=False)
    print(f"\nSaved {len(union_wrong)} stitched overlay images + "
          f"cross_seed_misclassified_summary.csv + venn_diagram.png to {out_dir}")


if __name__ == "__main__":
    main()