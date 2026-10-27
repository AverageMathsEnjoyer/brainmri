"""
Runs Grad-CAM (see gradcam_resnet34.py) on every Mendeley TEST image the model gets wrong -
no sampling, the full set of misclassifications. For each one, saves ONE combined image: the
predicted-class overlay (what mislead it) and the true-class overlay (what evidence, if any,
was there for the right answer) side by side, with a caption under each half naming its class
and confidence - the single most informative comparison for inspecting an error, with no need
to re-run this twice with different --target-class values. Both overlays are built and stitched
entirely in memory (see stitch_with_labels()); only the final combined image is ever written to
disk, so no intermediate single-overlay files are left behind.

Reuses GradCAM and overlay() from gradcam_resnet34.py unmodified.

Output goes in a dedicated subfolder: <out-dir>/misclassified_mendeley_test/
(default <out-dir> is outputs/gradcam, same root the other Grad-CAM script uses).

Usage:
    python gradcam_misclassified.py --ckpt outputs/baseline_resnet34/seed_123/best.pt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from gradcam_resnet34 import GradCAM, overlay
from train_noaug import CLASSES, MENDELEY_DIRS, list_split, preprocess, to_input
from train_resnet34 import build_model_r34


def stitch_with_labels(left_img: Image.Image, right_img: Image.Image,
                       left_label: str, right_label: str,
                       label_h: int = 32, gap: int = 4, bg=(255, 255, 255)) -> Image.Image:
    """Combines two same-size PIL Images side by side with a text caption under each, entirely
    in memory (left_img/right_img are the in-memory overlay() outputs - nothing is written to
    disk until the single returned, combined image is saved)."""
    w, h = left_img.size
    canvas = Image.new("RGB", (w * 2 + gap, h + label_h), bg)
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (w + gap, 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((w // 2, h + label_h // 2), left_label, fill=(0, 0, 0), font=font, anchor="mm")
    draw.text((w + gap + w // 2, h + label_h // 2), right_label, fill=(0, 0, 0), font=font, anchor="mm")
    return canvas


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="path to a best.pt from train_resnet34.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", type=Path, default=here)
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "gradcam")
    args = ap.parse_args()

    out_dir = args.out_dir / "misclassified_mendeley_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_model_r34().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    cam_engine = GradCAM(model, model.layer4)

    items, _ = list_split(args.data_root / "mendeley" / "test", MENDELEY_DIRS)
    print(f"Scanning {len(items)} Mendeley test images for misclassifications ...")

    rows = []
    n_wrong = 0
    for path, true_idx in items:
        img = preprocess(path)
        x = to_input(torch.from_numpy(img)[None], device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = logits.argmax(1).item()

        if pred_idx == true_idx:
            continue
        n_wrong += 1

        true_label, pred_label = CLASSES[true_idx], CLASSES[pred_idx]
        rows.append({"path": str(path), "true": true_label, "pred": pred_label,
                     "pred_confidence": probs[pred_idx].item(),
                     "true_class_confidence": probs[true_idx].item()})

        # Two overlays per error: what the model actually acted on, and what evidence (if any)
        # existed for the class it should have picked. Both built in memory and stitched into
        # ONE side-by-side image with a caption under each half - no separate files are ever
        # written for the two halves, only the final combined image.
        cam_pred, _ = cam_engine(x, pred_idx)
        cam_true, _ = cam_engine(x, true_idx)
        img_pred = overlay(img, cam_pred)
        img_true = overlay(img, cam_true)
        combined = stitch_with_labels(
            img_pred, img_true,
            left_label=f"Predicted: {pred_label} ({probs[pred_idx]:.2f})",
            right_label=f"True: {true_label} ({probs[true_idx]:.2f})")
        fname = f"{path.stem}__true-{true_label}__pred-{pred_label}({probs[pred_idx]:.2f}).png"
        combined.save(out_dir / fname)

        print(f"  [{n_wrong}] {path.name}: true={true_label} pred={pred_label} "
              f"({probs[pred_idx]:.2f})")

    cam_engine.remove()

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "misclassified_summary.csv", index=False)
    print(f"\n{n_wrong}/{len(items)} misclassified ({n_wrong / len(items) * 100:.1f}%).")
    print(f"Saved {n_wrong} stitched overlay images + misclassified_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()