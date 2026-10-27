"""
Runs the two-stage pipeline (binary ResNet-18 Tumor/No-Tumor -> baseline ResNet-34 seed 123
sub-classification, with fallback to 2nd-highest confidence if stage 2 contradicts stage 1) on
a single image, prints the prediction and confidence, and saves a Grad-CAM overlay.

Grad-CAM is taken from whichever model actually decided the final prediction:
  - if the binary gate says "no tumour", the CAM shown is the binary model's own CAM for that
    decision (its layer4, its own 2-class output space)
  - if a tumour type is predicted (directly, or via the fallback rule), the CAM shown is
    ResNet-34's CAM for that specific class

Output goes in a dedicated subfolder: outputs/gradcam/single_predictions/

Usage:
    python predict_single.py --image path/to/scan.jpg
    python predict_single.py --image path/to/scan.jpg --bin-ckpt outputs/binary_resnet18/seed_123/best.pt \
        --r34-ckpt outputs/baseline_resnet34/seed_123/best.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from gradcam_resnet34 import GradCAM, overlay
from train_noaug import CLASSES, preprocess, to_input
from train_resnet34 import build_model_r34

NO_TUMOR_IDX = CLASSES.index("no_tumor")
BINARY_LABELS = ["Tumor", "No Tumor"]  # bin_model's own 2-class output space: 0=Tumor, 1=No Tumor


def build_model_binary():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bin-ckpt", type=Path, default=None)
    ap.add_argument("--r34-ckpt", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "gradcam")
    args = ap.parse_args()
    if args.bin_ckpt is None:
        args.bin_ckpt = here / "outputs" / "binary_resnet18" / "seed_123" / "best.pt"
    if args.r34_ckpt is None:
        args.r34_ckpt = here / "outputs" / "baseline_resnet34" / "seed_123" / "best.pt"

    out_dir = args.out_dir / "single_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    bin_model = build_model_binary().to(device)
    bin_model.load_state_dict(torch.load(args.bin_ckpt, map_location=device))
    bin_model.eval()

    r34_model = build_model_r34().to(device)
    r34_model.load_state_dict(torch.load(args.r34_ckpt, map_location=device))
    r34_model.eval()

    bin_cam = GradCAM(bin_model, bin_model.layer4)
    r34_cam = GradCAM(r34_model, r34_model.layer4)

    img = preprocess(args.image)  # percentile clip + Otsu z-score + resize to 224, same as training
    x = to_input(torch.from_numpy(img)[None], device)

    with torch.no_grad():
        bin_pred = bin_model(x).argmax(1).item()

    if bin_pred == 1:
        # Stage 1 says No Tumor - this IS the final decision; show stage 1's own CAM for it.
        final_class = NO_TUMOR_IDX
        cam_map, bin_logits = bin_cam(x, target_idx=1)
        confidence = F.softmax(bin_logits, dim=1)[0, 1].item()
        decided_by = "binary gate (Stage 1)"
        fallback = False
    else:
        # Stage 1 says Tumor - defer to Stage 2 (ResNet-34) for the specific type.
        with torch.no_grad():
            r34_logits_peek = r34_model(x)
        r34_pred = r34_logits_peek.argmax(1).item()

        if r34_pred == NO_TUMOR_IDX:
            # Contradiction: Stage 2 said No Tumor despite Stage 1 saying Tumor. Fall back to
            # Stage 2's 2nd-highest class instead.
            logits = r34_logits_peek.clone()
            logits[0, NO_TUMOR_IDX] = -float("inf")
            final_class = logits.argmax(1).item()
            fallback = True
        else:
            final_class = r34_pred
            fallback = False

        cam_map, r34_logits = r34_cam(x, target_idx=final_class)
        confidence = F.softmax(r34_logits, dim=1)[0, final_class].item()
        decided_by = "ResNet-34 (Stage 2)" + (", via fallback rule" if fallback else "")

    bin_cam.remove()
    r34_cam.remove()

    out_img = overlay(img, cam_map)
    fname = f"{args.image.stem}__pred-{CLASSES[final_class]}({confidence:.2f}).png"
    out_img.save(out_dir / fname)

    print(f"Image:       {args.image.name}")
    print(f"Prediction:  {CLASSES[final_class]}")
    print(f"Confidence:  {confidence:.4f}")
    print(f"Decided by:  {decided_by}")
    print(f"Saved:       {out_dir / fname}")


if __name__ == "__main__":
    main()