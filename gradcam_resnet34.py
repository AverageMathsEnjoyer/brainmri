"""
Grad-CAM for the ResNet-34 + baseline model (see train_resnet34.py), targeting layer4 - the
last convolutional block before global-average-pooling, and the standard Grad-CAM target for
ResNet-style architectures since its spatial resolution (7x7 for a 224x224 input, given ResNet's
32x total downsampling) still roughly corresponds to image regions.

Implemented manually with forward/backward hooks rather than an external grad-cam library, to
avoid adding a new dependency for a small, well-understood algorithm:
  1. A forward hook captures layer4's output activations (C, 7, 7).
  2. A backward hook captures the gradient of the target class's logit w.r.t. those activations.
  3. Each channel's gradient is global-average-pooled into one importance weight; the weighted
     sum of activation channels, ReLU'd, is the class activation map (the original Grad-CAM
     formulation: Selvaraju et al. 2017).
  4. The map is upsampled to 224x224 and alpha-blended over the (display-normalised) input image.

Target class defaults to the model's own predicted class per image (what did the model "look at"
to reach the decision it actually made); pass --target-class true to use the image's true label
instead (what WOULD make it predict correctly, useful for inspecting misclassifications), or
an explicit class name to fix one class for every image.

Usage:
    # a handful of images per class, sampled from Mendeley test
    python gradcam_resnet34.py --ckpt outputs/baseline_resnet34/seed_123/best.pt --split mendeley_test --per-class 3

    # specific images
    python gradcam_resnet34.py --ckpt outputs/baseline_resnet34/seed_123/best.pt --images path/to/a.jpg path/to/b.jpg

    # inspect misclassifications specifically: what the model looked at vs what it should have
    python gradcam_resnet34.py --ckpt outputs/baseline_resnet34/seed_123/best.pt --split kaggle_test --per-class 3 --target-class true
"""
import argparse
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_noaug import CLASSES, KAGGLE_DIRS, MENDELEY_DIRS, is_augmented, list_split, preprocess, to_input
from train_resnet34 import build_model_r34


# ----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        self._h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_idx: int) -> np.ndarray:
        """x: (1, 3, 224, 224) input tensor (requires no special grad setup from the caller).
        Returns a (224, 224) float32 array in [0, 1] - the upsampled, normalised CAM."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, target_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)    # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam, logits.detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()


# ----------------------------------------------------------------------------
def overlay(img_zscored: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """img_zscored: the (224, 224) preprocessed (percentile-clipped, z-scored) float array the
    model actually saw - NOT the raw original image, so this shows what the model's input looked
    like, not the unprocessed source file. Min-max normalised to 0-255 here purely for display."""
    disp = img_zscored - img_zscored.min()
    disp = disp / (disp.max() + 1e-8)
    disp_rgb = np.stack([disp] * 3, axis=-1)  # grayscale -> RGB

    heat = matplotlib.colormaps["jet"](cam)[..., :3]  # (224, 224, 3), in [0, 1]
    blended = (1 - alpha) * disp_rgb + alpha * heat
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ----------------------------------------------------------------------------
def gather_images(args, here: Path):
    """Returns a list of (path, true_class_idx_or_None)."""
    if args.images:
        return [(Path(p), None) for p in args.images]

    if args.split == "mendeley_test":
        items, _ = list_split(here / "mendeley" / "test", MENDELEY_DIRS)
    elif args.split == "kaggle_test":
        items, _ = list_split(here / "kaggle" / "Testing", KAGGLE_DIRS)
        items = [(p, c) for p, c in items if not is_augmented(p)]
    else:
        raise ValueError(f"unknown --split {args.split}")

    by_class = {c: [] for c in range(len(CLASSES))}
    for p, c in items:
        by_class[c].append(p)
    rng = random.Random(0)
    sampled = []
    for c, paths in by_class.items():
        chosen = rng.sample(paths, min(args.per_class, len(paths)))
        sampled += [(p, c) for p in chosen]
    return sampled


# ----------------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="path to a best.pt from train_resnet34.py")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--images", nargs="+", default=None, help="explicit image paths; overrides --split")
    ap.add_argument("--split", choices=["mendeley_test", "kaggle_test"], default="mendeley_test")
    ap.add_argument("--per-class", type=int, default=3, help="images sampled per class when using --split")
    ap.add_argument("--target-class", default="predicted",
                    help="'predicted' (default, what the model actually acted on), 'true' "
                         "(needs known labels - from --split, not --images), or a class name "
                         "from " + str(CLASSES) + " to fix one class for every image")
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "gradcam")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = build_model_r34().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()  # matches how the model is evaluated elsewhere (fixed BN stats, no dropout);
                  # does NOT disable gradients, which Grad-CAM needs - no torch.no_grad() here

    cam_engine = GradCAM(model, model.layer4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = gather_images(args, here)
    if not images:
        print("No images found.")
        return
    print(f"Running Grad-CAM on {len(images)} image(s), target-class={args.target_class}, "
          f"saving to {args.out_dir}")

    fixed_class_idx = CLASSES.index(args.target_class) if args.target_class in CLASSES else None

    for path, true_idx in images:
        img = preprocess(path)  # same percentile-clip + Otsu z-score + resize the model trained on
        x = to_input(torch.from_numpy(img)[None], device)  # (1, 3, 224, 224)

        if fixed_class_idx is not None:
            target_idx = fixed_class_idx
        elif args.target_class == "true":
            if true_idx is None:
                raise ValueError("--target-class true needs known labels - use --split, not --images")
            target_idx = true_idx
        else:  # "predicted"
            with torch.no_grad():
                target_idx = model(x).argmax(1).item()

        cam, logits = cam_engine(x, target_idx)
        pred_idx = logits.argmax(1).item()
        probs = F.softmax(logits, dim=1)[0]

        out_img = overlay(img, cam)
        true_label = CLASSES[true_idx] if true_idx is not None else "unknown"
        correct_tag = "" if true_idx is None else ("correct" if pred_idx == true_idx else "WRONG")
        fname = (f"{path.stem}__true-{true_label}__pred-{CLASSES[pred_idx]}"
                f"({probs[pred_idx]:.2f})__cam-{CLASSES[target_idx]}__{correct_tag}.png")
        out_img.save(args.out_dir / fname)
        print(f"  {path.name}: true={true_label} pred={CLASSES[pred_idx]} "
              f"({probs[pred_idx]:.2f}) cam-target={CLASSES[target_idx]} {correct_tag}")

    cam_engine.remove()
    print(f"\nDone. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()