"""
Augmentation transforms for section 4 of the spec: the baseline geometric/compression pipeline
(table 4.2) plus each additive arm (table 4.3). All of this runs on CPU (numpy / OpenCV /
albumentations / torchio / scipy have no GPU path here) and operates on a single 2D uint8/float32
grayscale array, BEFORE percentile clipping and the Otsu z-score.

Rician noise, Gaussian noise, k-space motion blur, and the bias field are ported directly from
the user's own tested scripts (riciannoise.py, gaussnoise.py, myksblur.py, bfieldsim.py), adapted
from PIL-Image-in/PIL-Image-out to operate on the raw 2D array mid-pipeline, and to draw their
randomness from Python's/NumPy's *global* RNG state (rather than each instantiating its own
np.random.default_rng(seed)) so that this project's worker_init_fn seeding (spec section 7)
still governs them. The underlying math is unchanged from the original scripts.

Only `train` images ever pass through this module. Val/test/Kaggle are not augmented, so they
keep using train_noaug.py's cached, precomputed .npz arrays untouched.

Usage:
    from augmentations import build_baseline, ARMS, needs_post_zscore

    baseline = build_baseline()                  # albumentations.Compose
    extra = ARMS["rician_noise"]                 # callable: (img: np.ndarray) -> np.ndarray
    img = baseline(image=img[..., None])["image"][..., 0]
    img = extra(img)
    ... percentile clip -> Otsu z-score ...
    if needs_post_zscore("brightness_contrast"):
      img = ARMS_POST_ZSCORE["brightness_contrast"](img)   # applied AFTER the z-score instead
"""
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torchio as tio
from scipy.ndimage import rotate as _ndi_rotate
from scipy.ndimage import shift as _ndi_shift

# ----------------------------------------------------------------------------
# 4.2 Baseline augmentations (order matters; applied in this order, pre-clip)
# ----------------------------------------------------------------------------
def build_baseline() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.10, 0.10), rotate=(-10, 10), scale=(0.9, 1.1), p=0.5),
        A.ImageCompression(quality_range=(40, 90), p=0.2),
        RandomAxisCrop(min_frac=0.8, max_frac=0.95, p=0.3),
        A.Downscale(scale_range=(1 / 2.5, 1 / 1.5), interpolation_pair={"downscale": cv2.INTER_LINEAR,
                                                                        "upscale": cv2.INTER_LINEAR}, p=0.2),
    ])


class _FinalPipeline:
    """The settled combined pipeline: baseline's geometric/digital steps plus elastic
    deformation, ghosting and gaussian blur, in a specific, non-arbitrary order (see the
    reasoning in the project discussion - not reproduced here, just the resulting order):

      Stage 1 (geometric/anatomical, global pose then local tissue warp):
          Horizontal flip -> Affine -> Elastic deformation
      Stage 2 (acquisition-domain artifacts, act on the now-fixed anatomy):
          Ghosting -> Gaussian blur
      Stage 3 (digital/file-level degradation, act on the final reconstructed image):
          Downsample->Upsample -> JPEG compression -> Crop

    Brightness/contrast is NOT part of this class - it stays a separate, post-z-score step
    (see AugmentedMendeleyTrain in train_aug.py), same as for every other arm.

    A plain class rather than a closure, so DataLoader workers can pickle it on Windows (see
    the _BiasFieldArm comment earlier in this file for why that matters).
    """
    def __init__(self):
        self.geo = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.10, 0.10), rotate=(-10, 10), scale=(0.9, 1.1), p=0.5),
            # p=0.2 here (vs 0.3 for the standalone elastic_deform arm) - lowered specifically
            # for this settled pipeline.
            A.ElasticTransform(displacement_range=(0.09, 0.12), control_grid_shape=(5, 5), p=0.2),
        ])
        self.digital = A.Compose([
            A.Downscale(scale_range=(1 / 2.5, 1 / 1.5), interpolation_pair={"downscale": cv2.INTER_LINEAR,
                                                                            "upscale": cv2.INTER_LINEAR}, p=0.2),
            A.ImageCompression(quality_range=(40, 90), p=0.2),
            RandomAxisCrop(min_frac=0.8, max_frac=0.95, p=0.3),
        ])

    def __call__(self, img):
        img = self.geo(image=img[..., None])["image"][..., 0]
        img = _ghosting(img)          # unchanged, p=0.1 (same as the standalone ghosting arm)
        img = _combined_blur(img)     # p=0.15 (vs 0.2 for the standalone gaussian_blur arm)
        img = self.digital(image=img[..., None])["image"][..., 0]
        return img


def build_final_pipeline() -> "_FinalPipeline":
    return _FinalPipeline()


class _FinalPipeline2:
    """combined2: a lighter follow-up to the combined pipeline, after combined still trailed
    baseline on Kaggle. Same stage order/reasoning as _FinalPipeline, with these changes:
      - Elastic deformation: same p=0.2 as combined, but displacement_range halved to
        elastic_deform2's magnitude (0.045-0.06 instead of 0.09-0.12).
      - Ghosting: swapped for ghosting2's halved intensity (0.25-0.5 instead of 0.5-1.0),
        same p=0.1.
      - Gaussian blur: swapped for gaussian_blur2's halved sigma (0.25-0.75 instead of
        0.5-1.5), AND p lowered further to 0.1 (from combined's 0.15).
      - Brightness/contrast: removed entirely (not applied at all) - preprocessing already
        normalises brightness/contrast per image (percentile clip + Otsu z-score), so this
        augmentation was judged to add little here.
      - Downsample->Upsample, JPEG compression, Crop: unchanged from combined.

    Again a plain class, not a closure, for the same Windows-pickling reason as _FinalPipeline
    and _BiasFieldArm.
    """
    def __init__(self):
        self.geo = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.10, 0.10), rotate=(-10, 10), scale=(0.9, 1.1), p=0.5),
            A.ElasticTransform(displacement_range=(0.045, 0.06), control_grid_shape=(5, 5), p=0.2),
        ])
        self.digital = A.Compose([
            A.Downscale(scale_range=(1 / 2.5, 1 / 1.5), interpolation_pair={"downscale": cv2.INTER_LINEAR,
                                                                            "upscale": cv2.INTER_LINEAR}, p=0.2),
            A.ImageCompression(quality_range=(40, 90), p=0.2),
            RandomAxisCrop(min_frac=0.8, max_frac=0.95, p=0.3),
        ])

    def __call__(self, img):
        img = self.geo(image=img[..., None])["image"][..., 0]
        img = _ghosting2(img)          # p=0.1, intensity=(0.25, 0.5) - halved magnitude
        img = _combined2_blur(img)     # p=0.1, sigma=(0.25, 0.75) - halved magnitude, lower p
        img = self.digital(image=img[..., None])["image"][..., 0]
        return img
        # no brightness/contrast step - this pipeline ends here, before the z-score


def build_final_pipeline2() -> "_FinalPipeline2":
    return _FinalPipeline2()


class RandomAxisCrop(A.ImageOnlyTransform):
    """Keep a random 0.8-0.95 fraction of each axis independently, then let the pipeline's
    final resize (to 224x224) restore the output size. Matches "keep 0.8-0.95 of original in
    both axis" from the spec (albumentations has no single built-in for independent-axis crop
    fractions)."""
    def __init__(self, min_frac=0.8, max_frac=0.95, p=0.5):
        super().__init__(p=p)
        self.min_frac, self.max_frac = min_frac, max_frac

    def apply(self, img, **kwargs):
        h, w = img.shape[:2]
        fh = random.uniform(self.min_frac, self.max_frac)
        fw = random.uniform(self.min_frac, self.max_frac)
        ch, cw = max(1, int(round(h * fh))), max(1, int(round(w * fw)))
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        return img[top:top + ch, left:left + cw]

    def get_transform_init_args_names(self):
        return ("min_frac", "max_frac")


# ----------------------------------------------------------------------------
# torchio helper: torchio expects (C, W, H, D); we fake a unit depth axis and squeeze back.
# ----------------------------------------------------------------------------
def apply_torchio(transform, img: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(img.astype(np.float32))[None, :, :, None]  # (C=1, H, W, D=1)
    out = transform(t)
    return out[0, :, :, 0].numpy()


def _global_seed() -> int:
    """Draw a seed from Python's global `random` module (already seeded per-worker by
    worker_init_fn), so a function that wants its own np.random.default_rng(seed) still
    behaves reproducibly under this project's seeding scheme."""
    return random.randint(0, 2**32 - 1)


# ----------------------------------------------------------------------------
# Ported from riciannoise.py - unchanged formula, adapted to operate on the 2D array in the
# pipeline (rather than a PIL Image) and clipped/rounded back to uint8 the same way the
# original script does before saving.
# ----------------------------------------------------------------------------
def _rician_noise(img):
    if random.random() < 0.3:
        std = random.uniform(5, 15)
        n1 = np.random.normal(loc=0.0, scale=std, size=img.shape)
        n2 = np.random.normal(loc=0.0, scale=std, size=img.shape)
        noisy = np.sqrt((img.astype(np.float64) + n1) ** 2 + n2 ** 2)
        img = np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------
# Ported from gaussnoise.py - unchanged formula.
# ----------------------------------------------------------------------------
def _gaussian_noise(img):
    if random.random() < 0.3:
        std = random.uniform(5, 15)
        noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
        noisy = img.astype(np.float64) + noise
        img = np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)
    return img


def _gaussian_noise2(img):
    """Weaker gaussian_noise: std halved (2.5-7.5 instead of 5-15), same trigger probability."""
    if random.random() < 0.3:
        std = random.uniform(2.5, 7.5)
        noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
        noisy = img.astype(np.float64) + noise
        img = np.rint(np.clip(noisy, 0, 255)).astype(np.uint8)
    return img


# ----------------------------------------------------------------------------
# Ported from myksblur.py - unchanged algorithm (rigid poses -> per-pose FFT -> composite
# k-space assembled from phase-encode bands -> inverse FFT). random.uniform/randint replace the
# script's local np.random.default_rng so worker_init_fn seeding still governs it.
# ----------------------------------------------------------------------------
def _make_pose(image: np.ndarray, max_degrees: float, max_translation: float) -> np.ndarray:
    angle = random.uniform(-max_degrees, max_degrees)
    dy = random.uniform(-max_translation, max_translation)
    dx = random.uniform(-max_translation, max_translation)
    posed = _ndi_rotate(image, angle, reshape=False, order=1, mode="reflect")
    posed = _ndi_shift(posed, shift=(dy, dx), order=1, mode="reflect")
    return posed


def _kspace_motion_blur(img):
    """2-4 movements, +/-4deg rotation, max translation 5px per movement (spec table 4.3)."""
    if random.random() < 0.1:
        image = img.astype(np.float64)
        num_transforms = random.randint(2, 4)
        phase_encode_axis = 0
        n_lines = image.shape[phase_encode_axis]

        poses = [image] + [_make_pose(image, max_degrees=4.0, max_translation=5.0)
                           for _ in range(num_transforms - 1)]
        kspaces = [np.fft.fftshift(np.fft.fft2(p)) for p in poses]

        weights = np.ones(num_transforms) / num_transforms  # equal time per pose
        band_edges = np.round(np.cumsum(weights) * n_lines).astype(int)
        band_edges = np.concatenate([[0], band_edges])

        composite = np.zeros_like(kspaces[0], dtype=complex)
        for i in range(num_transforms):
            lo, hi = band_edges[i], band_edges[i + 1]
            if phase_encode_axis == 0:
                composite[lo:hi, :] = kspaces[i][lo:hi, :]
            else:
                composite[:, lo:hi] = kspaces[i][:, lo:hi]

        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(composite)))
        reconstructed = np.clip(reconstructed / (reconstructed.max() + 1e-8) * image.max(), 0, 255)
        img = reconstructed.astype(np.uint8)
    return img


# ----------------------------------------------------------------------------
# Ported from bfieldsim.py - unchanged algorithm (order-3 Legendre polynomial over
# square-to-circle mapped coordinates, field = exp(p(x,y))). `intensity` is bfieldsim's own
# parameter, empirically calibrated (200-sample Monte Carlo on 224x224 fields) so the field's
# mean max value lands near the spec's "max magnitude": intensity=0.50 -> mean max ~1.40,
# intensity=0.25 -> mean max ~1.18. Being a random polynomial, any single draw varies around
# that mean rather than hitting it exactly (matches "max magnitude" as a target, not a hard cap,
# consistent with how the spec's other stochastic augmentations behave).
# ----------------------------------------------------------------------------
_BF_TERM_ORDERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
_BF_BASE_COEFFS = np.array([0.525, 0.375, 0.375, 0.525, 0.45, 0.525, 0.225, 0.15, 0.15, 0.225])


def _bf_legendre(n, t):
    if n == 0:
        return np.ones_like(t)
    elif n == 1:
        return t
    elif n == 2:
        return 0.5 * (3 * t**2 - 1)
    elif n == 3:
        return 0.5 * (5 * t**3 - 3 * t)
    raise ValueError("Only L0-L3 needed here")


def _bf_make_field(width, height, seed, intensity):
    rng = np.random.default_rng(seed)
    k_scalar = 2 * rng.random(10) - 1  # 10 independent values in [-1, 1]
    c = k_scalar * _BF_BASE_COEFFS * intensity

    xs = np.linspace(-1, 1, width)
    ys = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(xs, ys)

    # square-to-circle mapping (Fernandez-Guasti style)
    Xc = X * np.sqrt(1 - (Y**2) / 2)
    Yc = Y * np.sqrt(1 - (X**2) / 2)

    Lx = {n: _bf_legendre(n, Xc) for n in range(4)}
    Ly = {n: _bf_legendre(n, Yc) for n in range(4)}

    p = np.zeros_like(X)
    for coeff, (nx, ny) in zip(c, _BF_TERM_ORDERS):
        p += coeff * Lx[nx] * Ly[ny]

    return np.exp(p)


class _BiasFieldArm:
    """A plain class (not a closure) so DataLoader workers can pickle it - needed on Windows,
    where multiprocessing's default 'spawn' start method requires pickling the Dataset (and
    everything it references) to hand to worker processes. A closure - a function returned from
    another function - can't be pickled by the standard library; a module-level class instance
    can. Linux/Colab's default 'fork' start method doesn't pickle at all (it copies process
    memory directly), which is why a closure version worked fine there but broke on Windows."""
    def __init__(self, intensity):
        self.intensity = intensity

    def __call__(self, img, p=0.3):
        if random.random() < p:
            h, w = img.shape
            z = _bf_make_field(w, h, seed=_global_seed(), intensity=self.intensity)
            result = img.astype(np.float64) * z
            img = np.clip(result, 0, 255).astype(np.uint8)
        return img


_bias_field_14 = _BiasFieldArm(0.50)   # calibrated for ~1.4x max magnitude
_bias_field_12 = _BiasFieldArm(0.25)   # ~1.2x max magnitude (mean max ~1.19; see calibration note above)


# ----------------------------------------------------------------------------
# Remaining arms (not covered by an uploaded script): unchanged from the earlier version.
# ----------------------------------------------------------------------------
_elastic_transform = A.ElasticTransform(displacement_range=(0.09, 0.12), control_grid_shape=(5, 5), p=0.3)
_elastic_transform2 = A.ElasticTransform(displacement_range=(0.045, 0.06), control_grid_shape=(5, 5), p=0.3)


def _elastic_deform(img):
    return _elastic_transform(image=img[..., None])["image"][..., 0]


def _elastic_deform2(img):
    """Weaker elastic_deform: displacement_range halved (0.045-0.06 instead of 0.09-0.12)."""
    return _elastic_transform2(image=img[..., None])["image"][..., 0]


def _gamma(img):
    if random.random() < 0.4:
        img = apply_torchio(tio.RandomGamma(log_gamma=(-0.1, 0.1)), img)
    return img


def _gaussian_blur(img):
    if random.random() < 0.2:
        sigma = random.uniform(0.5, 1.5)
        k = max(3, int(2 * round(3 * sigma) + 1))  # odd kernel covering ~3 sigma
        img = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma)
    return img


def _combined_blur(img):
    """Used only inside the combined pipeline: same sigma range as _gaussian_blur, but
    p=0.15 (vs 0.2 for the standalone gaussian_blur arm) - lowered specifically for this
    settled pipeline."""
    if random.random() < 0.15:
        sigma = random.uniform(0.5, 1.5)
        k = max(3, int(2 * round(3 * sigma) + 1))
        img = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma)
    return img


def _combined2_blur(img):
    """Used only inside the combined2 pipeline: gaussian_blur2's sigma range (0.25-0.75), at
    p=0.1 (vs 0.2 for the standalone gaussian_blur2 arm, and 0.15 in the original combined
    pipeline) - lowered further specifically for this pipeline."""
    if random.random() < 0.1:
        sigma = random.uniform(0.25, 0.75)
        k = max(3, int(2 * round(3 * sigma) + 1))
        img = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma)
    return img


def _gaussian_blur2(img):
    """Weaker gaussian_blur: sigma halved (0.25-0.75 instead of 0.5-1.5)."""
    if random.random() < 0.2:
        sigma = random.uniform(0.25, 0.75)
        k = max(3, int(2 * round(3 * sigma) + 1))
        img = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma)
    return img


def _spike(img):
    if random.random() < 0.1:
        img = apply_torchio(tio.RandomSpike(num_spikes=1, intensity=(0.1, 0.5)), img)
    return img


def _ghosting(img):
    if random.random() < 0.1:
        img = apply_torchio(tio.RandomGhosting(num_ghosts=4, intensity=(0.5, 1.0)), img)
    return img


def _ghosting2(img):
    """Weaker ghosting: intensity halved (0.25-0.5 instead of 0.5-1.0), same num_ghosts."""
    if random.random() < 0.1:
        img = apply_torchio(tio.RandomGhosting(num_ghosts=4, intensity=(0.25, 0.5)), img)
    return img


def _brightness_contrast_post_zscore(img):
    """Applied AFTER the Otsu z-score (spec 4.3), directly on z-scored values: x' = c*x + b."""
    if random.random() < 0.5:
        c = random.uniform(0.8, 1.2)
        b = random.uniform(-0.15, 0.15)
        img = c * img + b
    return img


def _combined_brightness_contrast(img):
    """Used only inside the combined pipeline: same c/b ranges as _brightness_contrast_post_zscore,
    but p=0.3 (vs 0.5 for the standalone brightness_contrast arm) - lowered specifically for this
    settled pipeline. Still applied after the z-score, same as the standalone version."""
    if random.random() < 0.3:
        c = random.uniform(0.8, 1.2)
        b = random.uniform(-0.15, 0.15)
        img = c * img + b
    return img


# Pre-clip additive arms (applied right after the baseline transform)
ARMS = {
    "elastic_deform": _elastic_deform,
    "elastic_deform2": _elastic_deform2,
    "gamma": _gamma,
    "gaussian_blur": _gaussian_blur,
    "gaussian_blur2": _gaussian_blur2,
    "rician_noise": _rician_noise,
    "gaussian_noise": _gaussian_noise,
    "gaussian_noise2": _gaussian_noise2,
    "kspace_motion_blur": _kspace_motion_blur,
    "bias_field": _bias_field_14,
    "bias_field2": _bias_field_12,
    "spike": _spike,
    "ghosting": _ghosting,
    "ghosting2": _ghosting2,
}

# Post-z-score arms (applied instead, after normalisation)
ARMS_POST_ZSCORE = {
    "brightness_contrast": _brightness_contrast_post_zscore,
}

ALL_ARM_NAMES = ["baseline", "combined", "combined2", "combined3"] + list(ARMS.keys()) + list(ARMS_POST_ZSCORE.keys())
# combined3 needs no dedicated pipeline: it's baseline's exact augmentation recipe (ARMS.get and
# ARMS_POST_ZSCORE.get both return None for an unrecognised name, so train_aug.py's default
# branch already builds plain build_baseline() with no extra step). It exists purely so baseline
# can be retrained under the new ES_PATIENCE=15 / ES_MIN_DELTA=0.003 protocol, saved separately
# from the original baseline run (which used the old patience=10 / min_delta=0.005), so the two
# training protocols can be compared on equal footing against combined/combined2.


def needs_post_zscore(arm: str) -> bool:
    return arm in ARMS_POST_ZSCORE