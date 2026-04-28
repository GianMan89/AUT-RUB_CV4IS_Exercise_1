"""Utility functions for CV4IS Exercise 1.

These functions intentionally use common packages (NumPy, scikit-image, OpenCV)
and are written in a transparent style for teaching.
"""

from pathlib import Path
import math
import os
from pathlib import Path as _Path
# Use a writable Matplotlib config directory in Binder/sandbox environments.
os.environ.setdefault("MPLCONFIGDIR", str(_Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
_Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, img_as_float, exposure, filters, morphology, measure, transform, util
from scipy import ndimage as ndi


def load_gray(path):
    """Load an image as float grayscale in [0, 1]."""
    img = io.imread(path, as_gray=True)
    return img_as_float(img)


def show_images(images, titles=None, cmap="gray", ncols=None, figsize=(12, 5), vmin=None, vmax=None):
    """Display a list of images in a grid."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    n = len(images)
    if titles is None:
        titles = [""] * n
    if ncols is None:
        ncols = n
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_histograms(images, titles=None, bins=64, figsize=(12, 4)):
    """Plot intensity histograms for a list of grayscale images in [0, 1]."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    if titles is None:
        titles = [""] * len(images)
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.hist(np.asarray(img).ravel(), bins=bins, range=(0, 1))
        ax.set_title(title)
        ax.set_xlabel("intensity")
        ax.set_ylabel("count")
    plt.tight_layout()
    return fig


def mm_per_pixel(fov_mm, n_pixels):
    """Object-space resolution in mm/px."""
    return float(fov_mm) / float(n_pixels)


def pixels_per_feature(feature_size_mm, mm_per_px):
    """How many pixels span a physical feature."""
    return float(feature_size_mm) / float(mm_per_px)


def downsample_image(image, scale, anti_aliasing):
    """Downsample image by a given scale using scikit-image transform.resize."""
    image = np.asarray(image)
    new_shape = (max(1, int(round(image.shape[0] * scale))),
                 max(1, int(round(image.shape[1] * scale))))
    return transform.resize(image, new_shape, anti_aliasing=anti_aliasing, preserve_range=True)


def motion_blur_kernel(length=15, angle_deg=0):
    """Create a simple normalized linear motion blur kernel."""
    length = int(length)
    if length < 1:
        raise ValueError("length must be >= 1")
    kernel = np.zeros((length, length), dtype=float)
    kernel[length // 2, :] = 1.0
    kernel = ndi.rotate(kernel, angle_deg, reshape=False, order=1)
    s = kernel.sum()
    if s == 0:
        kernel[length // 2, :] = 1.0
        s = kernel.sum()
    return kernel / s


def apply_motion_blur(image, length=15, angle_deg=0):
    """Apply a simple linear motion blur to a grayscale image."""
    kernel = motion_blur_kernel(length, angle_deg)
    return ndi.convolve(image, kernel, mode="reflect")


def add_gaussian_noise(image, sigma=0.05, seed=0):
    """Add Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy = np.asarray(image) + rng.normal(0, sigma, np.asarray(image).shape)
    return np.clip(noisy, 0, 1)


def gamma_correct(image, gamma=2.2):
    """Apply gamma correction y = x^(1/gamma) to image in [0, 1]."""
    image = np.clip(np.asarray(image), 0, 1)
    return image ** (1.0 / gamma)


def quantize_image(image, bits=8):
    """Quantize an image in [0, 1] to a given bit depth and return float image."""
    levels = 2 ** int(bits)
    q = np.round(np.clip(image, 0, 1) * (levels - 1)) / (levels - 1)
    return q


def sobel_energy(image):
    """Simple edge-strength proxy based on Sobel magnitude."""
    edge = filters.sobel(image)
    return float(np.mean(edge))


def laplacian_variance(image):
    """Sharpness proxy: variance of a Laplacian-filtered image."""
    lap = ndi.laplace(np.asarray(image).astype(float))
    return float(np.var(lap))


def otsu_segment(image, foreground="bright"):
    """Segment a grayscale image using Otsu threshold."""
    thr = filters.threshold_otsu(image)
    if foreground == "bright":
        mask = image > thr
    elif foreground == "dark":
        mask = image < thr
    else:
        raise ValueError("foreground must be 'bright' or 'dark'")
    return mask, float(thr)


def clean_mask(mask, min_size=50, closing_radius=2):
    """Remove small connected components and close gaps in a binary mask.

    Implemented explicitly to avoid version-dependent warnings in scikit-image.
    """
    mask = np.asarray(mask).astype(bool)
    labels = measure.label(mask)
    cleaned = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area >= min_size:
            cleaned[labels == region.label] = True
    if closing_radius > 0:
        selem = morphology.disk(closing_radius)
        cleaned = morphology.closing(cleaned, selem)
    return cleaned


def region_table(mask, mm_per_px=None):
    """Measure connected components.

    If mm_per_px is provided, adds area_mm2.
    """
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    rows = []
    for i, p in enumerate(props, 1):
        row = {
            "region": i,
            "area_px": int(p.area),
            "centroid_row": float(p.centroid[0]),
            "centroid_col": float(p.centroid[1]),
            "bbox_min_row": int(p.bbox[0]),
            "bbox_min_col": int(p.bbox[1]),
            "bbox_max_row": int(p.bbox[2]),
            "bbox_max_col": int(p.bbox[3]),
        }
        if mm_per_px is not None:
            row["area_mm2"] = float(p.area) * float(mm_per_px) ** 2
        rows.append(row)
    return rows


def iou(mask_a, mask_b):
    """Intersection-over-union between two binary masks."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)
# -----------------------------
# Advanced Exercise 1 utilities
# -----------------------------

def contrast_between_regions(image, foreground_mask, background_mask, eps=1e-12):
    """Michelson-like contrast between two regions.

    Returns (mu_fg - mu_bg) / (mu_fg + mu_bg + eps).
    """
    image = np.asarray(image, dtype=float)
    fg = np.asarray(foreground_mask).astype(bool)
    bg = np.asarray(background_mask).astype(bool)
    mu_fg = float(np.mean(image[fg]))
    mu_bg = float(np.mean(image[bg]))
    return (mu_fg - mu_bg) / (mu_fg + mu_bg + eps)


def cnr_between_regions(image, foreground_mask, background_mask, eps=1e-12):
    """Contrast-to-noise ratio between two regions.

    CNR = |mu_fg - mu_bg| / sqrt(sigma_fg^2 + sigma_bg^2)
    """
    image = np.asarray(image, dtype=float)
    fg = np.asarray(foreground_mask).astype(bool)
    bg = np.asarray(background_mask).astype(bool)
    mu_fg = float(np.mean(image[fg]))
    mu_bg = float(np.mean(image[bg]))
    sd_fg = float(np.std(image[fg]))
    sd_bg = float(np.std(image[bg]))
    return abs(mu_fg - mu_bg) / (np.sqrt(sd_fg**2 + sd_bg**2) + eps)


def weighted_gray(rgb, weights):
    """Create a weighted grayscale image from RGB image in [0, 1]."""
    rgb = np.asarray(rgb, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return np.clip(np.tensordot(rgb[..., :3], w, axes=([2], [0])), 0, 1)


def robust_zscore_map(image, median_image, mad_image, eps=1e-6):
    """Pixelwise robust absolute z-score map using median and MAD images."""
    return np.abs(np.asarray(image) - np.asarray(median_image)) / (1.4826 * np.asarray(mad_image) + eps)


def anomaly_score_from_map(score_map, percentile=99):
    """Image-level anomaly score as a high percentile of the score map."""
    return float(np.percentile(np.asarray(score_map), percentile))


def max_exposure_time_for_blur(blur_max_px, mm_per_px, object_speed_mm_s):
    """Maximum exposure time [s] for an allowed motion blur in pixels."""
    return float(blur_max_px) * float(mm_per_px) / float(object_speed_mm_s)


def required_pixels_for_feature(feature_size_mm, desired_pixels_per_feature):
    """Maximum mm/px required to achieve a desired number of pixels per feature."""
    return float(feature_size_mm) / float(desired_pixels_per_feature)


def required_fov_for_pixel_count(mm_per_px, n_pixels):
    """Field of view width from object-space resolution and number of pixels."""
    return float(mm_per_px) * float(n_pixels)
