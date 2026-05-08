"""Image corruption functions for CIFAKE robustness evaluation."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from .config import GAUSSIAN_BLUR_SIGMAS, GAUSSIAN_NOISE_SIGMAS, JPEG_QUALITIES


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression to a PIL image and return decompressed RGB image."""
    if quality not in JPEG_QUALITIES:
        raise ValueError(f"Unsupported JPEG quality: {quality}. Expected one of {JPEG_QUALITIES}.")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur using PIL radius mapped from sigma."""
    if sigma not in GAUSSIAN_BLUR_SIGMAS:
        raise ValueError(f"Unsupported blur sigma: {sigma}. Expected one of {GAUSSIAN_BLUR_SIGMAS}.")
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Apply additive Gaussian noise where sigma is in [0,1] image scale."""
    if sigma not in GAUSSIAN_NOISE_SIGMAS:
        raise ValueError(f"Unsupported noise sigma: {sigma}. Expected one of {GAUSSIAN_NOISE_SIGMAS}.")
    arr = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    arr_noisy = np.clip(arr + noise, 0.0, 1.0)
    out = (arr_noisy * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def corruption_grid() -> dict[str, tuple[float | int, ...]]:
    """Return the exact corruption settings used in Part 1 baseline."""
    return {
        "jpeg": JPEG_QUALITIES,
        "gaussian_blur": GAUSSIAN_BLUR_SIGMAS,
        "gaussian_noise": GAUSSIAN_NOISE_SIGMAS,
    }
