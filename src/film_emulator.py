#!/usr/bin/env python3
"""
film_emulator.py

A starter film-emulation script using OpenCV + NumPy.
Features:
- filmic tone curve
- optional stock presets
- highlight bloom
- red halation
- vignette
- luminance-weighted grain
- optional .cube LUT support

Usage:
    python film_emulator.py input.jpg output.jpg --preset portra
    python film_emulator.py input.jpg output.jpg --preset cinestill --grain 0.08
    python film_emulator.py input.jpg output.jpg --lut my_lut.cube
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ----------------------------
# Utility
# ----------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Approximate sRGB -> linear."""
    threshold = 0.04045
    return np.where(
        img <= threshold,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Approximate linear -> sRGB."""
    threshold = 0.0031308
    return np.where(
        img <= threshold,
        img * 12.92,
        1.055 * np.power(img, 1 / 2.4) - 0.055
    )


def read_image(path: str) -> np.ndarray:
    """Read image as float32 RGB in [0,1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def write_image(path: str, img: np.ndarray) -> None:
    """Write float32 RGB [0,1] image."""
    out = (clamp01(img) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)
    if not ok:
        raise IOError(f"Could not write image: {path}")


# ----------------------------
# Tone / color
# ----------------------------

def apply_channel_mix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 color transform matrix to RGB image.
    img shape: H,W,3
    """
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    out = flat @ matrix.T
    return out.reshape(h, w, 3)


def filmic_curve(img: np.ndarray, contrast: float = 1.08, fade: float = 0.03) -> np.ndarray:
    """
    Gentle film-like tone curve:
    - slightly lifted blacks
    - softer highlight rolloff
    - mild contrast
    """
    img = clamp01(img)

    # soft shoulder
    shoulder = img / (img + 0.6)
    shoulder /= (1.0 / (1.0 + 0.6))

    # blend original and shoulder
    out = 0.55 * img + 0.45 * shoulder

    # mild contrast around middle gray
    out = (out - 0.5) * contrast + 0.5

    # faded blacks
    out = out * (1.0 - fade) + fade

    return clamp01(out)


def apply_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    """Adjust saturation in HSV."""
    hsv = cv2.cvtColor((clamp01(img) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return rgb


def stock_color_preset(name: str) -> tuple[np.ndarray, dict]:
    """
    Returns:
        color matrix
        default tuning values
    """
    presets: dict[str, tuple[np.ndarray, dict]] = {
        "neutral": (
            np.array([
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ], dtype=np.float32),
            dict(saturation=1.00, contrast=1.05, fade=0.02),
        ),
        "portra": (
            np.array([
                [1.03, -0.01, -0.02],
                [0.02, 0.99, -0.01],
                [0.01, 0.03, 0.96],
            ], dtype=np.float32),
            dict(saturation=0.94, contrast=1.04, fade=0.04),
        ),
        "velvia": (
            np.array([
                [1.04, 0.00, -0.02],
                [-0.01, 1.05, -0.01],
                [-0.01, 0.02, 1.03],
            ], dtype=np.float32),
            dict(saturation=1.22, contrast=1.12, fade=0.01),
        ),
        "cinestill": (
            np.array([
                [1.05, 0.00, -0.03],
                [0.01, 0.98, 0.01],
                [0.02, 0.01, 0.97],
            ], dtype=np.float32),
            dict(saturation=0.96, contrast=1.06, fade=0.03),
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {', '.join(presets)}")
    return presets[name]


# ----------------------------
# Effects
# ----------------------------

def add_bloom(img: np.ndarray, strength: float = 0.18, threshold: float = 0.72, blur_sigma: float = 12.0) -> np.ndarray:
    """Soft glow in highlights."""
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mask = np.clip((lum - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    glow_src = img * mask[..., None]

    ksize = int(max(3, round(blur_sigma * 4) | 1))
    blurred = cv2.GaussianBlur(glow_src, (ksize, ksize), blur_sigma)

    return clamp01(img + blurred * strength)


def add_halation(
    img: np.ndarray,
    strength: float = 0.20,
    threshold: float = 0.78,
    blur_sigma: float = 10.0
) -> np.ndarray:
    """
    Approximate film halation:
    - isolate strong highlights
    - blur them
    - bias toward red/orange
    """
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    mask = np.clip((lum - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)

    source = img * mask[..., None]
    ksize = int(max(3, round(blur_sigma * 4) | 1))
    blurred = cv2.GaussianBlur(source, (ksize, ksize), blur_sigma)

    halo = np.zeros_like(img)
    halo[..., 0] = blurred[..., 0] * 1.0   # R
    halo[..., 1] = blurred[..., 1] * 0.35  # G
    halo[..., 2] = blurred[..., 2] * 0.10  # B

    return clamp01(img + halo * strength)


def add_vignette(img: np.ndarray, strength: float = 0.20) -> np.ndarray:
    """Darken edges slightly."""
    h, w = img.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    x = (x - w / 2) / (w / 2)
    y = (y - h / 2) / (h / 2)
    r = np.sqrt(x * x + y * y)

    mask = 1.0 - strength * np.clip(r ** 1.8, 0.0, 1.0)
    return clamp01(img * mask[..., None])


def add_film_grain(
    img: np.ndarray,
    amount: float = 0.05,
    size: float = 1.0,
    color: float = 0.25,
    seed: int | None = None
) -> np.ndarray:
    """
    Add luminance-weighted grain.
    - more visible in mids/shadows
    - mostly monochrome, with a little color variation
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    gh = max(1, int(h / size))
    gw = max(1, int(w / size))

    mono = rng.normal(0.0, 1.0, (gh, gw, 1)).astype(np.float32)
    chroma = rng.normal(0.0, 1.0, (gh, gw, 3)).astype(np.float32)

    grain = mono * (1.0 - color) + chroma * color
    grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_CUBIC)

    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    weight = 1.0 - np.clip((lum - 0.2) / 0.8, 0.0, 1.0) * 0.55
    weight = weight[..., None]

    return clamp01(img + grain * amount * weight)


# ----------------------------
# LUT support (.cube)
# ----------------------------

def load_cube_lut(path: str) -> np.ndarray:
    """
    Load a simple 3D .cube LUT.
    Returns array of shape [size, size, size, 3].
    Assumes standard .cube layout.
    """
    lines = Path(path).read_text().splitlines()

    size = None
    values = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()

        if upper.startswith("TITLE"):
            continue
        if upper.startswith("DOMAIN_MIN") or upper.startswith("DOMAIN_MAX"):
            continue
        if upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue

        parts = line.split()
        if len(parts) == 3:
            try:
                values.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass

    if size is None:
        raise ValueError("Could not find LUT_3D_SIZE in .cube file")

    arr = np.array(values, dtype=np.float32)
    expected = size * size * size
    if arr.shape[0] != expected:
        raise ValueError(f"Invalid LUT length: expected {expected}, got {arr.shape[0]}")

    return arr.reshape(size, size, size, 3)


def apply_cube_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation for 3D LUT.
    lut shape: [N, N, N, 3]
    """
    n = lut.shape[0]
    coords = clamp01(img) * (n - 1)

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)

    x1 = np.clip(x0 + 1, 0, n - 1)
    y1 = np.clip(y0 + 1, 0, n - 1)
    z1 = np.clip(z0 + 1, 0, n - 1)

    xd = (x - x0)[..., None]
    yd = (y - y0)[..., None]
    zd = (z - z0)[..., None]

    c000 = lut[x0, y0, z0]
    c001 = lut[x0, y0, z1]
    c010 = lut[x0, y1, z0]
    c011 = lut[x0, y1, z1]
    c100 = lut[x1, y0, z0]
    c101 = lut[x1, y0, z1]
    c110 = lut[x1, y1, z0]
    c111 = lut[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    out = c0 * (1 - zd) + c1 * zd
    return clamp01(out)


# ----------------------------
# Main pipeline
# ----------------------------

def emulate_film(
    img: np.ndarray,
    preset: str = "portra",
    lut_path: str | None = None,
    grain: float = 0.05,
    halation: float = 0.18,
    bloom: float = 0.14,
    vignette: float = 0.18,
    seed: int | None = None,
) -> np.ndarray:
    matrix, tuning = stock_color_preset(preset)

    # work in linear for nicer light effects
    lin = srgb_to_linear(img)

    # basic stock color skew
    lin = apply_channel_mix(lin, matrix)
    lin = clamp01(lin)

    # tone
    lin = filmic_curve(
        lin,
        contrast=tuning["contrast"],
        fade=tuning["fade"],
    )

    # glow effects
    lin = add_bloom(lin, strength=bloom)
    lin = add_halation(lin, strength=halation)

    # back to display space
    out = linear_to_srgb(clamp01(lin))
    out = clamp01(out)

    # saturation tuning in display space
    out = apply_saturation(out, tuning["saturation"])

    # optional LUT on top
    if lut_path:
        lut = load_cube_lut(lut_path)
        out = apply_cube_lut(out, lut)

    # lens / texture finishing
    out = add_vignette(out, strength=vignette)
    out = add_film_grain(out, amount=grain, size=1.35, color=0.18, seed=seed)

    return clamp01(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Starter film emulator")
    parser.add_argument("--input", help="Input image path")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument(
        "--preset",
        default="portra",
        choices=["neutral", "portra", "velvia", "cinestill"],
        help="Film preset"
    )
    parser.add_argument("--lut", default=None, help="Optional .cube LUT path")
    parser.add_argument("--grain", type=float, default=0.05, help="Grain strength")
    parser.add_argument("--halation", type=float, default=0.18, help="Halation strength")
    parser.add_argument("--bloom", type=float, default=0.14, help="Bloom strength")
    parser.add_argument("--vignette", type=float, default=0.18, help="Vignette strength")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable grain")
    return parser.parse_args()


def main_film() -> None:
    args = parse_args()

    img = read_image(args.input)
    out = emulate_film(
        img,
        preset=args.preset,
        lut_path=args.lut,
        grain=args.grain,
        halation=args.halation,
        bloom=args.bloom,
        vignette=args.vignette,
        seed=args.seed,
    )
    write_image(args.output, out)
    print(f"Saved film-emulated image to: {args.output}")
