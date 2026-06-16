#!/usr/bin/env python3
"""Preview image augmentations from replay/demo transition pickle files."""

from __future__ import annotations

import argparse
import pickle
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError(f"unsupported image shape {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    image = np.ascontiguousarray(image)
    color_type = 6 if image.shape[2] == 4 else 2
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    while image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got {image.shape}")
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def iter_image_arrays(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_image_arrays(child, prefix + (str(key),))
        return
    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            yield "/".join(prefix), value
        elif value.ndim == 4 and value.shape[-1] in (1, 3, 4):
            for idx, image in enumerate(value):
                yield "/".join(prefix + (f"{idx:02d}",)), image


def find_transition_images(transition: Any, image_keys: set[str] | None) -> Iterable[tuple[str, np.ndarray]]:
    for name, image in iter_image_arrays(transition):
        parts = name.split("/")
        if image_keys is not None and not any(part in image_keys for part in parts):
            continue
        yield name, image


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    x = np.clip(image.astype(np.float32) / 255.0, 0, 1)
    return np.clip((x ** gamma) * 255.0, 0, 255).astype(np.uint8)


def rgb_to_hsv(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = image.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)
    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    hue = hue / 6.0

    sat = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    sat[nonzero] = delta[nonzero] / maxc[nonzero]
    return hue, sat, maxc


def hsv_to_rgb(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    h = (hue % 1.0) * 6.0
    c = val * sat
    x = c * (1.0 - np.abs((h % 2.0) - 1.0))
    m = val - c
    z = np.zeros_like(h)

    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    masks = [
        (0 <= h) & (h < 1),
        (1 <= h) & (h < 2),
        (2 <= h) & (h < 3),
        (3 <= h) & (h < 4),
        (4 <= h) & (h < 5),
        (5 <= h) & (h < 6),
    ]
    vals = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)]
    for mask, channels in zip(masks, vals):
        rgb[..., 0][mask] = channels[0][mask]
        rgb[..., 1][mask] = channels[1][mask]
        rgb[..., 2][mask] = channels[2][mask]
    rgb = (rgb + m[..., None]) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    h, s, v = rgb_to_hsv(image)
    return hsv_to_rgb(h, np.clip(s * factor, 0, 1), v)


def shift_hue(image: np.ndarray, delta: float) -> np.ndarray:
    h, s, v = rgb_to_hsv(image)
    return hsv_to_rgb(h + delta, s, v)


def random_crop_pad(image: np.ndarray, rng: np.random.Generator, padding: int = 4) -> np.ndarray:
    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    y = int(rng.integers(0, 2 * padding + 1))
    x = int(rng.integers(0, 2 * padding + 1))
    h, w = image.shape[:2]
    return padded[y : y + h, x : x + w]


def gaussian_noise(image: np.ndarray, rng: np.random.Generator, sigma: float = 12.0) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def box_blur(image: np.ndarray, radius: int = 1) -> np.ndarray:
    pad = int(radius)
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    count = 0
    for dy in range(2 * pad + 1):
        for dx in range(2 * pad + 1):
            out += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
            count += 1
    return np.clip(out / count, 0, 255).astype(np.uint8)


def cutout(image: np.ndarray, rng: np.random.Generator, frac: float = 0.25) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    out[y : y + ch, x : x + cw] = np.array([127, 127, 127], dtype=np.uint8)
    return out


def make_augmentations(image: np.ndarray, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    return [
        ("original", image),
        ("bright_plus_40", adjust_brightness(image, 40)),
        ("bright_minus_40", adjust_brightness(image, -40)),
        ("contrast_0p70", adjust_contrast(image, 0.70)),
        ("contrast_1p40", adjust_contrast(image, 1.40)),
        ("gamma_0p70", adjust_gamma(image, 0.70)),
        ("gamma_1p50", adjust_gamma(image, 1.50)),
        ("saturation_0p45", adjust_saturation(image, 0.45)),
        ("saturation_1p80", adjust_saturation(image, 1.80)),
        ("hue_shift_18deg", shift_hue(image, 18.0 / 360.0)),
        ("rlpd_random_crop_pad4", random_crop_pad(image, rng, padding=4)),
        ("gaussian_noise_sigma12", gaussian_noise(image, rng, sigma=12.0)),
        ("box_blur_radius1", box_blur(image, radius=1)),
        ("cutout_25pct", cutout(image, rng, frac=0.25)),
    ]


def make_contact_sheet(items: list[tuple[str, np.ndarray]], columns: int = 4, gap: int = 4) -> np.ndarray:
    images = [img for _, img in items]
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = int(np.ceil(len(images) / columns))
    sheet = np.full(
        (rows * h + (rows - 1) * gap, columns * w + (columns - 1) * gap, 3),
        245,
        dtype=np.uint8,
    )
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        y = row * (h + gap)
        x = col * (w + gap)
        tile = np.full((h, w, 3), 245, dtype=np.uint8)
        tile[: image.shape[0], : image.shape[1]] = image
        sheet[y : y + h, x : x + w] = tile
    return sheet


def iter_pickle_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.pkl"))


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay_path",
        help="A transitions_*.pkl file or a directory containing transition pickle files.",
    )
    parser.add_argument("--out_dir", default="", help="Default: <replay_path>/augmentation_preview")
    parser.add_argument("--max_images", type=int, default=12)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image_keys",
        nargs="*",
        default=None,
        help="Optional camera/image keys to include, e.g. wrist_1 wrist_2.",
    )
    parser.add_argument("--write_variants", action="store_true", help="Also write each variant as a PNG.")
    args = parser.parse_args()

    replay_path = Path(args.replay_path).expanduser()
    out_dir = Path(args.out_dir) if args.out_dir else (
        replay_path.with_suffix("") if replay_path.is_file() else replay_path
    ) / "augmentation_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_keys = set(args.image_keys) if args.image_keys else None
    rng = np.random.default_rng(args.seed)
    written = 0

    for pkl_path in iter_pickle_paths(replay_path):
        with pkl_path.open("rb") as f:
            transitions = pickle.load(f)
        for transition_idx in range(0, len(transitions), max(1, args.stride)):
            transition = transitions[transition_idx]
            for image_name, image in find_transition_images(transition, image_keys):
                image = to_uint8_rgb(image)
                variants = make_augmentations(image, rng)
                stem = f"{pkl_path.stem}_t{transition_idx:06d}_{safe_name(image_name)}"
                write_png_rgb(out_dir / f"{stem}_sheet.png", make_contact_sheet(variants))
                if args.write_variants:
                    variant_dir = out_dir / stem
                    for variant_name, variant in variants:
                        write_png_rgb(variant_dir / f"{variant_name}.png", variant)
                written += 1
                print(f"wrote {stem}_sheet.png")
                if written >= args.max_images:
                    print(f"done: wrote {written} contact sheets to {out_dir}")
                    return

    print(f"done: wrote {written} contact sheets to {out_dir}")


if __name__ == "__main__":
    main()
