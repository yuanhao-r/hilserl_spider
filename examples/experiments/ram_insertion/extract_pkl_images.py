#!/usr/bin/env python3
"""Extract image arrays from transition pickle shards.

This script intentionally writes PNGs without Pillow/OpenCV so it can run in a
minimal environment.
"""

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
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {image.shape}")

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[2] == 4:
        color_type = 6
    else:
        color_type = 2

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


def iter_images(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], np.ndarray]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_images(child, prefix + (str(key),))
        return

    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            yield prefix, value
        elif value.ndim == 4 and value.shape[-1] in (1, 3, 4):
            for index, image in enumerate(value):
                yield prefix + (f"{index:02d}",), image


def extract_pickle(pkl_path: Path, overwrite: bool) -> tuple[int, int]:
    out_root = pkl_path.with_suffix("")
    out_root.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("rb") as f:
        transitions = pickle.load(f)

    written = 0
    skipped = 0
    for transition_index, transition in enumerate(transitions):
        for parts, image in iter_images(transition):
            filename = f"{transition_index:06d}.png"
            out_path = out_root.joinpath(*parts, filename)
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            write_png_rgb(out_path, image)
            written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "buffer_dir",
        nargs="?",
        default="hilserl_spider/examples/experiments/ram_insertion/first_run_260511/buffer",
        help="Directory containing transitions_*.pkl files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing PNG files.")
    args = parser.parse_args()

    buffer_dir = Path(args.buffer_dir)
    total_written = 0
    total_skipped = 0
    for pkl_path in sorted(buffer_dir.glob("*.pkl")):
        written, skipped = extract_pickle(pkl_path, overwrite=args.overwrite)
        total_written += written
        total_skipped += skipped
        print(f"{pkl_path.name}: wrote {written} images, skipped {skipped}")

    print(f"done: wrote {total_written} images, skipped {total_skipped}")


if __name__ == "__main__":
    main()
