#!/usr/bin/env python3
"""Generate samples from serl_launcher.vision.data_augmentations.color_transform."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


def _inject_paths() -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    paths = [
        workspace_root / "hilserl_spider" / "serl_launcher",
        workspace_root / "hilserl_spider" / "examples",
    ]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_inject_paths()

from experiments.ram_insertion.preview_replaybuffer_augmentations import (  # noqa: E402
    find_transition_images,
    iter_pickle_paths,
    make_contact_sheet,
    safe_name,
    to_uint8_rgb,
    write_png_rgb,
)


def _load_color_transform():
    workspace_root = Path(__file__).resolve().parents[4]
    module_path = (
        workspace_root
        / "hilserl_spider"
        / "serl_launcher"
        / "serl_launcher"
        / "vision"
        / "data_augmentations.py"
    )
    spec = importlib.util.spec_from_file_location("serl_data_augmentations", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.color_transform


color_transform = _load_color_transform()


@partial(
    jax.jit,
    static_argnames=(
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "to_grayscale_prob",
        "color_jitter_prob",
        "apply_prob",
        "shuffle",
    ),
)
def jitted_color_transform(
    image,
    rng,
    *,
    brightness,
    contrast,
    saturation,
    hue,
    to_grayscale_prob,
    color_jitter_prob,
    apply_prob,
    shuffle,
):
    return color_transform(
        image,
        rng,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        to_grayscale_prob=to_grayscale_prob,
        color_jitter_prob=color_jitter_prob,
        apply_prob=apply_prob,
        shuffle=shuffle,
    )


def _to_float01(image: np.ndarray) -> jnp.ndarray:
    image = to_uint8_rgb(image)
    return jnp.asarray(image, dtype=jnp.float32) / 255.0


def _to_uint8(image: jnp.ndarray) -> np.ndarray:
    arr = np.asarray(jax.device_get(image))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def apply_color_transform(image: np.ndarray, seed: int, **kwargs) -> np.ndarray:
    out = jitted_color_transform(_to_float01(image), jax.random.PRNGKey(seed), **kwargs)
    return _to_uint8(out)


def color_transform_samples(image: np.ndarray, seed: int) -> list[tuple[str, np.ndarray]]:
    base = dict(
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        to_grayscale_prob=0.0,
        color_jitter_prob=1.0,
        apply_prob=1.0,
        shuffle=False,
    )
    configs = [
        ("original", None),
        ("brightness_0p35", dict(base, brightness=0.35)),
        ("contrast_0p35", dict(base, contrast=0.35)),
        ("saturation_0p50", dict(base, saturation=0.50)),
        ("hue_0p10", dict(base, hue=0.10)),
        (
            "jitter_bchs_no_shuffle",
            dict(base, brightness=0.30, contrast=0.30, saturation=0.40, hue=0.08),
        ),
        (
            "jitter_bchs_shuffle",
            dict(
                base,
                brightness=0.30,
                contrast=0.30,
                saturation=0.40,
                hue=0.08,
                shuffle=True,
            ),
        ),
        ("grayscale_prob_1", dict(base, to_grayscale_prob=1.0)),
        (
            "jitter_plus_gray",
            dict(
                base,
                brightness=0.25,
                contrast=0.25,
                saturation=0.35,
                hue=0.05,
                to_grayscale_prob=1.0,
                shuffle=True,
            ),
        ),
    ]

    samples = []
    for idx, (name, cfg) in enumerate(configs):
        if cfg is None:
            samples.append((name, to_uint8_rgb(image)))
        else:
            samples.append((name, apply_color_transform(image, seed + idx, **cfg)))

    for idx in range(4):
        cfg = dict(
            base,
            brightness=0.30,
            contrast=0.35,
            saturation=0.40,
            hue=0.05,
            shuffle=True,
        )
        samples.append(
            (
                f"jitter_random_seed_{idx}",
                apply_color_transform(image, seed + 100 + idx, **cfg),
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replay_path",
        help="A transitions_*.pkl file or a directory containing transition pickle files.",
    )
    parser.add_argument("--out_dir", default="", help="Default: <replay_path>/color_transform_preview")
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_keys", nargs="*", default=None)
    parser.add_argument("--write_variants", action="store_true")
    args = parser.parse_args()

    replay_path = Path(args.replay_path).expanduser()
    out_dir = Path(args.out_dir) if args.out_dir else (
        replay_path.with_suffix("") if replay_path.is_file() else replay_path
    ) / "color_transform_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_keys = set(args.image_keys) if args.image_keys else None
    written = 0
    for pkl_path in iter_pickle_paths(replay_path):
        with pkl_path.open("rb") as f:
            transitions = pickle.load(f)
        for transition_idx in range(0, len(transitions), max(1, args.stride)):
            transition = transitions[transition_idx]
            for image_name, image in find_transition_images(transition, image_keys):
                image = to_uint8_rgb(image)
                samples = color_transform_samples(image, args.seed + written * 1000)
                stem = f"{pkl_path.stem}_t{transition_idx:06d}_{safe_name(image_name)}"
                write_png_rgb(out_dir / f"{stem}_color_transform_sheet.png", make_contact_sheet(samples))
                if args.write_variants:
                    variant_dir = out_dir / stem
                    for sample_name, sample_image in samples:
                        write_png_rgb(variant_dir / f"{sample_name}.png", sample_image)
                written += 1
                print(f"wrote {stem}_color_transform_sheet.png")
                if written >= args.max_images:
                    print(f"done: wrote {written} contact sheets to {out_dir}")
                    return
    print(f"done: wrote {written} contact sheets to {out_dir}")


if __name__ == "__main__":
    main()
