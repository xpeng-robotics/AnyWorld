#!/usr/bin/env python3
"""Apply synchronous crop, flip, and brightness augmentation to image pairs."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-subdir", type=Path, default=Path("augmented"),
        help="Output image directory relative to dataset root.",
    )
    parser.add_argument("--num-crops", type=int, default=2)
    parser.add_argument("--crop-ratio-min", type=float, default=0.6)
    parser.add_argument("--crop-ratio-max", type=float, default=0.9)
    parser.add_argument("--num-brightness", type=int, default=2)
    parser.add_argument("--brightness-min", type=float, default=0.9)
    parser.add_argument("--brightness-max", type=float, default=1.5)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--exclude-fisheye", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_under_root(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {path}") from exc
    return candidate


def random_crop(
    rng: random.Random,
    width: int,
    height: int,
    ratio_min: float,
    ratio_max: float,
) -> tuple[int, int, int, int]:
    ratio = rng.uniform(ratio_min, ratio_max)
    crop_width = max(1, round(width * ratio))
    crop_height = max(1, round(height * ratio))
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    return left, top, left + crop_width, top + crop_height


def transform(
    image: Image.Image,
    crop: tuple[int, int, int, int] | None,
    flip: bool,
    brightness: float,
) -> Image.Image:
    output = image
    if crop is not None:
        output = output.crop(crop).resize(image.size, Image.Resampling.LANCZOS)
    if flip:
        output = output.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if brightness != 1.0:
        output = ImageEnhance.Brightness(output).enhance(brightness)
    return output


def save_image(image: Image.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95)


def main() -> None:
    args = parse_args()
    if args.num_crops < 0 or args.num_brightness < 0:
        raise ValueError("augmentation counts cannot be negative")
    if not 0 < args.crop_ratio_min <= args.crop_ratio_max <= 1:
        raise ValueError("crop ratios must satisfy 0 < min <= max <= 1")
    if not 0 < args.brightness_min <= args.brightness_max:
        raise ValueError("brightness factors must satisfy 0 < min <= max")

    root = args.dataset_root.expanduser().resolve()
    input_json = resolve_under_root(args.input_json, root)
    output_json = resolve_under_root(args.output_json, root)
    output_root = resolve_under_root(args.output_subdir, root)
    records = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("input metadata must be a non-empty JSON list")

    rng = random.Random(args.seed)
    augmented = []
    for sample_index, record in enumerate(tqdm(records, desc="augment-pairs")):
        target_path = resolve_under_root(Path(record["image"]), root)
        source_values = record.get("edit_image")
        if not isinstance(source_values, list) or not source_values:
            raise ValueError(f"record {sample_index} has no edit_image list")

        if args.exclude_fisheye and any(
            "fish_eye" in value for value in [record["image"], *source_values]
        ):
            continue

        with Image.open(target_path) as opened_target:
            target = opened_target.convert("RGB")
        for source_value in source_values:
            source_path = resolve_under_root(Path(source_value), root)
            with Image.open(source_path) as opened_source:
                source = opened_source.convert("RGB")
            if source.size != target.size:
                raise ValueError(
                    f"pair size mismatch: {source_path} {source.size} != "
                    f"{target_path} {target.size}"
                )

            width, height = target.size
            crops = [None] + [
                random_crop(
                    rng, width, height, args.crop_ratio_min, args.crop_ratio_max
                )
                for _ in range(args.num_crops)
            ]
            brightnesses = [1.0] + [
                rng.uniform(args.brightness_min, args.brightness_max)
                for _ in range(args.num_brightness)
            ]
            flips = [False] if args.no_flip else [False, True]

            for flip, (crop_index, crop), (brightness_index, brightness) in itertools.product(
                flips, enumerate(crops), enumerate(brightnesses)
            ):
                tag = (
                    f"s{sample_index:06d}_f{int(flip)}_c{crop_index}_b{brightness_index}"
                )
                target_name = f"{target_path.stem}_{tag}.jpg"
                source_name = f"{source_path.stem}_{tag}.jpg"
                augmented_target = output_root / "robot" / target_name
                augmented_source = output_root / "human" / source_name

                save_image(
                    transform(target, crop, flip, brightness),
                    augmented_target,
                    args.overwrite,
                )
                save_image(
                    transform(source, crop, flip, brightness),
                    augmented_source,
                    args.overwrite,
                )
                augmented.append(
                    {
                        "image": augmented_target.relative_to(root).as_posix(),
                        "edit_image": [
                            augmented_source.relative_to(root).as_posix()
                        ],
                        "prompt": record["prompt"],
                    }
                )

    if not augmented:
        raise RuntimeError("no augmented pairs were generated")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(augmented, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    factor = len(augmented) / len(records)
    print(f"Wrote {len(augmented)} pairs ({factor:.1f}x) to {output_json}")


if __name__ == "__main__":
    main()
