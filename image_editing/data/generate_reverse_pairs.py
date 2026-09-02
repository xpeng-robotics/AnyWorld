#!/usr/bin/env python3
"""Build robot-to-human reverse pseudo-pairs with a Gemini image editor.

The generated human-like image is the source and the original robot frame is
the target when training the AnyWorld embodiment editor. API credentials are
read from an environment variable and are never written to the output log.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


PROMPTS = (
    "Edit only the provided image. Preserve the camera pose, background, "
    "objects, object state, and contact geometry. Replace visible robot hands, "
    "arms, and torso with realistic human anatomy. Keep pose, orientation, "
    "scale, and position unchanged. Do not modify anything else.",
    "Edit only the provided image. Preserve the camera pose, background, "
    "objects, object state, and contact geometry. Replace visible robot hands, "
    "arms, and torso with realistic human anatomy wearing subtle motion-capture "
    "gloves and wrist straps. Keep pose, orientation, scale, and position "
    "unchanged. Do not modify anything else.",
    "Edit only the provided image. Preserve the camera pose, background, "
    "objects, object state, and contact geometry. Replace visible robot hands, "
    "arms, and torso with realistic human anatomy. Add one subtle, realistic "
    "accessory such as a watch, bracelet, or ring. Keep pose, orientation, "
    "scale, and position unchanged. Do not modify anything else.",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-like images from robot frames for reverse pseudo-pairs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-csv",
        type=Path,
        help="CSV containing a path or filename column.",
    )
    source.add_argument(
        "--input-images",
        type=Path,
        nargs="+",
        help="Robot images to edit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("ANYWORLD_IMAGE_MODEL", "gemini-2.5-flash-image"),
        help="Gemini image-editing model name.",
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_API_KEY",
        help="Name of the environment variable containing the API key.",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="Optional .env file. Do not commit it to source control.",
    )
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show planned outputs without calling the API.",
    )
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("--variants must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    return args


def load_dotenv_file(path: Path | None) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file not found: {path}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install python-dotenv to use --dotenv") from exc
    load_dotenv(path)


def collect_images(
    input_csv: Path | None, input_images: Iterable[Path] | None
) -> list[Path]:
    if input_images:
        candidates = [Path(path).expanduser() for path in input_images]
    else:
        assert input_csv is not None
        input_csv = input_csv.expanduser()
        if not input_csv.is_file():
            raise FileNotFoundError(f"input CSV not found: {input_csv}")
        candidates = []
        with input_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not fields.intersection({"path", "filename"}):
                raise ValueError("input CSV must contain a path or filename column")
            for row in reader:
                value = (row.get("path") or row.get("filename") or "").strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = input_csv.parent / path
                candidates.append(path)

    unique: dict[str, Path] = {}
    for path in candidates:
        resolved = path.resolve()
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image extension: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"input image not found: {resolved}")
        unique[str(resolved)] = resolved
    if not unique:
        raise ValueError("no input images found")
    return [unique[key] for key in sorted(unique)]


def image_part(types_module, path: Path):
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[path.suffix.lower()]
    return types_module.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def save_response_image(response, target_size: tuple[int, int], output_path: Path) -> None:
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and inline_data.data:
                with Image.open(io.BytesIO(inline_data.data)) as generated:
                    image = generated.convert("RGB")
                    if image.size != target_size:
                        image = image.resize(target_size, Image.Resampling.LANCZOS)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(output_path, quality=95)
                return
    raise RuntimeError("the image-editing response did not contain an image")


def generate_one(client, types_module, model: str, source: Path, prompt: str, output: Path):
    with Image.open(source) as image:
        target_size = image.size
    response = client.models.generate_content(
        model=model,
        contents=[
            "TARGET IMAGE TO EDIT:",
            image_part(types_module, source),
            prompt,
        ],
        config=types_module.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        ),
    )
    save_response_image(response, target_size, output)


def main() -> None:
    args = parse_args()
    load_dotenv_file(args.dotenv)
    images = collect_images(args.input_csv, args.input_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    jobs = []
    for source in images:
        for variant in range(1, args.variants + 1):
            output = args.output_dir / f"{source.stem}_human_hands_v{variant:02d}.jpg"
            jobs.append((source, output, rng.choice(PROMPTS)))

    if args.dry_run:
        print(f"Validated {len(images)} input images; planned outputs: {len(jobs)}")
        for _, output, _ in jobs[:10]:
            print(output)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing API credential in environment variable {args.api_key_env}")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai before generating pseudo-pairs") from exc

    client = genai.Client(api_key=api_key)
    log_path = args.output_dir / "generation_manifest.jsonl"
    successes = 0
    failures = 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        for source, output, prompt in tqdm(jobs, desc="reverse-pairs"):
            status = "generated"
            error = ""
            if output.exists() and not args.overwrite:
                status = "skipped"
            else:
                for attempt in range(1, args.max_retries + 1):
                    try:
                        generate_one(client, types, args.model, source, prompt, output)
                        break
                    except Exception as exc:  # API errors vary by SDK version.
                        error = str(exc)
                        if attempt == args.max_retries:
                            status = "failed"
                        else:
                            time.sleep(args.delay * (2 ** (attempt - 1)))
            if status == "failed":
                failures += 1
            else:
                successes += 1
            log_handle.write(
                json.dumps(
                    {
                        "source_robot_image": str(source),
                        "generated_human_image": str(output),
                        "model": args.model,
                        "prompt": prompt,
                        "status": status,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_handle.flush()
            if args.delay > 0 and status == "generated":
                time.sleep(args.delay)

    print(f"Completed: {successes} successful/skipped, {failures} failed")
    print(f"Manifest: {log_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
