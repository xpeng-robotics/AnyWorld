#!/usr/bin/env python3
"""Run the fine-tuned Qwen embodiment editor on images or video first frames."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_PROMPT = (
    "Make minimal changes to the image and preserve the camera pose, background, "
    "objects, and contact geometry. Replace only visible human arms, hands, and "
    "body with IRON humanoid robot parts. Keep pose, scale, and position unchanged."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        help="Optional local model directory containing tokenizer/ and processor/.",
    )
    parser.add_argument(
        "--diffsynth-repo",
        type=Path,
        default=Path(os.environ["DIFFSYNTH_REPO"]) if "DIFFSYNTH_REPO" in os.environ else None,
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument(
        "--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1"))
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_inputs(values: list[Path]) -> list[Path]:
    files = []
    for value in values:
        value = value.expanduser().resolve()
        if value.is_dir():
            files.extend(
                path for path in sorted(value.rglob("*"))
                if path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
            )
        elif value.is_file() and value.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            files.append(value)
        else:
            raise FileNotFoundError(f"unsupported or missing input: {value}")
    unique = {str(path.resolve()): path.resolve() for path in files}
    if not unique:
        raise ValueError("no supported image or video input found")
    return [unique[key] for key in sorted(unique)]


def extract_first_frame(video: Path, output: Path, overwrite: bool) -> Path:
    if output.exists() and not overwrite:
        return output
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for video inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video), "-frames:v", "1", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return output


def prepare_images(
    inputs: list[Path], output_dir: Path, overwrite: bool
) -> list[tuple[Path, str]]:
    prepared = []
    frame_dir = output_dir / "_first_frames"
    for path in inputs:
        if path.suffix.lower() in VIDEO_SUFFIXES:
            frame = extract_first_frame(path, frame_dir / f"{path.stem}.png", overwrite)
            prepared.append((frame, path.stem))
        else:
            prepared.append((path, path.stem))
    return prepared


def configure_diffsynth(repo: Path | None) -> None:
    if repo is not None:
        repo = repo.expanduser().resolve()
        if not (repo / "diffsynth").is_dir():
            raise FileNotFoundError(f"DiffSynth-Studio package not found under {repo}")
        sys.path.insert(0, str(repo))


def load_pipeline(args: argparse.Namespace, device: str):
    import torch
    from diffsynth import load_state_dict
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    model_configs = [
        ModelConfig(
            model_id=args.model_id,
            origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id=args.model_id,
            origin_file_pattern="text_encoder/model*.safetensors",
        ),
        ModelConfig(
            model_id=args.model_id,
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
        ),
    ]
    tokenizer_config = None
    processor_config = None
    if args.base_model_path is not None:
        base = args.base_model_path.expanduser().resolve()
        tokenizer_config = ModelConfig(path=str(base / "tokenizer"))
        processor_config = ModelConfig(path=str(base / "processor"))

    pipeline = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        processor_config=processor_config,
    )
    state_dict = load_state_dict(str(args.checkpoint))
    pipeline.dit.load_state_dict(state_dict)
    return pipeline


def main() -> None:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank/world-size must satisfy 0 <= rank < world-size")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"editor checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args.input)
    prepared = prepare_images(inputs, args.output_dir, args.overwrite)
    shard = prepared[args.rank::args.world_size]
    if args.dry_run:
        print(f"Validated {len(prepared)} inputs; rank {args.rank} will process {len(shard)}")
        return

    configure_diffsynth(args.diffsynth_repo)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for embodiment-editor inference") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-Edit inference requires a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.rank)))
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    pipeline = load_pipeline(args, device)

    rows = []
    for index, (image_path, output_stem) in enumerate(shard):
        output_path = args.output_dir / f"{output_stem}_iron.png"
        status = "generated"
        error = ""
        if output_path.exists() and not args.overwrite:
            status = "skipped"
        else:
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                width = max(16, image.width // 16 * 16)
                height = max(16, image.height // 16 * 16)
                if image.size != (width, height):
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                output = pipeline(
                    args.prompt,
                    edit_image=[image],
                    seed=args.seed + index,
                    num_inference_steps=args.steps,
                    height=height,
                    width=width,
                    zero_cond_t=True,
                )
                output.save(output_path)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        rows.append(
            {
                "input": str(image_path),
                "output": str(output_path),
                "checkpoint": str(args.checkpoint),
                "status": status,
                "error": error,
            }
        )

    manifest = args.output_dir / f"manifest_rank{args.rank:02d}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["input", "output", "checkpoint", "status", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row["status"] == "failed" for row in rows)
    print(f"Rank {args.rank}: {len(rows) - failures} succeeded/skipped, {failures} failed")
    print(f"Manifest: {manifest}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
