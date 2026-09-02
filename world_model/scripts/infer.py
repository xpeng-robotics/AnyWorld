#!/usr/bin/env python3
"""Run AnyWorld combined action-camera-embodiment inference."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REQUIRED_FIELDS = (
    "caption",
    "image_path",
    "control_video_path",
    "state_camera_extrinsic_path",
    "camera_fx",
    "camera_fy",
    "camera_cx",
    "camera_cy",
    "camera_orig_width",
    "camera_orig_height",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--anyworld-model-path", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Override the model's sampling length for every sample.",
    )
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and manifest fields without loading the model.",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("validation manifest must contain a non-empty data list")
    return records


def validate_records(records: list[dict]) -> None:
    seen = set()
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if record.get(field) is None]
        if missing:
            raise ValueError(f"record {index} is missing: {', '.join(missing)}")
        episode = str(record.get("episode") or record.get("id") or "")
        if not episode:
            raise ValueError(f"record {index} needs an episode or id")
        if episode in seen:
            raise ValueError(f"duplicate episode/id: {episode}")
        seen.add(episode)
        if record.get("spatial_preprocess") != "center_crop_resize":
            raise ValueError(
                f"record {index} must set spatial_preprocess=center_crop_resize"
            )
        for field in ("image_path", "control_video_path", "state_camera_extrinsic_path"):
            if not Path(record[field]).expanduser().is_file():
                raise FileNotFoundError(f"record {index} {field} not found: {record[field]}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sample"


def main() -> None:
    args = parse_args()
    args.base_model_path = args.base_model_path.expanduser().resolve()
    args.anyworld_model_path = args.anyworld_model_path.expanduser().resolve()
    args.validation_file = args.validation_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    transformer_weights = args.anyworld_model_path / "transformer"
    if not args.base_model_path.is_dir():
        raise FileNotFoundError(f"base model directory not found: {args.base_model_path}")
    if not transformer_weights.is_dir():
        raise FileNotFoundError(
            f"world-model transformer weights directory not found: {transformer_weights}"
        )
    if not args.validation_file.is_file():
        raise FileNotFoundError(f"validation manifest not found: {args.validation_file}")
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank/world-size must satisfy 0 <= rank < world-size")

    records = read_records(args.validation_file)
    validate_records(records)
    shard = records[args.rank::args.world_size]
    print(
        f"Validated {len(records)} samples; rank {args.rank}/{args.world_size} "
        f"will process {len(shard)}"
    )
    if args.check_only:
        return

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from fastvideo import SamplingParam, VideoGenerator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "videos"
    metadata_dir = args.output_dir / "metadata"
    logs_dir = args.output_dir / "logs"
    for directory in (videos_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    generator = VideoGenerator.from_pretrained(
        str(args.base_model_path),
        workload_type="i2v_combined_control",
        init_weights_from_safetensors=str(transformer_weights),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
    )
    results = []
    try:
        for local_index, record in enumerate(shard):
            episode = str(record.get("episode") or record.get("id"))
            filename = safe_name(episode)
            video_path = videos_dir / f"{filename}.mp4"
            metadata_path = metadata_dir / f"{filename}.json"
            if video_path.exists() and metadata_path.exists():
                continue

            sampling = SamplingParam.from_pretrained(str(args.base_model_path))
            for key, value in record.items():
                if hasattr(sampling, key):
                    setattr(sampling, key, value)
            sampling.__post_init__()
            sampling.prompt = record["caption"]
            sampling.height = args.height
            sampling.width = args.width
            if args.num_frames is not None:
                sampling.num_frames = args.num_frames
            sampling.num_inference_steps = args.num_inference_steps
            sampling.guidance_scale = args.guidance_scale
            sampling.fps = args.fps
            sampling.seed = args.seed + args.rank + local_index * args.world_size
            sampling.output_path = str(video_path)
            sampling.save_video = True

            generator.generate_video(
                prompt=record["caption"],
                sampling_param=sampling,
            )
            result = {
                "episode": episode,
                "caption": record["caption"],
                "image_path": record["image_path"],
                "control_video_path": record["control_video_path"],
                "state_camera_extrinsic_path": record["state_camera_extrinsic_path"],
                "output_video_path": str(video_path),
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": sampling.seed,
                "spatial_preprocess": record["spatial_preprocess"],
            }
            metadata_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results.append(result)
    finally:
        generator.shutdown()

    summary = logs_dir / f"rank{args.rank:02d}_results.json"
    summary.write_text(
        json.dumps({"data": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Rank {args.rank} wrote {len(results)} videos; summary: {summary}")


if __name__ == "__main__":
    main()
