#!/usr/bin/env python3
"""Create Qwen-Image-Edit metadata from curated reverse pseudo-pairs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
HUMAN_SUFFIX = re.compile(r"^(?P<base>.+)_human_hands_v\d+$")
DEFAULT_PROMPT = (
    "Make minimal changes to the image and preserve the camera pose, background, "
    "objects, and contact geometry. Replace only visible human arms, hands, and "
    "body with IRON humanoid robot parts. Keep pose, scale, and position unchanged."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--human-dir", type=Path, default=Path("human_hand_output"))
    parser.add_argument("--robot-dir", type=Path, default=Path("robot_hand_frames"))
    parser.add_argument(
        "--selected-file",
        type=Path,
        help="Optional curated list of human image filenames/stems. Without it, use all.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-size-check", action="store_true")
    return parser.parse_args()


def under_root(path: Path, root: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    return resolved.expanduser().resolve()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def selected_human_images(human_dir: Path, selected_file: Path | None) -> list[Path]:
    by_name = {path.name: path for path in image_files(human_dir)}
    by_stem = {path.stem: path for path in by_name.values()}
    if selected_file is None:
        return list(by_name.values())

    selected = []
    for raw_line in selected_file.read_text(encoding="utf-8").splitlines():
        value = raw_line.split("#", 1)[0].strip()
        if not value:
            continue
        name = Path(value).name
        path = by_name.get(name) or by_stem.get(Path(name).stem)
        if path is None:
            path = human_dir / name
        selected.append(path)
    return selected


def robot_for_human(human_path: Path, robot_by_stem: dict[str, Path]) -> Path | None:
    match = HUMAN_SUFFIX.match(human_path.stem)
    base = match.group("base") if match else human_path.stem
    return robot_by_stem.get(base)


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside dataset root: {path}") from exc


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    human_dir = under_root(args.human_dir, root)
    robot_dir = under_root(args.robot_dir, root)
    selected_file = (
        under_root(args.selected_file, root) if args.selected_file is not None else None
    )
    if not human_dir.is_dir():
        raise FileNotFoundError(f"human image directory not found: {human_dir}")
    if not robot_dir.is_dir():
        raise FileNotFoundError(f"robot image directory not found: {robot_dir}")
    if selected_file is not None and not selected_file.is_file():
        raise FileNotFoundError(f"selected file not found: {selected_file}")

    robot_by_stem = {path.stem: path for path in image_files(robot_dir)}
    records = []
    errors = []
    for human_path in selected_human_images(human_dir, selected_file):
        robot_path = robot_for_human(human_path, robot_by_stem)
        if not human_path.is_file():
            errors.append(f"missing human image: {human_path}")
            continue
        if robot_path is None:
            errors.append(f"no robot target matches: {human_path.name}")
            continue
        if not args.no_size_check:
            with Image.open(human_path) as human, Image.open(robot_path) as robot:
                if human.size != robot.size:
                    errors.append(
                        f"size mismatch {human_path.name} {human.size} != "
                        f"{robot_path.name} {robot.size}"
                    )
                    continue
        records.append(
            {
                "image": relative_posix(robot_path, root),
                "edit_image": [relative_posix(human_path, root)],
                "prompt": args.prompt,
            }
        )

    if errors and not args.skip_missing:
        preview = "\n".join(errors[:20])
        raise RuntimeError(
            f"metadata validation failed with {len(errors)} error(s):\n{preview}"
        )
    if not records:
        raise RuntimeError("no valid pseudo-pairs were found")

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} pairs to {output}")
    if errors:
        print(f"Skipped {len(errors)} invalid entries")


if __name__ == "__main__":
    main()
