#!/usr/bin/env python3
"""Fail if a release tree contains likely credentials, private paths, or weights."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {
    "", ".cfg", ".cff", ".csv", ".ini", ".json", ".jsonl", ".md",
    ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
}
MODEL_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
MAX_TEXT_BYTES = 5 * 1024 * 1024
MAX_RELEASE_BYTES = 50 * 1024 * 1024
SELF = Path(__file__).resolve()

# Assemble a few signatures in pieces so this scanner does not flag its own
# source. The file is skipped as a second safeguard.
PRIVATE_DOMAIN = "xiao" + "peng.com"
PATTERNS = (
    ("private dataset path", re.compile(r"/dataset[_-]rc[_-]mm(?:/|$)", re.I)),
    ("private workspace path", re.compile(r"/workspace/[^\s]+@" + re.escape(PRIVATE_DOMAIN), re.I)),
    ("private object-store URI", re.compile(r"\boss" + r"://", re.I)),
    ("credential in URL", re.compile(r"https?://[^\s/:]+:[^\s/@]+@", re.I)),
    ("GitLab access token", re.compile("gl" + r"pat-[A-Za-z0-9_-]{12,}")),
    ("AWS access key", re.compile("AK" + r"IA[0-9A-Z]{16}")),
    ("Google API key", re.compile("AI" + r"za[0-9A-Za-z_-]{30,}")),
    (
        "hard-coded secret",
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*"
            r"['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    return parser.parse_args()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == SELF:
            continue
        if ".git" in path.relative_to(root).parts and path.name != "config":
            continue
        yield path


def main() -> int:
    root = parse_args().root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    findings: list[tuple[Path, str, int | None]] = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in MODEL_SUFFIXES:
            findings.append((rel, "model/weight artifact", None))
        if size > MAX_RELEASE_BYTES:
            findings.append((rel, f"file larger than {MAX_RELEASE_BYTES // 1024 // 1024} MiB", None))
        if suffix not in TEXT_SUFFIXES or size > MAX_TEXT_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS:
                if not pattern.search(line):
                    continue
                if label == "hard-coded secret" and any(
                    placeholder in line.lower()
                    for placeholder in ("your-key", "example", "changeme", "replace-me")
                ):
                    continue
                findings.append((rel, label, line_number))

    if findings:
        print(f"Release audit failed with {len(findings)} finding(s):")
        for path, label, line_number in findings:
            location = f"{path}:{line_number}" if line_number else str(path)
            print(f"  {location}: {label}")
        print("Potential secret values are intentionally not echoed.")
        return 1

    print(f"Release audit passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
