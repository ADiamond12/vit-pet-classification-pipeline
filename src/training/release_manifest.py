"""Create a checksum manifest for intentionally released model artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.training.metrics import sha256_file


DEFAULT_ALLOWED_SUFFIXES = {
    ".json",
    ".safetensors",
    ".bin",
    ".txt",
    ".md",
}


def build_release_manifest(
    artifact_dir: str | Path,
    output_path: str | Path,
    release_name: str,
    allowed_suffixes: set[str] | None = None,
    include_training_checkpoints: bool = False,
) -> dict[str, object]:
    source_dir = Path(artifact_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"artifact directory does not exist: {source_dir}")

    suffixes = allowed_suffixes or DEFAULT_ALLOWED_SUFFIXES
    files = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_dir)
        if not include_training_checkpoints and any(part.startswith("checkpoint-") for part in relative_path.parts):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        files.append(
            {
                "path": str(relative_path).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release_name": release_name,
        "artifact_dir": str(source_dir),
        "files": files,
        "artifact_policy": "Upload this manifest and model files as a GitHub Release asset; do not commit weights to git.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a checksum manifest for model release artifacts")
    parser.add_argument("--artifact-dir", default="models/vit_catsdogs")
    parser.add_argument("--output", default="outputs/release/model-release-manifest.json")
    parser.add_argument("--release-name", default="vit-catsdogs-checkpoint")
    parser.add_argument("--include-training-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = build_release_manifest(
        artifact_dir=args.artifact_dir,
        output_path=args.output,
        release_name=args.release_name,
        include_training_checkpoints=args.include_training_checkpoints,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
