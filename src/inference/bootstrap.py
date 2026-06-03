"""Helpers for resolving the inference checkpoint location."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_REPO_ID_ENV = "VIT_PET_MODEL_REPO_ID"
MODEL_CACHE_DIR_ENV = "VIT_PET_MODEL_CACHE_DIR"
DEFAULT_MODEL_DIR = Path("models") / "vit_catsdogs"
DEFAULT_CACHE_DIR = Path(".cache") / "vit-pet-models"

_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
)


def _has_complete_checkpoint(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any((path / name).is_file() for name in _PROCESSOR_FILES)
        and any((path / name).is_file() for name in _WEIGHT_FILES)
    )


def _cache_target(cache_root: Path, repo_id: str) -> Path:
    return cache_root / repo_id.replace("/", "--")


def _snapshot_download(repo_id: str, local_dir: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface-hub is required for model bootstrap. Install the project dependencies or provide a local checkpoint."
        ) from exc
    return snapshot_download(repo_id=repo_id, local_dir=local_dir)


def ensure_model_dir(
    model_dir: str | os.PathLike[str] | None = None,
    repo_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Return a local checkpoint path, downloading one when configured."""
    local_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    if _has_complete_checkpoint(local_dir):
        return str(local_dir)

    effective_repo_id = repo_id or os.getenv(MODEL_REPO_ID_ENV)
    if not effective_repo_id:
        raise FileNotFoundError(
            f"No local model checkpoint found at '{local_dir}'. "
            f"Train one locally or set {MODEL_REPO_ID_ENV} to a published fine-tuned checkpoint."
        )

    cache_root = Path(cache_dir or os.getenv(MODEL_CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
    target_dir = _cache_target(cache_root, effective_repo_id)
    if _has_complete_checkpoint(target_dir):
        return str(target_dir)

    downloaded_path = Path(_snapshot_download(repo_id=effective_repo_id, local_dir=str(target_dir)))
    resolved_path = downloaded_path if _has_complete_checkpoint(downloaded_path) else target_dir
    if _has_complete_checkpoint(resolved_path):
        return str(resolved_path)

    raise FileNotFoundError(
        f"Downloaded checkpoint for '{effective_repo_id}' is incomplete. "
        f"Expected config, processor, and weight files under '{resolved_path}'."
    )


def checkpoint_status(
    model_dir: str | os.PathLike[str] | None = None,
    repo_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str | bool]:
    """Report checkpoint readiness without downloading or loading model weights."""
    local_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    effective_repo_id = repo_id or os.getenv(MODEL_REPO_ID_ENV)
    cache_root = Path(cache_dir or os.getenv(MODEL_CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
    cache_dir_for_repo = _cache_target(cache_root, effective_repo_id) if effective_repo_id else None
    local_available = _has_complete_checkpoint(local_dir)
    cache_available = bool(cache_dir_for_repo and _has_complete_checkpoint(cache_dir_for_repo))

    if local_available:
        source = "local"
    elif cache_available:
        source = "bootstrap-cache"
    elif effective_repo_id:
        source = "configured-bootstrap-missing-cache"
    else:
        source = "missing"

    return {
        "model_dir": str(local_dir),
        "local_checkpoint_available": local_available,
        "bootstrap_repo_id_configured": bool(effective_repo_id),
        "bootstrap_cache_available": cache_available,
        "ready_for_prediction": local_available or cache_available,
        "artifact_source": source,
        "artifact_policy": "No dataset, checkpoint, or cache files are committed to the public repo.",
    }
