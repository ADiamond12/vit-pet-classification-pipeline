from pathlib import Path

import pytest

from src.inference import bootstrap


def create_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")


def test_ensure_model_dir_prefers_local_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "vit_catsdogs"
    create_checkpoint(model_dir)

    resolved = bootstrap.ensure_model_dir(model_dir=model_dir)

    assert resolved == str(model_dir)


def test_ensure_model_dir_downloads_when_repo_id_is_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = tmp_path / "cache"
    calls: list[tuple[str, str]] = []

    def fake_snapshot_download(repo_id: str, local_dir: str):
        calls.append((repo_id, local_dir))
        create_checkpoint(Path(local_dir))
        return local_dir

    monkeypatch.setattr(bootstrap, "_snapshot_download", fake_snapshot_download)

    resolved = bootstrap.ensure_model_dir(
        model_dir=tmp_path / "missing-model",
        repo_id="example-org/vit-pet-demo",
        cache_dir=cache_root,
    )

    expected_dir = cache_root / "example-org--vit-pet-demo"
    assert resolved == str(expected_dir)
    assert calls == [("example-org/vit-pet-demo", str(expected_dir))]


def test_ensure_model_dir_fails_without_local_or_repo_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        bootstrap.ensure_model_dir(model_dir=tmp_path / "missing-model")

    assert "VIT_PET_MODEL_REPO_ID" in str(exc_info.value)
