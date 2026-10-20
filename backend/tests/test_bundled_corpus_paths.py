"""Bundled corpora must resolve whether the deploy root is the repo or only backend/."""

import asyncio
import importlib.util
import logging
import shutil
import sys
from pathlib import Path

import pytest

from app import config as live_config
from app.config import Settings
from app.guide_knowledge import CuratedGuideSource
from app.main import core_knowledge_provider, create_app

APP_DIR = Path(live_config.__file__).resolve().parent


def _load_config_from(app_dir: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, app_dir / "config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _copy_package(dest: Path) -> Path:
    app_dir = dest / "app"
    app_dir.mkdir(parents=True)
    shutil.copy(APP_DIR / "config.py", app_dir / "config.py")
    shutil.copytree(APP_DIR / "data", app_dir / "data")
    return app_dir


@pytest.mark.parametrize(
    "layout",
    [
        "repo/backend/app",  # local development: the repository root is the project root
        "app",  # Railway with root directory /backend: the package sits at the deploy root
    ],
)
def test_default_corpus_paths_follow_the_package_not_the_project_root(
    tmp_path: Path, layout: str
) -> None:
    app_dir = _copy_package(tmp_path / layout.removesuffix("app").rstrip("/"))
    module = _load_config_from(app_dir, f"config_{layout.replace('/', '_')}")
    settings = module.Settings(_env_file=None)

    for configured in (settings.core_knowledge_file, settings.guide_corpus_file):
        resolved = settings.resolve_local_path(configured)
        assert resolved.is_file(), resolved
        assert app_dir / "data" in resolved.parents


def test_railway_layout_root_is_not_the_repository(tmp_path: Path) -> None:
    app_dir = _copy_package(tmp_path)
    module = _load_config_from(app_dir, "config_railway_root")

    # The old default "backend/app/data/..." resolved against ROOT_DIR and missed the file.
    assert not (module.ROOT_DIR / "backend/app/data/core_knowledge.json").exists()
    assert (
        module.Settings(_env_file=None)
        .resolve_local_path(module.Settings(_env_file=None).core_knowledge_file)
        .is_file()
    )


def test_explicit_overrides_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_KNOWLEDGE_FILE", "custom/core.json")
    monkeypatch.setenv("GUIDE_CORPUS_FILE", str(tmp_path / "guides.json"))
    settings = Settings(_env_file=None)

    assert settings.resolve_local_path(settings.core_knowledge_file) == (
        live_config.ROOT_DIR / "custom/core.json"
    )
    assert settings.resolve_local_path(settings.guide_corpus_file) == tmp_path / "guides.json"


def test_unrelated_paths_still_resolve_under_the_project_root() -> None:
    settings = Settings(_env_file=None)

    assert settings.resolve_local_path(settings.manifest_cache_dir) == (
        live_config.ROOT_DIR / ".cache/manifest"
    )


def test_default_core_corpus_initializes_with_entities(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.main"):
        provider = core_knowledge_provider(Settings(_env_file=None))

    assert provider is not None
    assert provider.engine.entity_count > 0
    assert any("Core knowledge ready" in record.getMessage() for record in caplog.records)


def test_default_guide_corpus_is_readable() -> None:
    settings = Settings(_env_file=None)
    source = CuratedGuideSource(settings.resolve_local_path(settings.guide_corpus_file))

    assert asyncio.run(source.records())
    assert source.version != "unknown"


def test_http_client_loggers_stay_below_info_so_request_urls_are_not_logged() -> None:
    create_app(Settings(_env_file=None, log_level="INFO"))

    for name in ("httpx", "httpcore"):
        assert not logging.getLogger(name).isEnabledFor(logging.INFO)
