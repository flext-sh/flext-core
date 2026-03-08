"""Pydantic centralizer behavior tests."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.pydantic_centralizer import (
    FlextInfraRefactorPydanticCentralizer,
)


def test_centralizer_converts_typed_dict_factory_to_model(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "pkg" / "service.py"
    src_file.parent.mkdir(parents=True)
    src_file.write_text(
        "from __future__ import annotations\n"
        "from typing import TypedDict\n\n"
        "Payload = TypedDict('Payload', {'name': str, 'active': bool}, total=False)\n\n"
        "def run(data: Payload) -> str:\n"
        "    return data['name']\n",
        encoding="utf-8",
    )

    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        tmp_path,
        apply_changes=True,
        normalize_remaining=False,
    )

    models_file = src_file.parent / "_models.py"
    updated_source = src_file.read_text(encoding="utf-8")
    generated_models = models_file.read_text(encoding="utf-8")

    assert summary["moved_classes"] == 1
    assert "Payload = TypedDict(" not in updated_source
    assert "from ._models import Payload" in updated_source
    assert "class Payload(BaseModel):" in generated_models
    assert "name: str | None = Field(default=None)" in generated_models
    assert "active: bool | None = Field(default=None)" in generated_models


def test_centralizer_does_not_touch_settings_module(tmp_path: Path) -> None:
    settings_file = tmp_path / "src" / "pkg" / "settings.py"
    settings_file.parent.mkdir(parents=True)
    original_source = (
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n\n"
        "class LocalSettings(BaseModel):\n"
        "    value: str\n"
    )
    _ = settings_file.write_text(original_source, encoding="utf-8")

    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        tmp_path,
        apply_changes=True,
        normalize_remaining=True,
    )

    assert summary["scanned_files"] == 0
    assert settings_file.read_text(encoding="utf-8") == original_source
