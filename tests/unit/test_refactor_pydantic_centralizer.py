"""Pydantic centralizer behavior tests."""

from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.pydantic_centralizer import (
    FlextInfraRefactorPydanticCentralizer,
)


def test_centralizer_converts_typed_dict_factory_to_model(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "pkg" / "service.py"
    src_file.parent.mkdir(parents=True)
    _ = (src_file.parent / "__init__.py").write_text("", encoding="utf-8")
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
    assert summary["parse_syntax_errors"] == 0
    assert summary["parse_encoding_errors"] == 0
    assert summary["parse_io_errors"] == 0
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


def test_centralizer_moves_manual_type_aliases_to_models_file(tmp_path: Path) -> None:
    src_file = tmp_path / "src" / "pkg" / "service.py"
    src_file.parent.mkdir(parents=True)
    _ = (src_file.parent / "__init__.py").write_text("", encoding="utf-8")
    src_file.write_text(
        "from __future__ import annotations\n"
        "from typing import TypeAlias\n\n"
        "PayloadMap: TypeAlias = dict[str, str]\n"
        "ConfigSchema: TypeAlias = dict[str, int]\n\n"
        "def run(data: PayloadMap) -> int:\n"
        "    return len(data)\n",
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

    assert summary["moved_aliases"] >= 2
    assert summary["created_typings_files"] == 0
    assert "PayloadMap: TypeAlias = dict[str, str]" not in updated_source
    assert "ConfigSchema: TypeAlias = dict[str, int]" not in updated_source
    assert "from ._models import ConfigSchema, PayloadMap" in updated_source
    assert "class PayloadMap(RootModel[dict[str, str]]):" in generated_models
    assert "class ConfigSchema(RootModel[dict[str, int]]):" in generated_models


def test_centralizer_moves_dict_alias_in_typings_without_keyword_name(
    tmp_path: Path,
) -> None:
    typings_file = tmp_path / "src" / "pkg" / "typings.py"
    typings_file.parent.mkdir(parents=True)
    _ = (typings_file.parent / "__init__.py").write_text("", encoding="utf-8")
    typings_file.write_text(
        "from __future__ import annotations\n"
        "from collections.abc import Mapping\n"
        "from typing import TypeAlias\n\n"
        "ScalarMap: TypeAlias = Mapping[str, str]\n",
        encoding="utf-8",
    )

    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        tmp_path,
        apply_changes=True,
        normalize_remaining=False,
    )

    updated_typings = typings_file.read_text(encoding="utf-8")
    generated_models = (typings_file.parent / "_models.py").read_text(encoding="utf-8")

    assert summary["moved_aliases"] == 1
    assert "ScalarMap: TypeAlias" not in updated_typings
    assert "from ._models import ScalarMap" in updated_typings
    assert "class ScalarMap(RootModel[Mapping[str, str]]):" in generated_models
