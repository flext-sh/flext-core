"""Behavioral contract tests for the project-metadata read/derive utilities.

Exercises the OBSERVABLE public surface of ``FlextUtilitiesProjectMetadata``
that the split sibling modules (``_read`` / ``_config`` /
``test_project_metadata_facade_access``) do not already cover:

- ``u.load_pyproject_toml`` — raw ``pyproject.toml`` parsing and its
  missing-file error path.
- ``u.derive_project_constants`` — deriving a populated ``ProjectConstants``
  from a real package root, and its error path on an incomplete root.
- ``u.read_lazy_alias_metadata`` — installed-distribution lazy-alias table
  and its unknown-distribution error path.
- ``m.LazyAliasMetadata`` — public field/serialization contract.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

import flext_core
from tests.constants import c
from tests.models import m
from tests.unit._project_metadata_support import write_pyproject
from tests.utilities import u


class TestsFlextCoreUtilitiesProjectMetadata:
    """Public-contract behavior of the project-metadata read/derive helpers."""

    @pytest.fixture
    def package_root(self) -> Path:
        """Resolve the installed flext-core package root (contains pyproject + src)."""
        return Path(flext_core.__file__).resolve().parent.parent.parent

    def test_load_pyproject_toml_parses_declared_project_table(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"
            description = "LDIF lib"
            """,
        )

        # Act
        raw = u.load_pyproject_toml(root)

        # Assert — parsed mapping exposes the declared project table.
        project_table = raw["project"]
        assert isinstance(project_table, dict)
        assert project_table["name"] == c.Tests.SAMPLE_PROJECT_NAME
        assert project_table["version"] == c.Tests.SAMPLE_PROJECT_VERSION

    def test_load_pyproject_toml_raises_when_file_absent(
        self,
        tmp_path: Path,
    ) -> None:
        # Act / Assert — missing pyproject.toml is a hard, observable failure.
        with pytest.raises(FileNotFoundError, match=r"pyproject\.toml"):
            u.load_pyproject_toml(tmp_path)

    def test_derive_project_constants_from_real_package_root(
        self,
        package_root: Path,
    ) -> None:
        # Act
        constants = u.derive_project_constants(package_root)

        # Assert — public fields consumers rely on, derived from the root.
        assert constants.PACKAGE_NAME == "flext-core"
        assert constants.PYTHON_PACKAGE_NAME == "flext_core"
        assert constants.CLASS_STEM == "Flext"
        assert "src" in constants.SCAN_DIRECTORIES

    def test_derive_project_constants_raises_on_incomplete_root(
        self,
        tmp_path: Path,
    ) -> None:
        # Act / Assert — a directory without pyproject.toml cannot yield constants.
        with pytest.raises(FileNotFoundError):
            u.derive_project_constants(tmp_path)

    def test_read_lazy_alias_metadata_returns_populated_alias_table(self) -> None:
        # Act
        entries = u.read_lazy_alias_metadata("flext-core")

        # Assert — every runtime alias is represented with populated fields.
        aliases = {entry.alias for entry in entries}
        assert {"c", "m", "p", "t", "u", "r"} <= aliases
        for entry in entries:
            assert entry.alias
            assert entry.module_path.startswith("flext_core")
            assert entry.suffix

    def test_read_lazy_alias_metadata_entry_module_path_matches_alias(self) -> None:
        # Act
        entries = u.read_lazy_alias_metadata("flext-core")
        by_alias = {entry.alias: entry for entry in entries}

        # Assert — the constants alias maps to the constants module and suffix.
        constants_entry = by_alias["c"]
        assert constants_entry.module_path == "flext_core.constants"
        assert constants_entry.suffix == "Constants"

    def test_read_lazy_alias_metadata_raises_for_unknown_distribution(self) -> None:
        # Act / Assert — an uninstalled package name is a hard failure, not empty.
        with pytest.raises(RuntimeError, match="installed distribution"):
            u.read_lazy_alias_metadata("does-not-exist-xyz")

    def test_lazy_alias_metadata_model_exposes_declared_fields(self) -> None:
        # Arrange / Act
        entry = m.LazyAliasMetadata(
            alias="c",
            module_path="flext_core.constants",
            parent_source="flext_core",
            suffix="Constants",
            facade=True,
        )

        # Assert — public field state.
        assert entry.alias == "c"
        assert entry.module_path == "flext_core.constants"
        assert entry.parent_source == "flext_core"
        assert entry.suffix == "Constants"
        assert entry.facade is True

    def test_lazy_alias_metadata_model_dump_is_public_serialization(self) -> None:
        # Arrange
        entry = m.LazyAliasMetadata(
            alias="m",
            module_path="flext_core.models",
            parent_source="flext_core",
            suffix="Models",
            facade=False,
        )

        # Act
        dumped = entry.model_dump()

        # Assert — model_dump reflects the constructed public state.
        assert dumped["alias"] == "m"
        assert dumped["suffix"] == "Models"
        assert dumped["facade"] is False
