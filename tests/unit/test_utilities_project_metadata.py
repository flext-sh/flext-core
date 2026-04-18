"""Tests for FlextUtilitiesProjectMetadata — Tier 4 SSOT utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from flext_core._models.project_metadata import (
    FlextModelsProjectMetadata as pm,
)
from flext_core._utilities.project_metadata import (
    FlextUtilitiesProjectMetadata as up,
)


def _write_pyproject(tmp_path: Path, body: str) -> Path:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(dedent(body).lstrip(), encoding="utf-8")
    return tmp_path


class TestDeriveClassStem:
    @pytest.mark.parametrize(
        ("project_name", "expected"),
        [
            ("flext-ldif", "FlextLdif"),
            ("flext-meltano", "FlextMeltano"),
            ("flext-tap-oracle", "FlextTapOracle"),
            ("flext-core", "Flext"),
            ("flext", "FlextRoot"),
            ("gruponos-meltano-native", "GruponosMeltanoNative"),
        ],
    )
    def test_derives_expected_stem(self, project_name: str, expected: str) -> None:
        assert up.derive_class_stem(project_name) == expected

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            up.derive_class_stem("")


class TestDerivePackageName:
    @pytest.mark.parametrize(
        ("project_name", "expected"),
        [
            ("flext-ldif", "flext_ldif"),
            ("flext-core", "flext_core"),
            ("gruponos-meltano-native", "gruponos_meltano_native"),
        ],
    )
    def test_derives_expected_package(self, project_name: str, expected: str) -> None:
        assert up.derive_package_name(project_name) == expected

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            up.derive_package_name("")


class TestDeriveTierFacadeName:
    @pytest.mark.parametrize(
        ("project_name", "tier", "expected"),
        [
            ("flext-ldif", "src", "FlextLdif"),
            ("flext-ldif", "tests", "TestsFlextLdif"),
            ("flext-ldif", "examples", "ExamplesFlextLdif"),
            ("flext-ldif", "scripts", "ScriptsFlextLdif"),
            ("flext-ldif", "docs", "DocsFlextLdif"),
            ("flext-core", "src", "Flext"),
            ("flext-core", "tests", "TestsFlext"),
        ],
    )
    def test_derives_facade(
        self, project_name: str, tier: str, expected: str
    ) -> None:
        assert up.derive_tier_facade_name(project_name, tier) == expected

    def test_unknown_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown tier"):
            up.derive_tier_facade_name("flext-ldif", "nonsense")


class TestReadProjectMetadata:
    def test_minimal(self, tmp_path: Path) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = "MIT"
            description = "LDIF"
            """,
        )
        meta = up.read_project_metadata(root)
        assert isinstance(meta, pm.Project)
        assert meta.name == "flext-ldif"
        assert meta.class_stem == "FlextLdif"

    def test_missing_pyproject_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            up.read_project_metadata(tmp_path)

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        root = _write_pyproject(tmp_path, '[project]\nversion="0.12.0"\n')
        with pytest.raises(ValueError, match=r"missing.*name"):
            up.read_project_metadata(root)

    def test_missing_version_raises(self, tmp_path: Path) -> None:
        root = _write_pyproject(tmp_path, '[project]\nname="x"\n')
        with pytest.raises(ValueError, match=r"missing.*version"):
            up.read_project_metadata(root)

    def test_spdx_license_dict(self, tmp_path: Path) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = {text = "MIT"}
            """,
        )
        meta = up.read_project_metadata(root)
        assert meta.license == "MIT"


class TestReadToolFlextConfig:
    def test_defaults_when_absent(self, tmp_path: Path) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = "MIT"
            """,
        )
        cfg = up.read_tool_flext_config(root)
        assert cfg.project.project_class == "library"
        assert cfg.namespace.enabled is True

    def test_custom_overrides(self, tmp_path: Path) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = "MIT"

            [tool.flext.project]
            project_class = "platform"

            [tool.flext.namespace]
            alias_parent_sources = {c = "flext_cli"}
            """,
        )
        cfg = up.read_tool_flext_config(root)
        assert cfg.project.project_class == "platform"
        assert cfg.namespace.alias_parent_sources["c"] == "flext_cli"


class TestComposeNamespaceConfig:
    def test_composes_project_name_and_merges_universals(self, tmp_path: Path) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = "MIT"

            [tool.flext.namespace]
            alias_parent_sources = {c = "flext_cli"}
            """,
        )
        ns = up.compose_namespace_config(root)
        assert ns.project_name == "flext-ldif"
        assert ns.alias_parent_sources["c"] == "flext_cli"
        assert ns.alias_parent_sources["r"] == "flext_core"
