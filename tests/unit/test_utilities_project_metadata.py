"""Behavior contract for u.* project-metadata utilities — public API only."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from flext_tests import tm

from tests import m, u


def _write_pyproject(tmp_path: Path, body: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(dedent(body).lstrip(), encoding="utf-8")
    return tmp_path


class TestsFlextCoreUtilitiesProjectMetadata:
    """Behavior contract for u.derive_*, u.pascalize, u.read_*, u.compose_*."""

    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            ("flext-ldif", "FlextLdif"),
            ("flext-tap-oracle", "FlextTapOracle"),
            ("flext-core", "Flext"),
            ("flext", "FlextRoot"),
            ("gruponos-meltano-native", "GruponosMeltanoNative"),
        ],
    )
    def test_derive_class_stem_produces_pascal_case_from_project_name(
        self,
        project_name: str,
        expected_stem: str,
    ) -> None:
        tm.that(u.derive_class_stem(project_name), eq=expected_stem)

    def test_pascalize_converts_dashes_and_underscores_to_camel_case(self) -> None:
        tm.that(u.pascalize("flext-ldif"), eq="FlextLdif")
        tm.that(u.pascalize("flext_ldif"), eq="FlextLdif")
        tm.that(u.pascalize(""), eq="")

    def test_derive_class_stem_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            u.derive_class_stem("")

    def test_read_project_metadata_parses_minimal_pyproject(
        self,
        tmp_path: Path,
    ) -> None:
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
        meta = u.read_project_metadata(root)
        tm.that(meta, is_=m.ProjectMetadata)
        tm.that(meta.name, eq="flext-ldif")
        tm.that(meta.class_stem, eq="FlextLdif")

    def test_read_project_metadata_accepts_spdx_license_dict(
        self,
        tmp_path: Path,
    ) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = {text = "MIT"}
            """,
        )
        tm.that(u.read_project_metadata(root).license, eq="MIT")

    def test_read_project_metadata_raises_on_missing_pyproject(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            u.read_project_metadata(tmp_path)

    @pytest.mark.parametrize(
        ("body", "match_pattern"),
        [
            ('[project]\nversion="0.12.0"\n', r"missing.*name"),
            ('[project]\nname="x"\n', r"missing.*version"),
        ],
        ids=["missing_name", "missing_version"],
    )
    def test_read_project_metadata_rejects_incomplete_pyproject(
        self,
        tmp_path: Path,
        body: str,
        match_pattern: str,
    ) -> None:
        root = _write_pyproject(tmp_path, body)
        with pytest.raises(ValueError, match=match_pattern):
            u.read_project_metadata(root)

    def test_read_tool_flext_config_returns_defaults_when_section_absent(
        self,
        tmp_path: Path,
    ) -> None:
        root = _write_pyproject(
            tmp_path,
            """
            [project]
            name = "flext-ldif"
            version = "0.12.0-dev"
            license = "MIT"
            """,
        )
        cfg = u.read_tool_flext_config(root)
        tm.that(cfg.project.project_class, eq="library")
        tm.that(cfg.namespace.enabled, eq=True)

    def test_read_tool_flext_config_applies_user_overrides(
        self,
        tmp_path: Path,
    ) -> None:
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
        cfg = u.read_tool_flext_config(root)
        tm.that(cfg.project.project_class, eq="platform")
        tm.that(cfg.namespace.alias_parent_sources["c"], eq="flext_cli")

    def test_compose_namespace_config_merges_universal_parent_sources(
        self,
        tmp_path: Path,
    ) -> None:
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
        ns = u.compose_namespace_config(root)
        tm.that(ns.project_name, eq="flext-ldif")
        tm.that(ns.alias_parent_sources["c"], eq="flext_cli")
        tm.that(ns.alias_parent_sources["r"], eq="flext_core")
