"""Project metadata flext config utility behavioral tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from tests.constants import c
from tests.unit._project_metadata_support import write_pyproject
from tests.utilities import u

if TYPE_CHECKING:
    from pathlib import Path

_BASE_PROJECT = f"""
[project]
name = "{c.Tests.SAMPLE_PROJECT_NAME}"
version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"
"""


class TestsFlextCoreUtilitiesProjectMetadataConfig:
    def test_read_tool_flext_config_returns_defaults_when_section_absent(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(tmp_path, _BASE_PROJECT)
        # Act
        cfg = u.read_tool_flext_config(root)
        # Assert
        tm.that(cfg.project.project_class, eq="library")
        tm.that(cfg.namespace.enabled, eq=True)

    def test_read_tool_flext_config_applies_user_overrides(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(
            tmp_path,
            _BASE_PROJECT
            + f"""
            [tool.flext.project]
            project_class = "{c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM}"

            [tool.flext.namespace]
            alias_parent_sources = {{c = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        # Act
        cfg = u.read_tool_flext_config(root)
        # Assert
        tm.that(cfg.project.project_class, eq=c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM)
        tm.that(
            cfg.namespace.alias_parent_sources["c"],
            eq=c.Tests.SAMPLE_ALIAS_PARENT_SOURCE,
        )

    def test_compose_namespace_config_reports_project_name_and_user_alias(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(
            tmp_path,
            _BASE_PROJECT
            + f"""
            [tool.flext.namespace]
            alias_parent_sources = {{c = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        # Act
        ns = u.compose_namespace_config(root)
        # Assert
        tm.that(ns.project_name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(ns.enabled, eq=True)
        tm.that(ns.alias_parent_sources["c"], eq=c.Tests.SAMPLE_ALIAS_PARENT_SOURCE)

    def test_compose_namespace_config_merges_universal_parent_sources(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(
            tmp_path,
            _BASE_PROJECT
            + f"""
            [tool.flext.namespace]
            alias_parent_sources = {{c = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        constants = u.read_project_constants(c.Tests.SAMPLE_PROJECT_NAME)
        # Act
        ns = u.compose_namespace_config(root)
        # Assert: user alias preserved alongside every universal alias
        tm.that(ns.alias_parent_sources["c"], eq=c.Tests.SAMPLE_ALIAS_PARENT_SOURCE)
        for alias, canonical in constants.UNIVERSAL_ALIAS_PARENT_SOURCES.items():
            tm.that(ns.alias_parent_sources[alias], eq=canonical)

    def test_compose_namespace_config_defaults_scan_dirs_to_constants(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange
        root = write_pyproject(tmp_path, _BASE_PROJECT)
        constants = u.read_project_constants(c.Tests.SAMPLE_PROJECT_NAME)
        # Act
        ns = u.compose_namespace_config(root)
        # Assert
        tm.that(tuple(ns.scan_dirs), eq=tuple(constants.SCAN_DIRECTORIES))

    def test_compose_namespace_config_allows_universal_alias_at_canonical_value(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange: restate a universal alias to its own canonical value (no override)
        constants = u.read_project_constants(c.Tests.SAMPLE_PROJECT_NAME)
        alias, canonical = next(
            iter(constants.UNIVERSAL_ALIAS_PARENT_SOURCES.items()),
        )
        root = write_pyproject(
            tmp_path,
            _BASE_PROJECT
            + f"""
            [tool.flext.namespace]
            alias_parent_sources = {{{alias} = "{canonical}"}}
            """,
        )
        # Act
        ns = u.compose_namespace_config(root)
        # Assert
        tm.that(ns.alias_parent_sources[alias], eq=canonical)

    @pytest.mark.parametrize(
        ("alias_table", "match"),
        [
            (
                f'{{z = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}',
                "unknown alias",
            ),
            (
                '{d = "custom_driver"}',
                "cannot override universal alias",
            ),
        ],
    )
    def test_compose_namespace_config_rejects_invalid_alias_tables(
        self,
        tmp_path: Path,
        alias_table: str,
        match: str,
    ) -> None:
        # Arrange
        root = write_pyproject(
            tmp_path,
            _BASE_PROJECT
            + f"""
            [tool.flext.namespace]
            alias_parent_sources = {alias_table}
            """,
        )
        # Act / Assert
        with pytest.raises(ValueError, match=match):
            u.compose_namespace_config(root)
