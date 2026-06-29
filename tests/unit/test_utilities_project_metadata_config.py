"""Project metadata flext config utility tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_tests import tm

from tests import c, u
from tests.unit._project_metadata_support import write_pyproject


class TestsFlextUtilitiesProjectMetadataConfig:
    def test_read_tool_flext_config_returns_defaults_when_section_absent(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"
            """,
        )
        cfg = u.read_tool_flext_config(root)
        tm.that(cfg.project.project_class, eq="library")
        tm.that(cfg.namespace.enabled, eq=True)

    def test_read_tool_flext_config_applies_user_overrides(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"

            [tool.flext.project]
            project_class = "{c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM}"

            [tool.flext.namespace]
            alias_parent_sources = {{c = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        cfg = u.read_tool_flext_config(root)
        tm.that(cfg.project.project_class, eq=c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM)
        tm.that(
            cfg.namespace.alias_parent_sources["c"],
            eq=c.Tests.SAMPLE_ALIAS_PARENT_SOURCE,
        )

    def test_compose_namespace_config_merges_universal_parent_sources(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"

            [tool.flext.namespace]
            alias_parent_sources = {{c = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        ns = u.compose_namespace_config(root)
        dynamic_constants = u.read_project_constants(c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(ns.project_name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(ns.alias_parent_sources["c"], eq=c.Tests.SAMPLE_ALIAS_PARENT_SOURCE)
        tm.that(
            ns.alias_parent_sources["d"],
            eq=dynamic_constants.UNIVERSAL_ALIAS_PARENT_SOURCES["d"],
        )

    def test_compose_namespace_config_rejects_unknown_alias(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"

            [tool.flext.namespace]
            alias_parent_sources = {{z = "{c.Tests.SAMPLE_ALIAS_PARENT_SOURCE}"}}
            """,
        )
        with pytest.raises(ValueError, match="unknown alias"):
            u.compose_namespace_config(root)

    def test_compose_namespace_config_rejects_universal_alias_override(
        self,
        tmp_path: Path,
    ) -> None:
        root = write_pyproject(
            tmp_path,
            f"""
            [project]
            name = "{c.Tests.SAMPLE_PROJECT_NAME}"
            version = "{c.Tests.SAMPLE_PROJECT_VERSION}"
            license = "{c.Tests.SAMPLE_PROJECT_LICENSE}"

            [tool.flext.namespace]
            alias_parent_sources = {{d = "custom_driver"}}
            """,
        )
        with pytest.raises(ValueError, match="cannot override universal alias"):
            u.compose_namespace_config(root)
