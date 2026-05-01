"""Project metadata model contract tests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from tests import c, m


class TestsFlextModelsProjectMetadata:
    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            (c.Tests.SAMPLE_PROJECT_NAME, c.Tests.SAMPLE_PROJECT_CLASS_STEM),
            ("flext_core", "Flext"),
            (c.Tests.SAMPLE_PROJECT_NAME_MIGRATION, "AlgarOudMig"),
            ("gruponos-meltano-native", "GruponosMeltanoNative"),
        ],
    )
    def test_derive_class_stem_covers_overrides_and_pascalization(
        self,
        project_name: str,
        expected_stem: str,
    ) -> None:
        assert m.derive_class_stem(project_name) == expected_stem

    def test_project_metadata_exposes_package_name_and_class_stem(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
        )

        assert metadata.package_name == "flext_ldif"
        assert metadata.class_stem == c.Tests.SAMPLE_PROJECT_CLASS_STEM

    def test_project_constants_from_metadata_reuses_canonical_fields(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
            authors=(c.Tests.SAMPLE_AUTHOR_ALICE,),
            url="https://example.test/flext-ldif",
        )

        constants = m.ProjectConstants.from_metadata(metadata)

        assert constants.PACKAGE_NAME == c.Tests.SAMPLE_PROJECT_NAME
        assert constants.PACKAGE_VERSION == c.Tests.SAMPLE_PROJECT_VERSION
        assert constants.PACKAGE_LICENSE == c.Tests.SAMPLE_PROJECT_LICENSE
        assert constants.PYTHON_PACKAGE_NAME == "flext_ldif"
        assert constants.CLASS_STEM == c.Tests.SAMPLE_PROJECT_CLASS_STEM
        assert constants.PACKAGE_AUTHORS == (c.Tests.SAMPLE_AUTHOR_ALICE,)

    def test_pyproject_project_normalizes_pep621_fields(self) -> None:
        project = m.PyprojectProject.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "license": {"text": c.Tests.SAMPLE_PROJECT_LICENSE},
            "authors": [
                {"name": c.Tests.SAMPLE_AUTHOR_ALICE, "email": "alice@example.com"},
                {"name": c.Tests.SAMPLE_AUTHOR_BOB},
            ],
            "urls": {"Homepage": "https://example.test/flext-ldif"},
            "requires-python": ">=3.13,<3.14",
        })

        assert project.license == c.Tests.SAMPLE_PROJECT_LICENSE
        assert project.authors == (
            c.Tests.SAMPLE_AUTHOR_ALICE,
            c.Tests.SAMPLE_AUTHOR_BOB,
        )
        assert project.urls == MappingProxyType({
            "Homepage": "https://example.test/flext-ldif"
        })
        assert project.requires_python == "3.13"

        metadata = project.to_metadata(Path("/tmp/flext-ldif"))
        assert metadata.url == "https://example.test/flext-ldif"
        assert metadata.requires_python == "3.13"

    def test_project_namespace_config_merges_universal_alias_sources(self) -> None:
        namespace = m.ProjectNamespaceConfig.model_validate({
            "project_name": c.Tests.SAMPLE_PROJECT_NAME,
            "alias_parent_sources": {"c": c.Tests.SAMPLE_ALIAS_PARENT_SOURCE},
        })

        assert namespace.alias_parent_sources["c"] == c.Tests.SAMPLE_ALIAS_PARENT_SOURCE
        assert namespace.alias_parent_sources["r"] == "flext_core"
        assert namespace.scan_dirs == c.SCAN_DIRECTORIES

    @pytest.mark.parametrize(
        ("sources", "match"),
        [
            ({"zzz": "custom"}, "unknown alias"),
            ({"r": "custom_runtime"}, "cannot override universal alias"),
        ],
        ids=("unknown-alias", "override-universal-alias"),
    )
    def test_project_namespace_config_rejects_invalid_alias_sources(
        self,
        sources: dict[str, str],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _ = m.ProjectNamespaceConfig.model_validate({
                "project_name": c.Tests.SAMPLE_PROJECT_NAME,
                "alias_parent_sources": sources,
            })

    def test_project_tool_flext_defaults_and_nested_overrides(self) -> None:
        tool_config = m.ProjectToolFlext.model_validate({
            "project": {"project_class": c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM},
            "namespace": {
                "alias_parent_sources": {"c": c.Tests.SAMPLE_ALIAS_PARENT_SOURCE},
                "include_dynamic_dirs": True,
            },
            "docs": {"site_title": "Flext LDIF"},
            "aliases": {"overrides": {"u": "TestsFlextUtilities"}},
            "workspace": {"attached": True},
        })

        assert (
            tool_config.project.project_class == c.Tests.SAMPLE_PROJECT_CLASS_PLATFORM
        )
        assert tool_config.namespace.alias_parent_sources == {
            "c": c.Tests.SAMPLE_ALIAS_PARENT_SOURCE
        }
        assert tool_config.namespace.include_dynamic_dirs is True
        assert tool_config.docs.site_title == "Flext LDIF"
        assert tool_config.aliases.overrides == {"u": "TestsFlextUtilities"}
        assert tool_config.workspace.attached is True
