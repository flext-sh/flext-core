"""Project metadata model contract tests.

Behavioral tests over the public ``m.*`` project-metadata SSOT surface:
name derivation, immutable/validated model fields, PEP 621 normalization,
and ``[tool.flext]`` aggregation. No private attributes, no mocking of the
unit under test — only the observable public contract callers depend on.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from tests.constants import c
from tests.models import m


class TestsFlextModelsProjectMetadata:
    # ------------------------------------------------------------------
    # derive_class_stem / pascalize — pure name-derivation contract
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            (c.Tests.SAMPLE_PROJECT_NAME, c.Tests.SAMPLE_PROJECT_CLASS_STEM),
            ("flext_core", "FlextCore"),
            (c.Tests.SAMPLE_PROJECT_NAME_MIGRATION, "DemoMigrationTool"),
            ("data-pipeline-native", "DataPipelineNative"),
            ("FLEXT-LDIF", c.Tests.SAMPLE_PROJECT_CLASS_STEM),
        ],
    )
    def test_derive_class_stem_covers_overrides_and_pascalization(
        self,
        project_name: str,
        expected_stem: str,
    ) -> None:
        assert m.derive_class_stem(project_name) == expected_stem

    def test_derive_class_stem_rejects_empty_project_name(self) -> None:
        with pytest.raises(ValueError, match="empty project name"):
            m.derive_class_stem("")

    @pytest.mark.parametrize(
        ("slug", "expected"),
        [
            ("data-pipeline-native", "DataPipelineNative"),
            ("flext_ldif", "FlextLdif"),
            ("single", "Single"),
            ("mixed-case_Slug", "MixedCaseSlug"),
        ],
    )
    def test_pascalize_normalizes_kebab_and_snake(
        self,
        slug: str,
        expected: str,
    ) -> None:
        assert m.pascalize(slug) == expected

    # ------------------------------------------------------------------
    # ProjectMetadata — derived names, defaults, immutability, validation
    # ------------------------------------------------------------------
    def test_project_metadata_exposes_package_name_and_class_stem(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
        )

        assert metadata.package_name == "flext_ldif"
        assert metadata.class_stem == c.Tests.SAMPLE_PROJECT_CLASS_STEM

    def test_project_metadata_optional_fields_default_to_empty(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
        )

        assert metadata.description == ""
        assert metadata.authors == ()
        assert metadata.url == ""
        assert metadata.requires_python == ""

    def test_project_metadata_is_frozen_against_mutation(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
        )

        with pytest.raises(c.ValidationError):
            metadata.version = "9.9.9"

    def test_project_metadata_rejects_unknown_fields(self) -> None:
        with pytest.raises(c.ValidationError):
            m.ProjectMetadata.model_validate({
                "name": c.Tests.SAMPLE_PROJECT_NAME,
                "version": c.Tests.SAMPLE_PROJECT_VERSION,
                "license": c.Tests.SAMPLE_PROJECT_LICENSE,
                "root": Path("/tmp/flext-ldif"),
                "unexpected": "boom",
            })

    def test_project_metadata_rejects_empty_name(self) -> None:
        with pytest.raises(c.ValidationError):
            m.ProjectMetadata.model_validate({
                "name": "",
                "version": c.Tests.SAMPLE_PROJECT_VERSION,
                "license": c.Tests.SAMPLE_PROJECT_LICENSE,
                "root": Path("/tmp/flext-ldif"),
            })

    def test_project_metadata_model_dump_roundtrips(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
            authors=(c.Tests.SAMPLE_AUTHOR_ALICE,),
        )

        rebuilt = m.ProjectMetadata.model_validate(metadata.model_dump())

        assert rebuilt == metadata
        assert rebuilt.class_stem == metadata.class_stem

    # ------------------------------------------------------------------
    # ProjectConstants — canonical field reuse
    # ------------------------------------------------------------------
    def test_project_constants_model_reuses_canonical_fields(self) -> None:
        metadata = m.ProjectMetadata(
            name=c.Tests.SAMPLE_PROJECT_NAME,
            version=c.Tests.SAMPLE_PROJECT_VERSION,
            license=c.Tests.SAMPLE_PROJECT_LICENSE,
            root=Path("/tmp/flext-ldif"),
            authors=(c.Tests.SAMPLE_AUTHOR_ALICE,),
            url="https://example.test/flext-ldif",
        )

        constants = m.ProjectConstants.model_validate({
            "PACKAGE_NAME": metadata.name,
            "PACKAGE_VERSION": metadata.version,
            "PACKAGE_LICENSE": metadata.license,
            "PACKAGE_URL": metadata.url,
            "PACKAGE_AUTHORS": metadata.authors,
            "PACKAGE_ROOT": metadata.root,
            "PYTHON_PACKAGE_NAME": metadata.package_name,
            "CLASS_STEM": metadata.class_stem,
            "ALIAS_TO_SUFFIX": {"c": "Constants"},
            "RUNTIME_ALIAS_NAMES": frozenset({"c"}),
            "FACADE_ALIAS_NAMES": frozenset({"c"}),
            "FACADE_MODULE_NAMES": frozenset({"constants"}),
            "UNIVERSAL_ALIAS_PARENT_SOURCES": {},
            "TIER_FACADE_PREFIX": {"src": metadata.class_stem},
            "SCAN_DIRECTORIES": ("src",),
            "TIER_SUB_NAMESPACE": {"src": ""},
            "PYPROJECT_FILENAME": c.PYPROJECT_FILENAME,
        })

        assert constants.PACKAGE_NAME == c.Tests.SAMPLE_PROJECT_NAME
        assert constants.PACKAGE_VERSION == c.Tests.SAMPLE_PROJECT_VERSION
        assert constants.PACKAGE_LICENSE == c.Tests.SAMPLE_PROJECT_LICENSE
        assert constants.PYTHON_PACKAGE_NAME == "flext_ldif"
        assert constants.CLASS_STEM == c.Tests.SAMPLE_PROJECT_CLASS_STEM
        assert constants.PACKAGE_AUTHORS == (c.Tests.SAMPLE_AUTHOR_ALICE,)

    # ------------------------------------------------------------------
    # PyprojectProject — PEP 621 normalization
    # ------------------------------------------------------------------
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
            "Homepage": "https://example.test/flext-ldif",
        })
        assert project.requires_python == "3.13"

        metadata = project.to_metadata(Path("/tmp/flext-ldif"))
        assert metadata.url == "https://example.test/flext-ldif"
        assert metadata.requires_python == "3.13"

    def test_pyproject_project_applies_defaults_for_missing_optionals(self) -> None:
        project = m.PyprojectProject.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
        })

        assert project.license == "UNLICENSED"
        assert project.authors == ()
        assert dict(project.urls) == {}
        assert project.requires_python == ""

        metadata = project.to_metadata(Path("/tmp/flext-ldif"))
        assert metadata.url == ""
        assert metadata.license == "UNLICENSED"

    def test_pyproject_project_ignores_unknown_keys(self) -> None:
        project = m.PyprojectProject.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "readme": "README.md",
            "dependencies": ["pydantic"],
        })

        assert project.name == c.Tests.SAMPLE_PROJECT_NAME
        assert not hasattr(project, "readme")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (">=3.13,<3.14", "3.13"),
            (">=3.11", "3.11"),
            (">=3.12,<4", "3.12"),
            ("", ""),
        ],
    )
    def test_pyproject_project_normalizes_requires_python(
        self,
        raw: str,
        expected: str,
    ) -> None:
        project = m.PyprojectProject.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "requires-python": raw,
        })

        assert project.requires_python == expected

    # ------------------------------------------------------------------
    # ProjectNamespaceConfig — namespace configuration contract
    # ------------------------------------------------------------------
    def test_project_namespace_config_preserves_explicit_namespace_values(
        self,
    ) -> None:
        namespace = m.ProjectNamespaceConfig.model_validate({
            "project_name": c.Tests.SAMPLE_PROJECT_NAME,
            "scan_dirs": ("src",),
            "alias_parent_sources": {"c": c.Tests.SAMPLE_ALIAS_PARENT_SOURCE},
            "include_dynamic_dirs": True,
        })

        assert namespace.alias_parent_sources["c"] == c.Tests.SAMPLE_ALIAS_PARENT_SOURCE
        assert namespace.scan_dirs == ("src",)
        assert namespace.include_dynamic_dirs is True

    def test_project_namespace_config_applies_defaults(self) -> None:
        namespace = m.ProjectNamespaceConfig.model_validate({
            "project_name": c.Tests.SAMPLE_PROJECT_NAME,
        })

        assert namespace.enabled is True
        assert namespace.scan_dirs == ()
        assert namespace.include_dynamic_dirs is False
        assert dict(namespace.alias_parent_sources) == {}

    # ------------------------------------------------------------------
    # ProjectToolFlext — [tool.flext] aggregation
    # ------------------------------------------------------------------
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
            "c": c.Tests.SAMPLE_ALIAS_PARENT_SOURCE,
        }
        assert tool_config.namespace.include_dynamic_dirs is True
        assert tool_config.docs.site_title == "Flext LDIF"
        assert tool_config.aliases.overrides == {"u": "TestsFlextUtilities"}
        assert tool_config.workspace.attached is True

    def test_project_tool_flext_uses_sub_table_defaults_when_empty(self) -> None:
        tool_config = m.ProjectToolFlext()

        assert tool_config.project.project_class == "library"
        assert tool_config.docs.site_title is None
        assert tool_config.namespace.include_dynamic_dirs is False
        assert tool_config.workspace.attached is False
        assert dict(tool_config.aliases.overrides) == {}
