"""Project metadata model contract tests.

Behavioral tests over the current public project-metadata SSOT surface:
name derivation (``u.derive_class_stem``), the immutable/validated
``m.ProjectMetadata`` model and its nested PEP 621 ``m.Project`` /
``[tool.flext]`` ``m.ProjectToolFlext`` contracts. No private attributes,
no mocking of the unit under test — only the observable public contract
callers depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.constants import c
from tests.models import m
from tests.utilities import u


class TestsFlextModelsProjectMetadata:
    # ------------------------------------------------------------------
    # derive_class_stem — pure name-derivation contract (behavior on u.*)
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
        self, project_name: str, expected_stem: str
    ) -> None:
        assert u.derive_class_stem(project_name) == expected_stem

    def test_derive_class_stem_returns_empty_for_empty_project_name(self) -> None:
        assert u.derive_class_stem("") == ""

    @pytest.mark.parametrize(
        ("slug", "expected"),
        [
            ("data-pipeline-native", "DataPipelineNative"),
            ("flext_ldif", "FlextLdif"),
            ("single", "Single"),
            ("mixed-case_slug", "MixedCaseSlug"),
        ],
    )
    def test_derive_class_stem_normalizes_kebab_and_snake(
        self, slug: str, expected: str
    ) -> None:
        assert u.derive_class_stem(slug) == expected

    # ------------------------------------------------------------------
    # ProjectMetadata — derived names, immutability, validation
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata(**overrides: object) -> p.ProjectMetadata:
        payload: dict[str, object] = {
            "root": Path("/tmp/flext-ldif"),
            "package_name": "flext_ldif",
            "class_stem": c.Tests.SAMPLE_PROJECT_CLASS_STEM,
            "project": {
                "name": c.Tests.SAMPLE_PROJECT_NAME,
                "version": c.Tests.SAMPLE_PROJECT_VERSION,
            },
            "flext": {},
        }
        payload.update(overrides)
        return m.ProjectMetadata.model_validate(payload)

    def test_project_metadata_exposes_package_name_and_class_stem(self) -> None:
        metadata = self._metadata()

        assert metadata.package_name == "flext_ldif"
        assert metadata.class_stem == c.Tests.SAMPLE_PROJECT_CLASS_STEM

    def test_project_metadata_exposes_nested_project(self) -> None:
        metadata = self._metadata()

        assert metadata.project.name == c.Tests.SAMPLE_PROJECT_NAME
        assert metadata.project.version == c.Tests.SAMPLE_PROJECT_VERSION
        assert metadata.project.authors == ()
        assert metadata.project.requires_python == ""

    def test_project_metadata_is_frozen_against_mutation(self) -> None:
        metadata = self._metadata()

        with pytest.raises(c.ValidationError):
            metadata.package_name = "other"

    def test_project_metadata_rejects_unknown_fields(self) -> None:
        with pytest.raises(c.ValidationError):
            m.ProjectMetadata.model_validate({
                "root": Path("/tmp/flext-ldif"),
                "package_name": "flext_ldif",
                "class_stem": c.Tests.SAMPLE_PROJECT_CLASS_STEM,
                "project": {
                    "name": c.Tests.SAMPLE_PROJECT_NAME,
                    "version": c.Tests.SAMPLE_PROJECT_VERSION,
                },
                "flext": {},
                "unexpected": "boom",
            })

    def test_project_metadata_model_dump_roundtrips(self) -> None:
        metadata = self._metadata(
            project={
                "name": c.Tests.SAMPLE_PROJECT_NAME,
                "version": c.Tests.SAMPLE_PROJECT_VERSION,
                "authors": [{"name": c.Tests.SAMPLE_AUTHOR_ALICE}],
            }
        )

        rebuilt = m.ProjectMetadata.model_validate(metadata.model_dump())

        assert rebuilt == metadata
        assert rebuilt.class_stem == metadata.class_stem

    # ------------------------------------------------------------------
    # Project — PEP 621 normalization
    # ------------------------------------------------------------------
    def test_project_normalizes_pep621_fields(self) -> None:
        project = m.Project.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "authors": [
                {"name": c.Tests.SAMPLE_AUTHOR_ALICE, "email": "alice@example.com"},
                {"name": c.Tests.SAMPLE_AUTHOR_BOB},
            ],
            "urls": {"Homepage": "https://example.test/flext-ldif"},
            "requires-python": ">=3.13,<3.14",
        })

        assert project.authors == (
            m.ProjectAuthor(
                name=c.Tests.SAMPLE_AUTHOR_ALICE, email="alice@example.com"
            ),
            m.ProjectAuthor(name=c.Tests.SAMPLE_AUTHOR_BOB),
        )
        assert project.urls.homepage == "https://example.test/flext-ldif"
        assert project.requires_python == ">=3.13,<3.14"

    def test_project_applies_defaults_for_missing_optionals(self) -> None:
        project = m.Project.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
        })

        assert project.authors == ()
        assert project.requires_python == ""
        assert project.urls.homepage == ""
        assert project.dependencies == ()

    def test_project_ignores_unknown_keys(self) -> None:
        project = m.Project.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "readme": "README.md",
        })

        assert project.name == c.Tests.SAMPLE_PROJECT_NAME
        assert not hasattr(project, "readme")

    # ------------------------------------------------------------------
    # ProjectToolFlext — [tool.flext] aggregation
    # ------------------------------------------------------------------
    def test_project_tool_flext_defaults_and_nested_overrides(self) -> None:
        tool_config = m.ProjectToolFlext.model_validate({
            "project": {"class_stem_override": c.Tests.SAMPLE_PROJECT_CLASS_STEM},
            "docs": {"site_title": "Flext LDIF"},
            "workspace": {"attached": True},
        })

        assert (
            tool_config.project.class_stem_override == c.Tests.SAMPLE_PROJECT_CLASS_STEM
        )
        assert tool_config.docs.site_title == "Flext LDIF"
        assert tool_config.workspace.attached is True

    def test_project_tool_flext_uses_sub_table_defaults_when_empty(self) -> None:
        tool_config = m.ProjectToolFlext()

        assert tool_config.project.class_stem_override is None
        assert tool_config.workspace.attached is False
