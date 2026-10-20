"""Project metadata model contract tests.

Behavioral tests over the current public project-metadata SSOT surface:
name derivation (``u.derive_class_stem``), the immutable/validated
``m.ProjectMetadata`` model and its nested PEP 621 ``m.Project`` /
``[tool.flext]`` ``m.ProjectToolFlext`` contracts. No private attributes,
no mocking of the unit under test — only the observable public contract
callers depend on.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import c, m, p, t, tm, u


class TestsFlextModelsProjectMetadata:
    # ------------------------------------------------------------------
    # derive_class_stem — pure name-derivation contract (behavior on u.*)
    # ------------------------------------------------------------------
    """Tests for ``FlextModelsProjectMetadata``."""

    @staticmethod
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
        project_name: str,
        expected_stem: str,
    ) -> None:
        """Test derive class stem covers overrides and pascalization."""
        tm.that(u.derive_class_stem(project_name), eq=expected_stem)

    @staticmethod
    def test_derive_class_stem_returns_empty_for_empty_project_name() -> None:
        """Test derive class stem returns empty for empty project name."""
        tm.that(u.derive_class_stem(""), eq="")

    @staticmethod
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
        slug: str,
        expected: str,
    ) -> None:
        """Test derive class stem normalizes kebab and snake."""
        tm.that(u.derive_class_stem(slug), eq=expected)

    # ------------------------------------------------------------------
    # ProjectMetadata — derived names, immutability, validation
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata(
        tmp_path: Path,
        **overrides: p.AttributeProbe,
    ) -> m.ProjectMetadata:
        payload: t.MutableMappingKV[str, p.AttributeProbe] = {
            "root": tmp_path,
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

    def test_project_metadata_exposes_package_name_and_class_stem(
        self,
        tmp_path: Path,
    ) -> None:
        """Test project metadata exposes package name and class stem."""
        metadata = self._metadata(tmp_path)

        tm.that(metadata.package_name, eq="flext_ldif")
        tm.that(metadata.class_stem, eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM)

    def test_project_metadata_exposes_nested_project(self, tmp_path: Path) -> None:
        """Test project metadata exposes nested project."""
        metadata = self._metadata(tmp_path)

        tm.that(metadata.project.name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(metadata.project.version, eq=c.Tests.SAMPLE_PROJECT_VERSION)
        tm.that(metadata.project.authors, eq=())
        tm.that(metadata.project.requires_python, eq="")

    def test_project_metadata_is_frozen_against_mutation(self, tmp_path: Path) -> None:
        """Test project metadata is frozen against mutation."""
        metadata = self._metadata(tmp_path)

        field_name = "package_name"
        with pytest.raises(c.ValidationError):
            setattr(metadata, field_name, "other")

    @staticmethod
    def test_project_metadata_rejects_unknown_fields(tmp_path: Path) -> None:
        """Test project metadata rejects unknown fields."""
        with pytest.raises(c.ValidationError):
            m.ProjectMetadata.model_validate({
                "root": tmp_path,
                "package_name": "flext_ldif",
                "class_stem": c.Tests.SAMPLE_PROJECT_CLASS_STEM,
                "project": {
                    "name": c.Tests.SAMPLE_PROJECT_NAME,
                    "version": c.Tests.SAMPLE_PROJECT_VERSION,
                },
                "flext": {},
                "unexpected": "boom",
            })

    def test_project_metadata_model_dump_roundtrips(self, tmp_path: Path) -> None:
        """Test project metadata model dump roundtrips."""
        metadata = self._metadata(
            tmp_path,
            project={
                "name": c.Tests.SAMPLE_PROJECT_NAME,
                "version": c.Tests.SAMPLE_PROJECT_VERSION,
                "authors": [{"name": c.Tests.SAMPLE_AUTHOR_ALICE}],
            },
        )

        rebuilt = m.ProjectMetadata.model_validate(metadata.model_dump())

        tm.that(rebuilt, eq=metadata)
        tm.that(rebuilt.class_stem, eq=metadata.class_stem)

    # ------------------------------------------------------------------
    # Project — PEP 621 normalization
    # ------------------------------------------------------------------
    @staticmethod
    def test_project_normalizes_pep621_fields() -> None:
        """Test project normalizes pep621 fields."""
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

        tm.that(
            project.authors,
            eq=(
                m.ProjectAuthor(
                    name=c.Tests.SAMPLE_AUTHOR_ALICE, email="alice@example.com"
                ),
                m.ProjectAuthor(name=c.Tests.SAMPLE_AUTHOR_BOB),
            ),
        )
        tm.that(project.urls.homepage, eq="https://example.test/flext-ldif")
        tm.that(project.requires_python, eq=">=3.13,<3.14")

    @staticmethod
    def test_project_applies_defaults_for_missing_optionals() -> None:
        """Test project applies defaults for missing optionals."""
        project = m.Project.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
        })

        tm.that(project.authors, eq=())
        tm.that(project.requires_python, eq="")
        tm.that(project.urls.homepage, eq="")
        tm.that(project.dependencies, eq=())

    @staticmethod
    def test_project_ignores_unknown_keys() -> None:
        """Test project ignores unknown keys."""
        project = m.Project.model_validate({
            "name": c.Tests.SAMPLE_PROJECT_NAME,
            "version": c.Tests.SAMPLE_PROJECT_VERSION,
            "readme": "README.md",
        })

        tm.that(project.name, eq=c.Tests.SAMPLE_PROJECT_NAME)
        tm.that(hasattr(project, "readme"), truth=False)

    # ------------------------------------------------------------------
    # ProjectToolFlext — [tool.flext] aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def test_project_tool_flext_defaults_and_nested_overrides() -> None:
        """Test project tool flext defaults and nested overrides."""
        tool_config = m.ProjectToolFlext.model_validate({
            "project": {"class_stem_override": c.Tests.SAMPLE_PROJECT_CLASS_STEM},
            "docs": {"site_title": "Flext LDIF"},
            "workspace": {"attached": True},
        })

        tm.that(
            tool_config.project.class_stem_override,
            eq=c.Tests.SAMPLE_PROJECT_CLASS_STEM,
        )
        tm.that(tool_config.docs.site_title, eq="Flext LDIF")
        tm.that(tool_config.workspace.attached is True, truth=True)

    @staticmethod
    def test_project_tool_flext_uses_sub_table_defaults_when_empty() -> None:
        """Test project tool flext uses sub table defaults when empty."""
        tool_config = m.ProjectToolFlext()

        tm.that(tool_config.project.class_stem_override, none=True)
        tm.that(tool_config.workspace.attached is False, truth=True)

    @staticmethod
    def test_project_tool_flext_budget_table_round_trips() -> None:
        """The budget-gate projection emitted by flext-infra ingress-validates."""
        budget = {"lint": {"time-seconds": 30, "memory-mb": 1024, "tokens": 100000}}
        tool_config = m.ProjectToolFlext.model_validate({"project": {"budget": budget}})

        tm.that(tool_config.project.budget, eq=budget)

    @staticmethod
    def test_project_tool_flext_budget_defaults_to_none() -> None:
        """Test project tool flext budget defaults to none."""
        tool_config = m.ProjectToolFlext()

        tm.that(tool_config.project.budget, none=True)
