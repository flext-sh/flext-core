"""Behavioral contract tests for the project-metadata SSOT surface.

Exercises the OBSERVABLE public behavior consumers depend on:

- ``u.derive_class_stem`` name-derivation output, including the special-name
  overrides and the empty-input path.
- ``m.ProjectMetadata`` invariants: immutability (``frozen``),
  ``extra="forbid"``, nested ``project`` PEP 621 data and derived
  ``package_name`` / ``class_stem`` fields.
- ``m.ProjectToolFlext`` public sub-table defaults.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.constants import c
from tests.models import m
from tests.utilities import u


class TestsFlextFacadeFlatSsotAccess:
    """Public-contract behavior of the project-metadata SSOT.

    (Class name preserved: three export manifests under ``tests/`` import
    this symbol by name and are out of edit scope.)
    """

    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            ("acme-widget", "AcmeWidget"),
            ("acme-api-client", "AcmeApiClient"),
            ("a-b-c", "ABC"),
            ("--leading--", "Leading"),
        ],
    )
    def test_derive_class_stem_pascalizes_unlisted_names(
        self, project_name: str, expected_stem: str
    ) -> None:
        assert u.derive_class_stem(project_name) == expected_stem

    @pytest.mark.parametrize(
        ("project_name", "declared_stem"), c.SPECIAL_NAME_OVERRIDES
    )
    def test_derive_class_stem_honours_declared_overrides_case_insensitively(
        self, project_name: str, declared_stem: str
    ) -> None:
        assert u.derive_class_stem(project_name) == declared_stem
        assert u.derive_class_stem(project_name.upper()) == declared_stem

    def test_derive_class_stem_returns_empty_for_empty_name(self) -> None:
        assert u.derive_class_stem("") == ""

    @staticmethod
    def _metadata(tmp_path: Path, **overrides: object) -> m.ProjectMetadata:
        payload: dict[str, object] = {
            "root": tmp_path,
            "package_name": "flext_ldif",
            "class_stem": "FlextLdif",
            "project": {"name": "flext-ldif", "version": "1.2.3"},
            "flext": {},
        }
        payload.update(overrides)
        return m.ProjectMetadata.model_validate(payload)

    def test_project_metadata_exposes_declared_field_values(
        self, tmp_path: Path
    ) -> None:
        metadata = self._metadata(tmp_path)

        assert metadata.package_name == "flext_ldif"
        assert metadata.class_stem == "FlextLdif"
        assert metadata.project.name == "flext-ldif"
        assert metadata.project.version == "1.2.3"

    def test_project_metadata_is_immutable(self, tmp_path: Path) -> None:
        metadata = self._metadata(tmp_path)

        with pytest.raises(m.ValidationError):
            metadata.package_name = "other"

    def test_project_metadata_rejects_unknown_field(self, tmp_path: Path) -> None:
        with pytest.raises(m.ValidationError):
            m.ProjectMetadata.model_validate({
                "root": tmp_path,
                "package_name": "flext_ldif",
                "class_stem": "FlextLdif",
                "project": {"name": "flext-ldif", "version": "1.0.0"},
                "flext": {},
                "unexpected": "value",
            })

    def test_project_metadata_model_dump_exposes_public_fields(
        self, tmp_path: Path
    ) -> None:
        dumped = self._metadata(tmp_path).model_dump()

        assert dumped["package_name"] == "flext_ldif"
        assert dumped["class_stem"] == "FlextLdif"
        assert dumped["project"]["name"] == "flext-ldif"

    def test_tool_flext_root_builds_default_subtables(self) -> None:
        tool = m.ProjectToolFlext()

        assert tool.project is not None
        assert tool.docs is not None
        assert tool.workspace is not None
