"""Behavioral contract tests for the project-metadata SSOT surface.

Exercises the OBSERVABLE public behavior consumers depend on:

- ``m.pascalize`` / ``m.derive_class_stem`` name-derivation outputs,
  including the special-name overrides and the empty-input error path.
- ``m.ProjectMetadata`` invariants: immutability (``frozen``),
  ``extra="forbid"``, field defaults and computed ``package_name`` /
  ``class_stem`` properties.
- ``m.ProjectNamespaceConfig`` / ``m.ProjectToolFlext`` public defaults.
- ``u.read_project_constants`` producing a populated ``ProjectConstants``
  and being cached (same object for the same package).
- ``c.PYPROJECT_FILENAME`` public constant value.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
        ("slug", "expected"),
        [
            ("flext-ldif", "FlextLdif"),
            ("flext_api_client", "FlextApiClient"),
            ("flext-core", "FlextCore"),
            ("single", "Single"),
            ("a-b-c", "ABC"),
            ("--leading--", "Leading"),
        ],
    )
    def test_pascalize_converts_kebab_and_snake_to_pascalcase(
        self,
        slug: str,
        expected: str,
    ) -> None:
        # Act
        result = m.pascalize(slug)

        # Assert
        assert result == expected

    @pytest.mark.parametrize(
        ("project_name", "expected_stem"),
        [
            ("flext-core", "Flext"),
            ("flext", "FlextRoot"),
            ("FLEXT-CORE", "Flext"),
            ("flext-ldif", "FlextLdif"),
            ("flext-api-client", "FlextApiClient"),
        ],
    )
    def test_derive_class_stem_applies_overrides_then_pascalizes(
        self,
        project_name: str,
        expected_stem: str,
    ) -> None:
        # Act
        result = m.derive_class_stem(project_name)

        # Assert
        assert result == expected_stem

    def test_derive_class_stem_is_case_insensitive_for_overrides(self) -> None:
        # Arrange / Act / Assert — mixed case resolves to the same override.
        assert m.derive_class_stem("Flext-Core") == m.derive_class_stem("flext-core")

    def test_derive_class_stem_rejects_empty_name(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="empty project name"):
            m.derive_class_stem("")

    def test_u_and_m_derive_class_stem_agree(self) -> None:
        # The utilities facade re-exposes the same SSOT callable via MRO;
        # both must return identical results for the same input.
        assert u.derive_class_stem("flext-core") == m.derive_class_stem("flext-core")
        assert u.pascalize("flext-ldif") == m.pascalize("flext-ldif")

    def test_project_metadata_exposes_declared_field_values(self) -> None:
        # Arrange / Act
        metadata = m.ProjectMetadata(
            name="flext-ldif",
            version="1.2.3",
            license="MIT",
            root="/tmp/flext-ldif",
        )

        # Assert — public field state.
        assert metadata.name == "flext-ldif"
        assert metadata.version == "1.2.3"
        assert metadata.license == "MIT"

    def test_project_metadata_optional_fields_default_empty(self) -> None:
        # Act
        metadata = m.ProjectMetadata(
            name="flext-ldif",
            version="1.0.0",
            license="MIT",
            root="/tmp",
        )

        # Assert
        assert metadata.description == ""
        assert metadata.authors == ()
        assert metadata.url == ""
        assert metadata.requires_python == ""

    @pytest.mark.parametrize(
        ("name", "expected_package", "expected_stem"),
        [
            ("flext-ldif", "flext_ldif", "FlextLdif"),
            ("flext-core", "flext_core", "Flext"),
            ("flext-api-client", "flext_api_client", "FlextApiClient"),
        ],
    )
    def test_project_metadata_computed_properties(
        self,
        name: str,
        expected_package: str,
        expected_stem: str,
    ) -> None:
        # Arrange
        metadata = m.ProjectMetadata(
            name=name,
            version="1.0.0",
            license="MIT",
            root="/tmp",
        )

        # Assert — derived public accessors.
        assert metadata.package_name == expected_package
        assert metadata.class_stem == expected_stem

    def test_project_metadata_is_immutable(self) -> None:
        # Arrange
        metadata = m.ProjectMetadata(
            name="flext-ldif",
            version="1.0.0",
            license="MIT",
            root="/tmp",
        )

        # Act / Assert — frozen model rejects mutation.
        with pytest.raises(m.ValidationError):
            metadata.name = "other"

    def test_project_metadata_rejects_unknown_field(self) -> None:
        # Act / Assert — extra="forbid" contract.
        with pytest.raises(m.ValidationError):
            m.ProjectMetadata(
                name="flext-ldif",
                version="1.0.0",
                license="MIT",
                root="/tmp",
                unexpected="value",
            )

    @pytest.mark.parametrize("blank_field", ["name", "version", "license"])
    def test_project_metadata_rejects_blank_required_strings(
        self,
        blank_field: str,
    ) -> None:
        # Arrange
        kwargs: dict[str, str] = {
            "name": "flext-ldif",
            "version": "1.0.0",
            "license": "MIT",
            "root": "/tmp",
        }
        kwargs[blank_field] = ""

        # Act / Assert — min_length=1 enforced.
        with pytest.raises(m.ValidationError):
            m.ProjectMetadata(**kwargs)

    def test_project_metadata_model_dump_exposes_public_fields(self) -> None:
        # Arrange
        metadata = m.ProjectMetadata(
            name="flext-ldif",
            version="1.0.0",
            license="MIT",
            root="/tmp",
        )

        # Act
        dumped = metadata.model_dump()

        # Assert — model_dump is the public serialization contract.
        assert dumped["name"] == "flext-ldif"
        assert dumped["version"] == "1.0.0"
        assert dumped["license"] == "MIT"

    def test_namespace_config_defaults(self) -> None:
        # Act
        config = m.ProjectNamespaceConfig(project_name="flext-ldif")

        # Assert — public default state.
        assert config.project_name == "flext-ldif"
        assert config.enabled is True
        assert config.scan_dirs == ()
        assert config.include_dynamic_dirs is False

    def test_namespace_config_requires_project_name(self) -> None:
        # Act / Assert — project_name has min_length=1.
        with pytest.raises(m.ValidationError):
            m.ProjectNamespaceConfig(project_name="")

    def test_tool_flext_root_builds_default_subtables(self) -> None:
        # Act
        tool = m.ProjectToolFlext()

        # Assert — every [tool.flext.*] sub-table is populated by default.
        assert tool.project is not None
        assert tool.namespace is not None
        assert tool.docs is not None
        assert tool.aliases is not None
        assert tool.workspace is not None

    def test_read_project_constants_returns_populated_constants(self) -> None:
        # Act
        constants = u.read_project_constants("flext-core")

        # Assert — public fields consumers rely on.
        assert constants.PYTHON_PACKAGE_NAME == "flext_core"
        assert constants.CLASS_STEM == "Flext"
        assert constants.PYPROJECT_FILENAME == c.PYPROJECT_FILENAME
        assert constants.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert constants.PACKAGE_VERSION

    def test_read_project_constants_is_cached_per_package(self) -> None:
        # Act — repeated reads for the same package return the identical object.
        first = u.read_project_constants("flext-core")
        second = u.read_project_constants("flext-core")

        # Assert
        assert first is second

    def test_pyproject_filename_constant(self) -> None:
        # Assert — stable public constant value.
        assert c.PYPROJECT_FILENAME == "pyproject.toml"
