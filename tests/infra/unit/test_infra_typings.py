"""Tests for FlextInfraTypes facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextTypes
from flext_infra import FlextInfraTypes, t


class TestFlextInfraTypesImport:
    """Test FlextInfraTypes class import and structure."""

    def test_flext_infra_types_is_importable(self) -> None:
        """Test that FlextInfraTypes can be imported."""
        assert FlextInfraTypes is not None

    def test_flext_infra_types_inherits_from_flext_types(self) -> None:
        """Test that FlextInfraTypes extends FlextTypes."""
        assert issubclass(FlextInfraTypes, FlextTypes)

    def test_runtime_alias_t_is_flext_infra_types(self) -> None:
        """Test that t is an alias for FlextInfraTypes."""
        assert t is FlextInfraTypes

    def test_flext_infra_types_has_scalar_type(self) -> None:
        """Test that FlextInfraTypes has Scalar type."""
        assert hasattr(FlextTypes, "Scalar") or hasattr(FlextInfraTypes, "Scalar")

    def test_flext_infra_types_has_general_value_type(self) -> None:
        """Test that FlextInfraTypes has GeneralValueType."""
        assert hasattr(FlextTypes, "GeneralValueType") or hasattr(
            FlextInfraTypes,
            "GeneralValueType",
        )

    def test_flext_infra_types_has_dict_type(self) -> None:
        """Test that FlextInfraTypes has Dict type."""
        assert hasattr(FlextTypes, "Dict") or hasattr(FlextInfraTypes, "Dict")

    def test_flext_infra_types_has_json_value_type(self) -> None:
        """Test that FlextInfraTypes has Serializable type."""
        assert hasattr(FlextTypes, "Serializable") or hasattr(
            FlextInfraTypes, "Serializable"
        )

    def test_flext_infra_types_has_json_dict_type(self) -> None:
        """Test that FlextInfraTypes has ConfigMap or ContainerDict type."""
        assert hasattr(FlextTypes, "ConfigMap") or hasattr(
            FlextInfraTypes.Infra, "ContainerDict"
        )

    def test_flext_infra_types_has_metadata_value_type(self) -> None:
        """Test that FlextInfraTypes has MetadataValue type."""
        assert hasattr(FlextTypes, "MetadataValue") or hasattr(
            FlextInfraTypes,
            "MetadataValue",
        )

    def test_flext_infra_types_has_configuration_mapping_type(self) -> None:
        """Test that FlextInfraTypes has ConfigurationMapping type."""
        assert hasattr(FlextTypes, "ConfigurationMapping") or hasattr(
            FlextInfraTypes,
            "ConfigurationMapping",
        )

    def test_flext_infra_types_has_serializable_type(self) -> None:
        """Test that FlextInfraTypes has Serializable type."""
        assert hasattr(FlextTypes, "Serializable") or hasattr(
            FlextInfraTypes,
            "Serializable",
        )

    def test_flext_infra_types_has_registerable_service_type(self) -> None:
        """Test that FlextInfraTypes has RegisterableService type."""
        assert hasattr(FlextTypes, "RegisterableService") or hasattr(
            FlextInfraTypes,
            "RegisterableService",
        )

    def test_flext_infra_types_has_container_type(self) -> None:
        """Test that FlextInfraTypes has Container type."""
        assert hasattr(FlextTypes, "Container") or hasattr(FlextInfraTypes, "Container")

    def test_flext_infra_types_has_factory_callable_type(self) -> None:
        """Test that FlextInfraTypes has FactoryCallable type."""
        assert hasattr(FlextTypes, "FactoryCallable") or hasattr(
            FlextInfraTypes,
            "FactoryCallable",
        )

    def test_flext_infra_types_has_resource_callable_type(self) -> None:
        """Test that FlextInfraTypes has ResourceCallable type."""
        assert hasattr(FlextTypes, "ResourceCallable") or hasattr(
            FlextInfraTypes,
            "ResourceCallable",
        )

    def test_flext_infra_types_has_validation_namespace(self) -> None:
        """Test that FlextInfraTypes has Validation namespace."""
        assert hasattr(FlextTypes, "Validation") or hasattr(
            FlextInfraTypes,
            "Validation",
        )
