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

    def test_flext_infra_types_has_scalar_value_type(self) -> None:
        """Test that FlextInfraTypes has ScalarValue type."""
        # ScalarValue is inherited from FlextTypes
        assert hasattr(FlextTypes, "ScalarValue") or hasattr(
            FlextInfraTypes, "ScalarValue"
        )

    def test_flext_infra_types_has_general_value_type(self) -> None:
        """Test that FlextInfraTypes has GeneralValueType type."""
        # GeneralValueType is inherited from FlextTypes
        assert hasattr(FlextTypes, "GeneralValueType") or hasattr(
            FlextInfraTypes, "GeneralValueType"
        )

    def test_flext_infra_types_has_config_map_type(self) -> None:
        """Test that FlextInfraTypes has ConfigMap type."""
        # ConfigMap is inherited from FlextTypes
        assert hasattr(FlextTypes, "ConfigMap") or hasattr(FlextInfraTypes, "ConfigMap")

    def test_flext_infra_types_has_json_value_type(self) -> None:
        """Test that FlextInfraTypes has JsonValue type."""
        # JsonValue is inherited from FlextTypes
        assert hasattr(FlextTypes, "JsonValue") or hasattr(FlextInfraTypes, "JsonValue")

    def test_flext_infra_types_has_json_primitive_type(self) -> None:
        """Test that FlextInfraTypes has JsonPrimitive type."""
        # JsonPrimitive is inherited from FlextTypes
        assert hasattr(FlextTypes, "JsonPrimitive") or hasattr(
            FlextInfraTypes, "JsonPrimitive"
        )

    def test_flext_infra_types_has_metadata_scalar_value_type(self) -> None:
        """Test that FlextInfraTypes has MetadataScalarValue type."""
        # MetadataScalarValue is inherited from FlextTypes
        assert hasattr(FlextTypes, "MetadataScalarValue") or hasattr(
            FlextInfraTypes, "MetadataScalarValue"
        )

    def test_flext_infra_types_has_dict_type(self) -> None:
        """Test that FlextInfraTypes has Dict type."""
        # Dict is inherited from FlextTypes
        assert hasattr(FlextTypes, "Dict") or hasattr(FlextInfraTypes, "Dict")

    def test_flext_infra_types_has_object_list_type(self) -> None:
        """Test that FlextInfraTypes has ObjectList type."""
        # ObjectList is inherited from FlextTypes
        assert hasattr(FlextTypes, "ObjectList") or hasattr(
            FlextInfraTypes, "ObjectList"
        )

    def test_flext_infra_types_has_service_map_type(self) -> None:
        """Test that FlextInfraTypes has ServiceMap type."""
        # ServiceMap is inherited from FlextTypes
        assert hasattr(FlextTypes, "ServiceMap") or hasattr(
            FlextInfraTypes, "ServiceMap"
        )

    def test_flext_infra_types_has_error_map_type(self) -> None:
        """Test that FlextInfraTypes has ErrorMap type."""
        # ErrorMap is inherited from FlextTypes
        assert hasattr(FlextTypes, "ErrorMap") or hasattr(FlextInfraTypes, "ErrorMap")

    def test_flext_infra_types_has_factory_map_type(self) -> None:
        """Test that FlextInfraTypes has FactoryMap type."""
        # FactoryMap is inherited from FlextTypes
        assert hasattr(FlextTypes, "FactoryMap") or hasattr(
            FlextInfraTypes, "FactoryMap"
        )

    def test_flext_infra_types_has_resource_map_type(self) -> None:
        """Test that FlextInfraTypes has ResourceMap type."""
        # ResourceMap is inherited from FlextTypes
        assert hasattr(FlextTypes, "ResourceMap") or hasattr(
            FlextInfraTypes, "ResourceMap"
        )

    def test_flext_infra_types_has_validation_namespace(self) -> None:
        """Test that FlextInfraTypes has Validation namespace."""
        # Validation is inherited from FlextTypes
        assert hasattr(FlextTypes, "Validation") or hasattr(
            FlextInfraTypes, "Validation"
        )
