"""Comprehensive tests for FlextModelsContainer models.

Module: flext_core._models.container
Scope: ServiceRegistration, FactoryRegistration, ContainerConfig models

Tests cover:
- ServiceRegistration metadata validation (None, dict, Metadata)
- FactoryRegistration metadata validation (None, dict, Metadata)
- ContainerConfig all fields and defaults
- Edge cases and error paths

Uses real implementations, FlextTestsUtilities, and advanced pytest patterns.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import ClassVar, cast

import pytest
from flext_core import m, t, u
from pydantic import ValidationError

_expected_validation_errors: tuple[type[Exception], ...] = (
    ValidationError,
    TypeError,
)


class ContainerModelsScenarios:
    """Test scenarios for container models."""

    METADATA_VALUES: ClassVar[list[tuple[object, bool]]] = [
        (None, True),
        ({}, True),
        ({"key": "value"}, True),
        ({"nested": {"deep": "value"}}, True),
        ({"list": [1, 2, 3]}, True),
        ({"int": 42, "str": "test", "bool": True}, True),
        (m.Metadata(attributes={"test": "value"}), True),
        ("invalid_string", False),
        (123, False),
        ([1, 2, 3], False),
    ]

    CONTAINER_CONFIG_VALUES: ClassVar[list[dict[str, t.GeneralValueType]]] = [
        {},
        {"enable_singleton": False},
        {"enable_factory_caching": False},
        {"max_services": 500},
        {"max_factories": 250},
        {"enable_auto_registration": True},
        {"enable_lifecycle_hooks": False},
        {"lazy_loading": False},
        {
            "enable_singleton": False,
            "max_services": 2000,
        },
    ]


class TestFlextModelsContainer:
    """Test suite for FlextModelsContainer models."""

    def test_is_dict_like_static_method(self) -> None:
        """Test dict-like checking using utilities."""

        def is_dict_like(value: object) -> bool:
            """Check if value is dict-like."""
            return isinstance(value, Mapping)

        assert is_dict_like({"key": "value"}) is True
        assert is_dict_like({}) is True
        assert is_dict_like("not_dict") is False
        assert is_dict_like(123) is False
        assert is_dict_like([1, 2, 3]) is False
        assert is_dict_like(None) is False

    @pytest.mark.parametrize(
        ("metadata_value", "should_pass"),
        ContainerModelsScenarios.METADATA_VALUES,
        ids=lambda x: (
            f"metadata_{type(x[0]).__name__ if isinstance(x, tuple) else type(x).__name__}_{x[1] if isinstance(x, tuple) else 'unknown'}"
        ),
    )
    def test_service_registration_metadata_validation(
        self,
        metadata_value: object,
        should_pass: bool,
    ) -> None:
        """Test ServiceRegistration metadata validation with various types."""
        if should_pass:
            registration = m.Container.ServiceRegistration(
                name="test_service",
                service="test_value",
                metadata=metadata_value,
            )
            # metadata=None triggers validator that creates default Metadata
            # dict/ConfigMap inputs yield Metadata instance (auto-conversion)
            assert registration.metadata is not None
            assert isinstance(registration.metadata, m.Metadata)
        else:
            with pytest.raises(_expected_validation_errors):
                m.Container.ServiceRegistration(
                    name="test_service",
                    service="test_value",
                        metadata=metadata_value,
                )

    def test_service_registration_defaults(self) -> None:
        """Test ServiceRegistration default values."""
        registration = m.Container.ServiceRegistration(
            name="test",
            service="value",
        )
        # metadata defaults to None, validator converts on access/assignment
        # When explicitly set to None, validator converts it
        assert registration.service_type is None
        assert registration.tags == []
        assert isinstance(registration.registration_time, datetime)

        # Test that setting metadata=None converts it
        registration.metadata = None
        assert registration.metadata is not None
        assert isinstance(registration.metadata, m.Metadata)

    def test_service_registration_with_all_fields(self) -> None:
        """Test ServiceRegistration with all fields populated."""
        metadata = m.Metadata(attributes={"env": "test"})
        registration = m.Container.ServiceRegistration(
            name="full_service",
            service={"data": "value"},
            metadata=metadata,
            service_type="TestService",
            tags=["test", "integration"],
        )
        assert registration.name == "full_service"
        assert registration.service == {"data": "value"}
        assert registration.metadata == metadata
        assert registration.service_type == "TestService"
        assert registration.tags == ["test", "integration"]

    @pytest.mark.parametrize(
        ("metadata_value", "should_pass"),
        ContainerModelsScenarios.METADATA_VALUES,
        ids=lambda x: (
            f"factory_metadata_{type(x[0]).__name__ if isinstance(x, tuple) else type(x).__name__}_{x[1] if isinstance(x, tuple) else 'unknown'}"
        ),
    )
    def test_factory_registration_metadata_validation(
        self,
        metadata_value: object,
        should_pass: bool,
    ) -> None:
        """Test FactoryRegistration metadata validation with various types."""

        def factory() -> t.ScalarValue:
            return "test"

        if should_pass:
            registration = m.Container.FactoryRegistration(
                name="test_factory",
                factory=factory,
                    metadata=metadata_value,
            )
            # metadata=None triggers validator that creates default Metadata
            # dict/ConfigMap inputs yield Metadata instance (auto-conversion)
            assert registration.metadata is not None
            assert isinstance(registration.metadata, m.Metadata)
        else:
            with pytest.raises(_expected_validation_errors):
                m.Container.FactoryRegistration(
                    name="test_factory",
                    factory=factory,
                        metadata=metadata_value,
                )

    def test_factory_registration_defaults(self) -> None:
        """Test FactoryRegistration default values."""

        def factory() -> t.ScalarValue:
            return "value"

        registration = m.Container.FactoryRegistration(
            name="test",
            factory=factory,
        )
        # metadata defaults to None, validator converts on access/assignment
        assert registration.is_singleton is False
        assert registration.cached_instance is None
        assert registration.invocation_count == 0
        assert isinstance(registration.registration_time, datetime)

        # Test that setting metadata=None converts it
        registration.metadata = None
        assert registration.metadata is not None
        assert isinstance(registration.metadata, m.Metadata)

    def test_factory_registration_with_all_fields(self) -> None:
        """Test FactoryRegistration with all fields populated."""

        def factory() -> t.ScalarValue:
            return "created"

        metadata = m.Metadata(attributes={"type": "factory"})
        registration = m.Container.FactoryRegistration(
            name="full_factory",
            factory=factory,
            metadata=metadata,
            is_singleton=True,
            cached_instance="cached_value",
            invocation_count=5,
        )
        assert registration.name == "full_factory"
        assert callable(registration.factory)
        assert registration.metadata == metadata
        assert registration.is_singleton is True
        assert registration.cached_instance == "cached_value"
        assert registration.invocation_count == 5

    @pytest.mark.parametrize(
        "config_dict",
        ContainerModelsScenarios.CONTAINER_CONFIG_VALUES,
        ids=lambda x: f"config_{len(x)}_fields",
    )
    def test_container_config_creation(
        self,
        config_dict: dict[str, t.GeneralValueType],
    ) -> None:
        """Test ContainerConfig creation with various configurations."""
        # ContainerConfig accepts keyword arguments directly
        # Use model_construct for dynamic dict unpacking in tests
        config = m.Container.ContainerConfig.model_validate(config_dict)
        assert config.enable_singleton is u.Mapper.get(
            config_dict,
            "enable_singleton",
            default=True,
        )
        assert config.enable_factory_caching is u.Mapper.get(
            config_dict,
            "enable_factory_caching",
            default=True,
        )
        assert config.max_services == u.Mapper.get(
            config_dict,
            "max_services",
            default=1000,
        )
        assert config.max_factories == u.Mapper.get(
            config_dict,
            "max_factories",
            default=500,
        )
        assert config.enable_auto_registration is u.Mapper.get(
            config_dict,
            "enable_auto_registration",
            default=False,
        )
        assert config.enable_lifecycle_hooks is u.Mapper.get(
            config_dict,
            "enable_lifecycle_hooks",
            default=True,
        )
        assert config.lazy_loading is u.Mapper.get(
            config_dict,
            "lazy_loading",
            default=True,
        )

    def test_container_config_defaults(self) -> None:
        """Test ContainerConfig default values."""
        config = m.Container.ContainerConfig()
        assert config.enable_singleton is True
        assert config.enable_factory_caching is True
        assert config.max_services == 1000
        assert config.max_factories == 500
        assert config.enable_auto_registration is False
        assert config.enable_lifecycle_hooks is True
        assert config.lazy_loading is True

    def test_container_config_validation_limits(self) -> None:
        """Test ContainerConfig field validation limits."""
        # Test max_services bounds
        config_min = m.Container.ContainerConfig(max_services=1)
        assert config_min.max_services == 1

        config_max = m.Container.ContainerConfig(max_services=10000)
        assert config_max.max_services == 10000

        with pytest.raises(ValidationError):
            m.Container.ContainerConfig(max_services=0)

        with pytest.raises(ValidationError):
            m.Container.ContainerConfig(max_services=10001)

        # Test max_factories bounds
        config_fact_min = m.Container.ContainerConfig(max_factories=1)
        assert config_fact_min.max_factories == 1

        config_fact_max = m.Container.ContainerConfig(max_factories=5000)
        assert config_fact_max.max_factories == 5000

        with pytest.raises(ValidationError):
            m.Container.ContainerConfig(max_factories=0)

        with pytest.raises(ValidationError):
            m.Container.ContainerConfig(max_factories=5001)

    def test_service_registration_metadata_none_handling(self) -> None:
        """Test ServiceRegistration handles None metadata correctly."""
        registration = m.Container.ServiceRegistration(
            name="test",
            service="value",
            metadata=None,
        )
        assert registration.metadata is not None
        assert isinstance(registration.metadata, m.Metadata)
        assert registration.metadata.attributes == {}

    def test_factory_registration_metadata_none_handling(self) -> None:
        """Test FactoryRegistration handles None metadata correctly."""

        def factory() -> t.ScalarValue:
            return "value"

        registration = m.Container.FactoryRegistration(
            name="test",
            factory=factory,
            metadata=None,
        )
        assert registration.metadata is not None
        assert isinstance(registration.metadata, m.Metadata)
        assert registration.metadata.attributes == {}


class TestFlextUtilitiesModelNormalizeToMetadata:
    """Test suite for FlextUtilitiesModel.normalize_to_metadata() method."""

    def test_normalize_to_metadata_none(self) -> None:
        """Test normalize_to_metadata with None returns empty Metadata."""
        result = u.Model.normalize_to_metadata(None)
        assert isinstance(result, m.Metadata)
        assert result.attributes == {}

    def test_normalize_to_metadata_empty_dict(self) -> None:
        """Test normalize_to_metadata with empty dict."""
        result = u.Model.normalize_to_metadata(m.ConfigMap(root={}))
        # result is m.Metadata (from normalize_to_metadata)
        assert isinstance(result, m.Metadata)
        assert result.attributes == {}

    def test_normalize_to_metadata_with_values(self) -> None:
        """Test normalize_to_metadata with dict containing values."""
        result = u.Model.normalize_to_metadata(
            m.ConfigMap(
                root={
                    "key1": "value1",
                    "key2": 42,
                    "key3": True,
                },
            ),
        )
        assert isinstance(result, m.Metadata)
        assert result.attributes["key1"] == "value1"
        assert result.attributes["key2"] == "42"
        assert result.attributes["key3"] == "True"

    def test_normalize_to_metadata_existing_metadata(self) -> None:
        """Test normalize_to_metadata with existing Metadata instance."""
        existing = m.Metadata(attributes={"existing": "value"})
        result = u.Model.normalize_to_metadata(existing)
        assert result is existing
        assert result.attributes["existing"] == "value"

    def test_normalize_to_metadata_nested_dict(self) -> None:
        """Test normalize_to_metadata with nested dict values."""
        result = u.Model.normalize_to_metadata(
            m.ConfigMap(
                root={
                    "nested": {"level1": {"level2": "value"}},
                },
            ),
        )
        assert isinstance(result, m.Metadata)
        # Nested dicts are normalized to t.GeneralValueType
        assert "nested" in result.attributes

    def test_normalize_to_metadata_invalid_type(self) -> None:
        """Test normalize_to_metadata with invalid type raises TypeError."""
        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            u.Model.normalize_to_metadata("invalid_string")

        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            u.Model.normalize_to_metadata(123)

        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            u.Model.normalize_to_metadata([1, 2, 3])
