"""Comprehensive tests for FlextModelsContainer models.

Module: flext_core._models.container
Scope: ServiceRegistration, FactoryRegistration, ContainerConfig models

Tests cover:
- ServiceRegistration metadata validation (None, dict, Metadata)
- FactoryRegistration metadata validation (None, dict, Metadata)
- ContainerConfig all fields and defaults
- Edge cases and error paths

Uses real implementations, u, and advanced pytest patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import pytest
from pydantic import ValidationError

from flext_tests import tm
from tests import m, t


class TestFlextModelsContainer:
    """Test suite for FlextModelsContainer models."""

    _EXPECTED_VALIDATION_ERRORS: ClassVar[tuple[type[Exception], ...]] = (
        ValidationError,
        TypeError,
    )

    class _ContainerModelsScenarios:
        """Test scenarios for container models."""

        METADATA_VALUES: ClassVar[Sequence[tuple[t.ValueOrModel, bool]]] = [
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
        CONTAINER_CONFIG_VALUES: ClassVar[Sequence[Mapping[str, t.ValueOrModel]]] = [
            {},
            {"enable_singleton": False},
            {"enable_factory_caching": False},
            {"max_services": 500},
            {"max_factories": 250},
            {"enable_auto_registration": True},
            {"enable_lifecycle_hooks": False},
            {"lazy_loading": False},
            {"enable_singleton": False, "max_services": 2000},
        ]

    @staticmethod
    def _service_reg_with_metadata(
        name: str,
        service: str,
        metadata: t.NormalizedValue,
    ) -> m.ServiceRegistration:
        """Create ServiceRegistration with arbitrary metadata for validation testing."""
        return m.ServiceRegistration.model_validate({
            "name": name,
            "service": service,
            "metadata": metadata,
        })

    @staticmethod
    def _factory_reg_with_metadata(
        name: str,
        factory: Callable[[], t.Scalar],
        metadata: t.NormalizedValue,
    ) -> m.FactoryRegistration:
        """Create FactoryRegistration with arbitrary metadata for validation testing."""
        return m.FactoryRegistration.model_validate({
            "name": name,
            "factory": factory,
            "metadata": metadata,
        })

    def test_is_dict_like_static_method(self) -> None:
        """Test dict-like checking using utilities."""

        def is_dict_like(value: t.NormalizedValue) -> bool:
            """Check if value is dict-like."""
            return isinstance(value, Mapping)

        tm.that(is_dict_like({"key": "value"}), eq=True)
        tm.that(is_dict_like({}), eq=True)
        tm.that(not is_dict_like("not_dict"), eq=True)
        tm.that(not is_dict_like(123), eq=True)
        tm.that(not is_dict_like([1, 2, 3]), eq=True)
        tm.that(not is_dict_like(None), eq=True)

    @pytest.mark.parametrize(
        ("metadata_value", "should_pass"),
        _ContainerModelsScenarios.METADATA_VALUES,
    )
    def test_service_registration_metadata_validation(
        self,
        metadata_value: t.NormalizedValue,
        should_pass: bool,
    ) -> None:
        """Test ServiceRegistration metadata validation with various types."""
        if should_pass:
            registration = self._service_reg_with_metadata(
                "test_service",
                "test_value",
                metadata_value,
            )
            tm.that(registration.metadata, none=False)
        else:
            with pytest.raises(self._EXPECTED_VALIDATION_ERRORS):
                self._service_reg_with_metadata(
                    "test_service",
                    "test_value",
                    metadata_value,
                )

    def test_service_registration_defaults(self) -> None:
        """Test ServiceRegistration default values."""
        registration = m.ServiceRegistration(name="test", service="value")
        tm.that(registration.service_type, none=True)
        tm.that(registration.tags, eq=[])
        tm.that(registration.registration_time, none=False)
        registration.metadata = None
        tm.that(registration.metadata, none=False)

    def test_service_registration_with_all_fields(self) -> None:
        """Test ServiceRegistration with all fields populated."""
        metadata = m.Metadata(attributes={"env": "test"})
        registration = m.ServiceRegistration(
            name="full_service",
            service={"data": "value"},
            metadata=metadata,
            service_type="TestService",
            tags=["test", "integration"],
        )
        tm.that(registration.name, eq="full_service")
        assert registration.service == t.ConfigMap(root={"data": "value"})
        tm.that(registration.metadata, eq=metadata)
        tm.that(registration.service_type, eq="TestService")
        tm.that(registration.tags, eq=["test", "integration"])

    @pytest.mark.parametrize(
        ("metadata_value", "should_pass"),
        _ContainerModelsScenarios.METADATA_VALUES,
    )
    def test_factory_registration_metadata_validation(
        self,
        metadata_value: t.NormalizedValue,
        should_pass: bool,
    ) -> None:
        """Test FactoryRegistration metadata validation with various types."""

        def factory() -> t.Scalar:
            return "test"

        if should_pass:
            registration = self._factory_reg_with_metadata(
                "test_factory",
                factory,
                metadata_value,
            )
            tm.that(registration.metadata, none=False)
        else:
            with pytest.raises(self._EXPECTED_VALIDATION_ERRORS):
                self._factory_reg_with_metadata(
                    "test_factory",
                    factory,
                    metadata_value,
                )

    def test_factory_registration_defaults(self) -> None:
        """Test FactoryRegistration default values."""

        def factory() -> t.Scalar:
            return "value"

        registration = m.FactoryRegistration(name="test", factory=factory)
        tm.that(not registration.is_singleton, eq=True)
        assert registration.cached_instance is None
        tm.that(registration.invocation_count, eq=0)
        tm.that(registration.registration_time, none=False)
        registration.metadata = None
        tm.that(registration.metadata, none=False)

    def test_factory_registration_with_all_fields(self) -> None:
        """Test FactoryRegistration with all fields populated."""

        def factory() -> t.Scalar:
            return "created"

        metadata = m.Metadata(attributes={"type": "factory"})
        registration = m.FactoryRegistration(
            name="full_factory",
            factory=factory,
            metadata=metadata,
            is_singleton=True,
            cached_instance="cached_value",
            invocation_count=5,
        )
        tm.that(registration.name, eq="full_factory")
        tm.that(callable(registration.factory), eq=True)
        tm.that(registration.metadata, eq=metadata)
        tm.that(registration.is_singleton, eq=True)
        assert registration.cached_instance == "cached_value"
        tm.that(registration.invocation_count, eq=5)

    @pytest.mark.parametrize(
        "config_dict",
        _ContainerModelsScenarios.CONTAINER_CONFIG_VALUES,
        ids=lambda x: f"config_{len(x)}_fields",
    )
    def test_container_config_creation(
        self,
        config_dict: Mapping[str, t.ValueOrModel],
    ) -> None:
        """Test ContainerConfig creation with various configurations."""
        config = m.ContainerConfig.model_validate(config_dict)
        tm.that(
            config.enable_singleton,
            eq=config_dict.get("enable_singleton", True),
        )
        tm.that(
            config.enable_factory_caching,
            eq=config_dict.get("enable_factory_caching", True),
        )
        tm.that(
            config.max_services,
            eq=config_dict.get("max_services", 1000),
        )
        tm.that(
            config.max_factories,
            eq=config_dict.get("max_factories", 500),
        )
        tm.that(
            config.enable_auto_registration,
            eq=config_dict.get("enable_auto_registration", False),
        )
        tm.that(
            config.enable_lifecycle_hooks,
            eq=config_dict.get("enable_lifecycle_hooks", True),
        )
        tm.that(
            config.lazy_loading,
            eq=config_dict.get("lazy_loading", True),
        )

    def test_container_config_defaults(self) -> None:
        """Test ContainerConfig default values."""
        config = m.ContainerConfig()
        tm.that(config.enable_singleton, eq=True)
        tm.that(config.enable_factory_caching, eq=True)
        tm.that(config.max_services, eq=1000)
        tm.that(config.max_factories, eq=500)
        tm.that(not config.enable_auto_registration, eq=True)
        tm.that(config.enable_lifecycle_hooks, eq=True)
        tm.that(config.lazy_loading, eq=True)

    def test_container_config_validation_limits(self) -> None:
        """Test ContainerConfig field validation limits."""
        config_min = m.ContainerConfig(max_services=1)
        tm.that(config_min.max_services, eq=1)
        config_max = m.ContainerConfig(max_services=10000)
        tm.that(config_max.max_services, eq=10000)
        with pytest.raises(ValidationError):
            m.ContainerConfig(max_services=0)
        with pytest.raises(ValidationError):
            m.ContainerConfig(max_services=10001)
        config_fact_min = m.ContainerConfig(max_factories=1)
        tm.that(config_fact_min.max_factories, eq=1)
        config_fact_max = m.ContainerConfig(max_factories=5000)
        tm.that(config_fact_max.max_factories, eq=5000)
        with pytest.raises(ValidationError):
            m.ContainerConfig(max_factories=0)
        with pytest.raises(ValidationError):
            m.ContainerConfig(max_factories=5001)

    def test_service_registration_metadata_none_handling(self) -> None:
        """Test ServiceRegistration handles None metadata correctly."""
        registration = m.ServiceRegistration(
            name="test",
            service="value",
            metadata=None,
        )
        tm.that(registration.metadata, none=False)
        if registration.metadata is not None:
            metadata_dump = registration.metadata.model_dump()
            tm.that(metadata_dump.get("attributes", {}), eq={})

    def test_factory_registration_metadata_none_handling(self) -> None:
        """Test FactoryRegistration handles None metadata correctly."""

        def factory() -> t.Scalar:
            return "value"

        registration = m.FactoryRegistration(
            name="test",
            factory=factory,
            metadata=None,
        )
        tm.that(registration.metadata, none=False)
        if registration.metadata is not None:
            metadata_dump = registration.metadata.model_dump()
            tm.that(metadata_dump.get("attributes", {}), eq={})

    def test_resource_registration_metadata_normalized(self) -> None:
        """Test ResourceRegistration with metadata attribute."""
        reg = m.ResourceRegistration(
            name="r1",
            factory=lambda: 1,
            metadata=m.Metadata(attributes={"value": "x"}),
        )
        assert reg.metadata is not None
        assert isinstance(reg.metadata, m.Metadata)
        assert reg.metadata.attributes["value"] == "x"
