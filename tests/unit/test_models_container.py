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

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import ClassVar

import pytest
from flext_tests import t, tm
from pydantic import BaseModel, ValidationError

from tests import m, u


class TestFlextModelsContainer:
    """Test suite for FlextModelsContainer models."""

    _EXPECTED_VALIDATION_ERRORS: ClassVar[tuple[type[Exception], ...]] = (
        ValidationError,
        TypeError,
    )

    class _ContainerModelsScenarios:
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
        CONTAINER_CONFIG_VALUES: ClassVar[
            list[dict[str, t.NormalizedValue | BaseModel]]
        ] = [
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

    @staticmethod
    def _normalize_metadata_obj(value: list[int]) -> m.Metadata:
        """Call ensure_metadata with arbitrary object for error-path testing."""
        fn: Callable[..., m.Metadata] = getattr(u, "ensure_metadata")
        return fn(value)

    def test_is_dict_like_static_method(self) -> None:
        """Test dict-like checking using utilities."""

        def is_dict_like(value: t.NormalizedValue) -> bool:
            """Check if value is dict-like."""
            return isinstance(value, Mapping)

        tm.that(is_dict_like({"key": "value"}), eq=True)
        tm.that(is_dict_like({}), eq=True)
        tm.that(is_dict_like("not_dict"), eq=False)
        tm.that(is_dict_like(123), eq=False)
        tm.that(is_dict_like([1, 2, 3]), eq=False)
        tm.that(is_dict_like(None), eq=False)

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
            tm.that(hasattr(registration.metadata, "attributes"), eq=True)
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
        tm.that(isinstance(registration.registration_time, datetime), eq=True)
        registration.metadata = None
        tm.that(registration.metadata, none=False)
        tm.that(hasattr(registration.metadata, "attributes"), eq=True)

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
        tm.that(registration.service, eq=t.ConfigMap(root={"data": "value"}))
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
            tm.that(hasattr(registration.metadata, "attributes"), eq=True)
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
        tm.that(registration.is_singleton, eq=False)
        tm.that(registration.cached_instance, none=True)
        tm.that(registration.invocation_count, eq=0)
        tm.that(isinstance(registration.registration_time, datetime), eq=True)
        registration.metadata = None
        tm.that(registration.metadata, none=False)
        tm.that(hasattr(registration.metadata, "attributes"), eq=True)

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
        tm.that(registration.cached_instance, eq="cached_value")
        tm.that(registration.invocation_count, eq=5)

    @pytest.mark.parametrize(
        "config_dict",
        _ContainerModelsScenarios.CONTAINER_CONFIG_VALUES,
        ids=lambda x: f"config_{len(x)}_fields",
    )
    def test_container_config_creation(
        self,
        config_dict: dict[str, t.NormalizedValue | BaseModel],
    ) -> None:
        """Test ContainerConfig creation with various configurations."""
        config = m.ContainerConfig.model_validate(config_dict)
        tm.that(
            config.enable_singleton,
            eq=u.get(
                config_dict,
                "enable_singleton",
                default=True,
            ),
        )
        tm.that(
            config.enable_factory_caching,
            eq=u.get(
                config_dict,
                "enable_factory_caching",
                default=True,
            ),
        )
        tm.that(
            config.max_services,
            eq=u.get(config_dict, "max_services", default=1000),
        )
        tm.that(
            config.max_factories,
            eq=u.get(config_dict, "max_factories", default=500),
        )
        tm.that(
            config.enable_auto_registration,
            eq=u.get(
                config_dict,
                "enable_auto_registration",
                default=False,
            ),
        )
        tm.that(
            config.enable_lifecycle_hooks,
            eq=u.get(
                config_dict,
                "enable_lifecycle_hooks",
                default=True,
            ),
        )
        tm.that(
            config.lazy_loading,
            eq=u.get(
                config_dict,
                "lazy_loading",
                default=True,
            ),
        )

    def test_container_config_defaults(self) -> None:
        """Test ContainerConfig default values."""
        config = m.ContainerConfig()
        tm.that(config.enable_singleton, eq=True)
        tm.that(config.enable_factory_caching, eq=True)
        tm.that(config.max_services, eq=1000)
        tm.that(config.max_factories, eq=500)
        tm.that(config.enable_auto_registration, eq=False)
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

    def test_normalize_to_metadata_none(self) -> None:
        """Test normalize_to_metadata with None returns empty Metadata."""
        result = u.ensure_metadata(None)
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(result.attributes, eq={})

    def test_normalize_to_metadata_empty_dict(self) -> None:
        """Test normalize_to_metadata with empty dict."""
        result = u.ensure_metadata(t.ConfigMap(root={}))
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(result.attributes, eq={})

    def test_normalize_to_metadata_with_values(self) -> None:
        """Test normalize_to_metadata with dict containing values."""
        result = u.ensure_metadata(
            t.ConfigMap(root={"key1": "value1", "key2": 42, "key3": True}),
        )
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(result.attributes["key1"], eq="value1")
        tm.that(result.attributes["key2"], eq=42)
        tm.that(result.attributes["key3"], eq=True)

    def test_normalize_to_metadata_existing_metadata(self) -> None:
        """Test normalize_to_metadata with existing Metadata instance."""
        existing = m.Metadata(attributes={"existing": "value"})
        result = u.ensure_metadata(existing)
        tm.that(result is existing, eq=True)
        tm.that(result.attributes["existing"], eq="value")

    def test_normalize_to_metadata_nested_dict(self) -> None:
        """Test normalize_to_metadata with nested dict values."""
        result = u.ensure_metadata(
            t.ConfigMap(root={"nested": {"level1": {"level2": "value"}}}),
        )
        tm.that(hasattr(result, "attributes"), eq=True)
        tm.that(result.attributes, has="nested")

    def test_normalize_to_metadata_invalid_type(self) -> None:
        """Test normalize_to_metadata with invalid type raises TypeError."""
        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            u.ensure_metadata("invalid_string")
        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            u.ensure_metadata(123)
        with pytest.raises(
            TypeError,
            match=r"metadata must be None, dict, or.*Metadata",
        ):
            self._normalize_metadata_obj([1, 2, 3])
