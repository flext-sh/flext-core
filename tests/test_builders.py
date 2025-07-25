"""Tests for FlextBuilder patterns and factory utilities.

Comprehensive tests for all builder patterns, factory utilities, and
boilerplate reduction features.
"""

from __future__ import annotations

import pytest

from flext_core.builders import FlextConfigBuilder
from flext_core.builders import FlextFactory
from flext_core.builders import FlextServiceBuilder
from flext_core.builders import FlextSingletonFactory
from flext_core.builders import build_config
from flext_core.builders import build_service_config
from flext_core.builders import create_factory
from flext_core.builders import create_singleton_factory


class TestFlextServiceBuilder:
    """Test FlextServiceBuilder functionality."""

    def test_empty_builder(self) -> None:
        """Test building empty service configuration."""
        builder = FlextServiceBuilder()
        result = builder.build()

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["services"] == {}
        assert config["factories"] == {}
        assert config["singletons"] == []

    def test_add_service(self) -> None:
        """Test adding a service instance."""
        service = object()
        builder = FlextServiceBuilder()

        result = builder.add_service("test_service", service).build()

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["services"]["test_service"] is service

    def test_add_service_empty_name(self) -> None:
        """Test adding service with empty name."""
        builder = FlextServiceBuilder()

        result = builder.add_service("", object()).build()

        assert result.is_failure
        assert "Service name cannot be empty" in (result.error or "")

    def test_add_service_duplicate(self) -> None:
        """Test adding duplicate service."""
        builder = FlextServiceBuilder()

        result = (
            builder.add_service("test", object()).add_service("test", object()).build()
        )

        assert result.is_failure
        assert "already registered" in (result.error or "")

    def test_add_factory(self) -> None:
        """Test adding a factory function."""

        def factory() -> str:
            return "created"

        builder = FlextServiceBuilder()
        result = builder.add_factory("test_factory", factory).build()

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["factories"]["test_factory"] is factory
        assert "test_factory" in config["singletons"]

    def test_add_factory_non_singleton(self) -> None:
        """Test adding factory as non-singleton."""

        def factory() -> str:
            return "created"

        builder = FlextServiceBuilder()
        result = builder.add_factory("test_factory", factory, singleton=False).build()

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["factories"]["test_factory"] is factory
        assert "test_factory" not in config["singletons"]

    def test_add_factory_invalid(self) -> None:
        """Test adding non-callable factory."""
        builder = FlextServiceBuilder()

        result = builder.add_factory("test", "not_callable").build()

        assert result.is_failure
        assert "must be callable" in (result.error or "")

    def test_add_services_bulk(self) -> None:
        """Test adding multiple services at once."""
        services = {
            "service1": object(),
            "service2": object(),
            "service3": object(),
        }

        builder = FlextServiceBuilder()
        result = builder.add_services(**services).build()

        assert result.is_success
        config = result.data
        assert config is not None
        for name, service in services.items():
            assert config["services"][name] is service

    def test_can_build_validation(self) -> None:
        """Test can_build method."""
        builder = FlextServiceBuilder()
        assert builder.can_build() is True

        # Add error
        builder.add_service("", object())
        assert builder.can_build() is False

    def test_get_errors(self) -> None:
        """Test get_errors method."""
        builder = FlextServiceBuilder()

        builder.add_service("", object())
        builder.add_factory("test", "not_callable")

        errors = builder.get_errors()
        assert len(errors) == 2
        assert "Service name cannot be empty" in errors[0]
        assert "must be callable" in errors[1]

    def test_builder_reuse_prevention(self) -> None:
        """Test that builder cannot be reused after build."""
        builder = FlextServiceBuilder()
        builder.add_service("test", object())

        # First build should work
        result1 = builder.build()
        assert result1.is_success

        # Second build should fail
        with pytest.raises(ValueError, match="already been used"):
            builder.build()


class TestFlextConfigBuilder:
    """Test FlextConfigBuilder functionality."""

    def test_empty_config(self) -> None:
        """Test building empty configuration."""
        builder = FlextConfigBuilder()
        result = builder.build()

        assert result.is_success
        assert result.data == {}

    def test_set_values(self) -> None:
        """Test setting configuration values."""
        builder = FlextConfigBuilder()

        result = (
            builder.set("key1", "value1")
            .set("key2", 42)
            .set("key3", True)  # noqa: FBT003
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["key1"] == "value1"
        assert config["key2"] == 42
        assert config["key3"] is True

    def test_set_empty_key(self) -> None:
        """Test setting value with empty key."""
        builder = FlextConfigBuilder()

        result = builder.set("", "value").build()

        assert result.is_failure
        assert "Configuration key cannot be empty" in (result.error or "")

    def test_set_required(self) -> None:
        """Test setting required values."""
        builder = FlextConfigBuilder()

        result = (
            builder.set_required("required_key", "value")
            .set("optional_key", "optional")
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["required_key"] == "value"
        assert config["optional_key"] == "optional"

    def test_missing_required_keys(self) -> None:
        """Test validation of missing required keys."""
        builder = FlextConfigBuilder()

        result = (
            builder.require("required1", "required2")
            .set("required1", "value1")
            # Missing required2
            .build()
        )

        assert result.is_failure
        assert "Missing required keys: required2" in (result.error or "")

    def test_set_if_not_none(self) -> None:
        """Test conditional setting based on None check."""
        builder = FlextConfigBuilder()

        result = (
            builder.set_if_not_none("key1", "value1")
            .set_if_not_none("key2", None)
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["key1"] == "value1"
        assert "key2" not in config

    def test_set_default(self) -> None:
        """Test setting default values."""
        builder = FlextConfigBuilder()

        result = (
            builder.set("existing", "original")
            .set_default("existing", "default")  # Should not override
            .set_default("new_key", "default")  # Should set
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["existing"] == "original"
        assert config["new_key"] == "default"

    def test_merge_config(self) -> None:
        """Test merging another configuration."""
        other_config = {
            "key1": "from_other",
            "key2": "other_value",
        }

        builder = FlextConfigBuilder()

        result = builder.set("key1", "original").merge(other_config).build()

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["key1"] == "from_other"  # Merged value wins
        assert config["key2"] == "other_value"


class TestFlextFactory:
    """Test FlextFactory functionality."""

    def test_create_success(self) -> None:
        """Test successful object creation."""

        def factory_func() -> str:
            return "created"

        factory = FlextFactory(factory_func)
        result = factory.create()

        assert result.is_success
        assert result.data == "created"
        assert factory.created_count == 1

    def test_create_with_args(self) -> None:
        """Test object creation with arguments."""

        def factory_func(prefix: str, value: int) -> str:
            return f"{prefix}_{value}"

        factory = FlextFactory(factory_func)
        result = factory.create("test", 42)

        assert result.is_success
        assert result.data == "test_42"

    def test_create_failure(self) -> None:
        """Test failed object creation."""

        def failing_factory() -> str:
            msg = "Creation failed"
            raise ValueError(msg)

        factory = FlextFactory(failing_factory)
        result = factory.create()

        assert result.is_failure
        assert "Creation failed" in (result.error or "")
        assert factory.created_count == 0

    def test_try_create(self) -> None:
        """Test try_create method."""

        def factory_func() -> str:
            return "success"

        factory = FlextFactory(factory_func)
        result = factory.try_create()

        assert result == "success"

    def test_try_create_failure(self) -> None:
        """Test try_create with failure."""

        def failing_factory() -> str:
            msg = "Failed"
            raise ValueError(msg)

        factory = FlextFactory(failing_factory)

        with pytest.raises(ValueError, match="Failed"):
            factory.try_create()

    def test_create_many_success(self) -> None:
        """Test creating multiple objects."""
        counter = 0

        def factory_func() -> int:
            nonlocal counter
            counter += 1
            return counter

        factory = FlextFactory(factory_func)
        result = factory.create_many(3)

        assert result.is_success
        assert result.data == [1, 2, 3]
        assert factory.created_count == 3

    def test_create_many_failure(self) -> None:
        """Test create_many with failure."""
        counter = 0

        def failing_factory() -> int:
            nonlocal counter
            counter += 1
            if counter == 2:
                msg = "Failed on second"
                raise ValueError(msg)
            return counter

        factory = FlextFactory(failing_factory)
        result = factory.create_many(3)

        assert result.is_failure
        assert "Failed to create item 2" in (result.error or "")

    def test_create_many_negative_count(self) -> None:
        """Test create_many with negative count."""
        factory = FlextFactory(lambda: "test")
        result = factory.create_many(-1)

        assert result.is_failure
        assert "Count cannot be negative" in (result.error or "")

    def test_invalid_factory_function(self) -> None:
        """Test factory with non-callable."""
        with pytest.raises(ValueError, match="must be callable"):
            FlextFactory("not_callable")


class TestFlextSingletonFactory:
    """Test FlextSingletonFactory functionality."""

    def test_singleton_creation(self) -> None:
        """Test singleton instance creation."""
        counter = 0

        def factory_func() -> int:
            nonlocal counter
            counter += 1
            return counter

        factory = FlextSingletonFactory(factory_func)

        # First call creates instance
        result1 = factory.get_instance()
        assert result1.is_success
        assert result1.data == 1
        assert factory.is_created

        # Second call returns same instance
        result2 = factory.get_instance()
        assert result2.is_success
        assert result2.data == 1  # Same value, not incremented

        # Factory function was only called once
        assert counter == 1

    def test_singleton_failure(self) -> None:
        """Test singleton creation failure."""

        def failing_factory() -> str:
            msg = "Creation failed"
            raise ValueError(msg)

        factory = FlextSingletonFactory(failing_factory)
        result = factory.get_instance()

        assert result.is_failure
        assert "Creation failed" in (result.error or "")
        assert not factory.is_created

    def test_singleton_reset(self) -> None:
        """Test resetting singleton factory."""
        counter = 0

        def factory_func() -> int:
            nonlocal counter
            counter += 1
            return counter

        factory = FlextSingletonFactory(factory_func)

        # Create first instance
        result1 = factory.get_instance()
        assert result1.data == 1

        # Reset and create new instance
        factory.reset()
        assert not factory.is_created

        result2 = factory.get_instance()
        assert result2.data == 2  # New instance

    def test_singleton_with_args(self) -> None:
        """Test singleton with arguments."""

        def factory_func(prefix: str) -> str:
            return f"{prefix}_singleton"

        factory = FlextSingletonFactory(factory_func)

        # First call with args
        result1 = factory.get_instance("test")
        assert result1.data == "test_singleton"

        # Second call ignores args (returns cached instance)
        result2 = factory.get_instance("different")
        assert result2.data == "test_singleton"

    def test_invalid_singleton_factory(self) -> None:
        """Test singleton factory with non-callable."""
        with pytest.raises(TypeError, match="must be callable"):
            FlextSingletonFactory("not_callable")


class TestUtilityFunctions:
    """Test utility functions for builders and factories."""

    def test_build_service_config(self) -> None:
        """Test build_service_config utility."""
        services = {
            "service1": object(),
            "service2": object(),
        }

        result = build_service_config(**services)

        assert result.is_success
        config = result.data
        assert config is not None
        assert len(config["services"]) == 2
        assert config["services"]["service1"] is services["service1"]
        assert config["services"]["service2"] is services["service2"]

    def test_build_config(self) -> None:
        """Test build_config utility."""
        result = build_config(
            key1="value1",
            key2=42,
            key3=True,
        )

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["key1"] == "value1"
        assert config["key2"] == 42
        assert config["key3"] is True

    def test_create_factory(self) -> None:
        """Test create_factory utility."""

        def test_func() -> str:
            return "test"

        factory = create_factory(test_func)

        assert isinstance(factory, FlextFactory)
        result = factory.create()
        assert result.is_success
        assert result.data == "test"

    def test_create_singleton_factory(self) -> None:
        """Test create_singleton_factory utility."""

        def test_func() -> str:
            return "singleton"

        factory = create_singleton_factory(test_func)

        assert isinstance(factory, FlextSingletonFactory)
        result = factory.get_instance()
        assert result.is_success
        assert result.data == "singleton"


@pytest.mark.integration
class TestBuildersIntegration:
    """Integration tests combining multiple builder patterns."""

    def test_complete_service_setup(self) -> None:
        """Test complete service configuration setup."""
        # Create some test services
        database_service = object()

        def cache_factory() -> str:
            return "cache_instance"

        def logger_factory() -> str:
            return "logger_instance"

        # Build complete service configuration
        result = (
            FlextServiceBuilder()
            .add_service("database", database_service)
            .add_factory("cache", cache_factory, singleton=True)
            .add_factory("logger", logger_factory, singleton=False)
            .add_services(
                config_service=object(),
                metrics_service=object(),
            )
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None

        # Verify all services are configured
        assert len(config["services"]) == 3  # database + 2 from add_services
        assert len(config["factories"]) == 2  # cache + logger
        assert "cache" in config["singletons"]
        assert "logger" not in config["singletons"]

    def test_config_with_validation(self) -> None:
        """Test configuration building with validation."""
        # Build configuration with required and optional values
        result = (
            FlextConfigBuilder()
            .set_required("database_url", "postgresql://localhost:5432/db")
            .set_required("api_key", "secret-key")
            .set_default("port", 8080)
            .set_default("debug", False)  # noqa: FBT003
            .set_if_not_none("cache_ttl", 3600)
            .set_if_not_none("unused_setting", None)
            .merge(
                {
                    "additional_setting": "from_merge",
                    "port": 9000,  # Override default
                },
            )
            .build()
        )

        assert result.is_success
        config = result.data
        assert config is not None

        # Verify all expected values
        assert config["database_url"] == "postgresql://localhost:5432/db"
        assert config["api_key"] == "secret-key"
        assert config["port"] == 9000  # Overridden by merge
        assert config["debug"] is False
        assert config["cache_ttl"] == 3600
        assert "unused_setting" not in config
        assert config["additional_setting"] == "from_merge"
