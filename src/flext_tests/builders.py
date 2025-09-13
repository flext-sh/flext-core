"""Test builders using Builder pattern with pytest integration.

Provides fluent builders for creating complex test objects with
pytest-mock, pytest-httpx, and other testing capabilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from unittest.mock import Mock

from pytest_mock import MockerFixture

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextResult,
    FlextTypes,
)


class FlextTestsBuilders:
    """Unified test builders for FLEXT ecosystem.

    Consolidates all builder patterns into a single class interface.
    Implements Builder pattern with fluent interface for creating
    complex test scenarios with proper setup and teardown.
    """

    # === Result Builder ===

    @classmethod
    def result(cls) -> FlextTestsBuilders.ResultBuilder:
        """Create a result builder."""
        return cls.ResultBuilder()

    class ResultBuilder:
        """Builder for FlextResult objects with various scenarios."""

        def __init__(self) -> None:
            """Initialize result builder."""
            self._data: object = None
            self._error: str | None = None
            self._error_code: str | None = None
            self._error_data: FlextTypes.Core.Dict | None = None
            self._is_success: bool = True

        def with_success_data(self, data: object) -> FlextTestsBuilders.ResultBuilder:
            """Set successful result data."""
            self._data = data
            self._is_success = True
            return self

        def with_failure(
            self,
            error: str,
            error_code: str | None = None,
            error_data: FlextTypes.Core.Dict | None = None,
        ) -> FlextTestsBuilders.ResultBuilder:
            """Set failure result with error details."""
            self._error = error
            self._error_code = error_code
            self._error_data = error_data
            self._is_success = False
            return self

        def build(self) -> FlextResult[object]:
            """Build the FlextResult object."""
            if self._is_success:
                return FlextResult[object].ok(self._data)

            return FlextResult[object].fail(
                self._error or "Test error",
                error_code=self._error_code,
                error_data=self._error_data,
            )

    # === Container Builder ===

    @classmethod
    def container(cls) -> FlextTestsBuilders.ContainerBuilder:
        """Create a container builder."""
        return cls.ContainerBuilder()

    class ContainerBuilder:
        """Builder for FlextContainer with pre-configured services."""

        def __init__(self) -> None:
            """Initialize container builder."""
            self._container = FlextContainer()
            self._services: FlextTypes.Core.Dict = {}
            self._factories: dict[str, Callable[[], object]] = {}

        def with_service(
            self, name: str, service: object
        ) -> FlextTestsBuilders.ContainerBuilder:
            """Add a service to the container."""
            self._services[name] = service
            return self

        def with_factory(
            self, name: str, factory: Callable[[], object]
        ) -> FlextTestsBuilders.ContainerBuilder:
            """Add a factory to the container."""
            self._factories[name] = factory
            return self

        def with_database_service(self) -> FlextTestsBuilders.ContainerBuilder:
            """Add a mock database service."""
            return self.with_service(
                "database",
                {
                    "host": "localhost",
                    "port": 5432,
                    "name": "test_db",
                    "connected": True,
                },
            )

        def with_cache_service(self) -> FlextTestsBuilders.ContainerBuilder:
            """Add a mock cache service."""
            return self.with_service(
                "cache",
                {
                    "host": "localhost",
                    "port": 6379,
                    "ttl": 3600,
                    "connected": True,
                },
            )

        def with_logger_service(self) -> FlextTestsBuilders.ContainerBuilder:
            """Add a mock logger service."""
            return self.with_service(
                "logger",
                {
                    "level": "DEBUG",
                    "format": "json",
                    "handlers": ["console"],
                },
            )

        def build(self) -> FlextContainer:
            """Build the container with all configured services."""
            for name, service in self._services.items():
                result = self._container.register(name, service)
                if result.is_failure:
                    msg = f"Failed to register service {name}: {result.error}"
                    raise RuntimeError(msg)

            for name, factory in self._factories.items():
                result = self._container.register_factory(name, factory)
                if result.is_failure:
                    msg = f"Failed to register factory {name}: {result.error}"
                    raise RuntimeError(msg)

            return self._container

    # === Field Builder ===

    @classmethod
    def field(cls, field_type: str) -> FlextTestsBuilders.FieldBuilder:
        """Create a field builder."""
        return cls.FieldBuilder(field_type)

    class FieldBuilder:
        """Builder for FlextFields objects with various configurations."""

        def __init__(self, field_type: str) -> None:
            """Initialize field builder."""
            self._field_type = field_type
            self._field_id = f"test_{field_type}_field"
            self._field_name = f"test_{field_type}_field"
            self._config: FlextTypes.Core.Dict = {}

        def with_id(self, field_id: str) -> FlextTestsBuilders.FieldBuilder:
            """Set field ID."""
            self._field_id = field_id
            return self

        def with_name(self, field_name: str) -> FlextTestsBuilders.FieldBuilder:
            """Set field name."""
            self._field_name = field_name
            return self

        def with_validation(self, **rules: object) -> FlextTestsBuilders.FieldBuilder:
            """Add validation rules."""
            # Convert rules to JsonValue compatible format
            for key, value in rules.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    self._config[key] = value
                else:
                    self._config[key] = str(value)
            return self

        def with_string_constraints(
            self,
            min_length: int | None = None,
            max_length: int | None = None,
            pattern: str | None = None,
        ) -> FlextTestsBuilders.FieldBuilder:
            """Add string-specific constraints."""
            if min_length is not None:
                self._config["min_length"] = min_length
            if max_length is not None:
                self._config["max_length"] = max_length
            if pattern is not None:
                self._config["pattern"] = pattern
            return self

        def with_numeric_constraints(
            self,
            min_value: float | None = None,
            max_value: float | None = None,
        ) -> FlextTestsBuilders.FieldBuilder:
            """Add numeric constraints."""
            if min_value is not None:
                self._config["min_value"] = min_value
            if max_value is not None:
                self._config["max_value"] = max_value
            return self

        def required(
            self, *, is_required: bool = True
        ) -> FlextTestsBuilders.FieldBuilder:
            """Set field required status."""
            self._config["required"] = is_required
            return self

        def with_default(
            self, default_value: object
        ) -> FlextTestsBuilders.FieldBuilder:
            """Set default value."""
            # Convert default value to JsonValue format
            if isinstance(default_value, (str, int, float, bool, type(None))):
                self._config["default_value"] = default_value
            else:
                self._config["default_value"] = str(default_value)
            return self

        def build(self) -> object:
            """Build the field object."""
            # Create field directly without non-existent Factory
            return {
                "type": self._field_type,
                "name": self._field_name,
                **self._config,
            }

    # === Config Builder ===

    @classmethod
    def config(cls) -> FlextTestsBuilders.ConfigBuilder:
        """Create a config builder."""
        return cls.ConfigBuilder()

    class ConfigBuilder:
        """Builder for FlextConfig objects with various settings."""

        def __init__(self) -> None:
            """Initialize config builder."""
            self._config_data: FlextTypes.Core.Dict = {}

        def with_debug(self, *, debug: bool = True) -> FlextTestsBuilders.ConfigBuilder:
            """Set debug mode."""
            self._config_data["debug"] = debug
            return self

        def with_log_level(
            self, level: str = "INFO"
        ) -> FlextTestsBuilders.ConfigBuilder:
            """Set log level."""
            self._config_data["log_level"] = level
            return self

        def with_environment(
            self, env: str = "test"
        ) -> FlextTestsBuilders.ConfigBuilder:
            """Set environment."""
            self._config_data["environment"] = env
            return self

        def with_database_url(self, url: str) -> FlextTestsBuilders.ConfigBuilder:
            """Set database URL."""
            self._config_data["database_url"] = url
            return self

        def with_custom_setting(
            self, key: str, value: object
        ) -> FlextTestsBuilders.ConfigBuilder:
            """Add custom configuration setting."""
            # Convert value to JsonValue format
            if isinstance(value, (str, int, float, bool, type(None))):
                self._config_data[key] = value
            else:
                self._config_data[key] = str(value)
            return self

        def build(self) -> FlextConfig:
            """Build the configuration object."""
            # Extract known fields with proper type casting
            # Get environment with validation
            env_value = str(self._config_data.get("environment", "test"))
            valid_envs = ["development", "staging", "production", "test", "local"]
            if env_value not in valid_envs:
                env_value = "test"  # Default fallback

            # Use FlextConfig.create() method for proper configuration creation
            config_data = {
                "log_level": str(self._config_data.get("log_level", "INFO")),
                "environment": env_value,
                "debug": bool(self._config_data.get("debug", False)),
            }

            # Create configuration using the factory method
            result = FlextConfig.create(constants=config_data)
            if result.is_success:
                return result.unwrap()

            # Fallback to default config if creation fails
            return FlextConfig()

    # === Mock Builder ===

    @classmethod
    def mock(cls, mocker: MockerFixture) -> FlextTestsBuilders.MockBuilder:
        """Create a mock builder."""
        return cls.MockBuilder(mocker)

    class MockBuilder:
        """Builder for mock objects with pytest-mock integration."""

        def __init__(self, mocker: MockerFixture) -> None:
            """Initialize mock builder."""
            self._mocker = mocker
            self._mock = Mock()
            self._return_values: FlextTypes.Core.List = []
            self._side_effects: FlextTypes.Core.List = []

        def returns(self, value: object) -> FlextTestsBuilders.MockBuilder:
            """Set return value."""
            self._return_values.append(value)
            return self

        def returns_result_success(
            self, data: object
        ) -> FlextTestsBuilders.MockBuilder:
            """Return successful FlextResult."""
            result = FlextResult[object].ok(data)
            return self.returns(result)

        def returns_result_failure(self, error: str) -> FlextTestsBuilders.MockBuilder:
            """Return failed FlextResult."""
            result = FlextResult[object].fail(error)
            return self.returns(result)

        def raises(self, exception: Exception) -> FlextTestsBuilders.MockBuilder:
            """Set side effect to raise exception."""
            self._side_effects.append(exception)
            return self

        def build(self) -> Mock:
            """Build the mock object."""
            if len(self._return_values) == 1:
                self._mock.return_value = self._return_values[0]
            elif len(self._return_values) > 1:
                self._mock.side_effect = self._return_values

            if self._side_effects:
                if self._return_values:
                    # Combine returns and exceptions
                    effects = self._return_values + self._side_effects
                    self._mock.side_effect = effects
                else:
                    self._mock.side_effect = self._side_effects[0]

            return self._mock

    # === File Builder ===

    @classmethod
    def file(cls) -> FlextTestsBuilders.FileBuilder:
        """Builder for test files with automatic cleanup."""
        return cls.FileBuilder()

    class FileBuilder:
        """Builder for test files with automatic cleanup."""

        def __init__(self) -> None:
            """Initialize file builder."""
            self._content = ""
            self._suffix = ".txt"
            self._encoding = "utf-8"
            self._mode = "w"

        def with_json_content(self, data: object) -> FlextTestsBuilders.FileBuilder:
            """Set JSON content."""
            self._content = json.dumps(data, indent=2)
            self._suffix = ".json"
            return self

        def with_text_content(self, content: str) -> FlextTestsBuilders.FileBuilder:
            """Set text content."""
            self._content = content
            self._suffix = ".txt"
            return self

        def with_config_content(self, config: object) -> FlextTestsBuilders.FileBuilder:
            """Set configuration file content."""
            return self.with_json_content(config).with_suffix(".config.json")

        def with_suffix(self, suffix: str) -> FlextTestsBuilders.FileBuilder:
            """Set file suffix."""
            self._suffix = suffix
            return self

        def build(self) -> str:
            """Build temporary file and return path."""
            with tempfile.NamedTemporaryFile(
                mode=self._mode,
                suffix=self._suffix,
                delete=False,
                encoding=self._encoding,
            ) as f:
                f.write(self._content)
                return f.name

    # === Convenience Methods ===

    @classmethod
    def success_result(cls, data: object = "test_data") -> FlextResult[object]:
        """Build a successful result quickly."""
        return cls.result().with_success_data(data).build()

    @classmethod
    def failure_result(cls, error: str = "test_error") -> FlextResult[object]:
        """Build a failure result quickly."""
        return cls.result().with_failure(error).build()

    @classmethod
    def test_container(cls) -> FlextContainer:
        """Build a container with common test services."""
        return (
            cls.container()
            .with_database_service()
            .with_cache_service()
            .with_logger_service()
            .build()
        )

    @classmethod
    def string_field(
        cls,
        field_id: str = "test_string",
        *,
        required: bool = True,
        **constraints: object,
    ) -> object:  # Return object to avoid field type import issues
        """Build a string field quickly."""
        builder = cls.field("string").with_id(field_id).required(is_required=required)

        if constraints:
            builder = builder.with_validation(**constraints)

        return builder.build()


# Export only the unified class
__all__ = [
    "FlextTestsBuilders",
]
