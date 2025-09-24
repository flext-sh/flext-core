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
from typing import Protocol, cast

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextResult,
    FlextTypes,
)


class FlextTestsBuilders:
    """Unified test builders for FLEXT ecosystem.

    Single responsibility class implementing Builder pattern with fluent interface
    for creating complex test scenarios with proper setup and teardown.
    Follows FLEXT patterns with single unified class and nested helper classes.
    """

    class _TestCallableProtocol(Protocol):
        """Protocol for test callable objects to avoid explicit object."""

        def __call__(self, *args: object, **kwargs: object) -> object:
            """Call signature for test functions."""
            ...

    def __init__(self) -> None:
        """Initialize unified test builders service."""
        try:
            self._container: FlextContainer | None = FlextContainer.get_global()
        except Exception:
            self._container = None

        # Dynamic container attributes (set by methods when needed)
        self._container_services: FlextTypes.Core.Dict = {}
        self._container_factories: dict[str, Callable[[], object]] = {}

        # Attributes for deprecated builder patterns
        self._building_container: bool = False

        # Deprecated result builder attributes
        self._result_success: bool = True
        self._result_data: object = None
        self._result_error: str = ""
        self._result_error_code: str | None = None
        self._result_error_data: FlextTypes.Core.Dict | None = None

    # === Result Builder Methods (Unified) ===

    def create_success_result(self, data: object = "test_data") -> FlextResult[object]:
        """Create a successful FlextResult for testing.

        Returns:
            FlextResult[object]: Success result with test data

        """
        return FlextResult[object].ok(data)

    def create_failure_result(
        self,
        error: str = "test_error",
        error_code: str | None = None,
        error_data: FlextTypes.Core.Dict | None = None,
    ) -> FlextResult[object]:
        """Create a failed FlextResult for testing.

        Returns:
            FlextResult[object]: Failure result with test error

        """
        return FlextResult[object].fail(
            error,
            error_code=error_code,
            error_data=error_data,
        )

    # === Container Builder Methods (Unified) ===

    def create_test_container(self: object) -> FlextContainer:
        """Create a FlextContainer with common test services.

        Returns:
            FlextContainer: Container with registered test services

        Raises:
            RuntimeError: If service registration fails

        """
        container = FlextContainer()

        # Register common test services
        services = {
            "database": {
                "host": FlextConstants.Platform.DEFAULT_HOST,
                "port": FlextConstants.Platform.POSTGRES_DEFAULT_PORT,
                "name": "test_db",
                "connected": True,
            },
            "cache": {
                "host": FlextConstants.Platform.DEFAULT_HOST,
                "port": FlextConstants.Platform.REDIS_DEFAULT_PORT,
                "ttl": FlextConstants.Performance.DEFAULT_TTL_SECONDS,
                "connected": True,
            },
            "logger": {
                "level": FlextConstants.Config.LogLevel.DEBUG,
                "format": "json",
                "handlers": ["console"],
            },
        }

        for name, service in services.items():
            result = container.register(name, service)
            if result.is_failure:
                msg = f"Failed to register service {name}: {result.error}"
                raise RuntimeError(msg)

        return container

    def create_container_with_service(
        self,
        name: str,
        service: object,
    ) -> FlextContainer:
        """Create a container with a specific service.

        Returns:
            FlextContainer: Container with registered service

        Raises:
            RuntimeError: If service registration fails

        """
        container = FlextContainer()
        result = container.register(name, service)
        if result.is_failure:
            msg = f"Failed to register service {name}: {result.error}"
            raise RuntimeError(msg)
        return container

    # === Config Builder Methods (Unified) ===

    def create_test_config(
        self,
        *,
        debug: bool = True,
        log_level: str = FlextConstants.Config.LogLevel.INFO,
        environment: str = FlextConstants.Environment.ConfigEnvironment.TESTING,
        database_url: str | None = None,
        **custom_settings: object,
    ) -> FlextConfig:
        """Create a FlextConfig for testing."""
        # Validate environment
        valid_envs = list(FlextConstants.Config.ENVIRONMENTS)
        if environment not in valid_envs:
            environment = FlextConstants.Environment.ConfigEnvironment.TESTING

        config_data = {
            "log_level": log_level,
            "environment": environment,
            "debug": debug,
        }

        if database_url:
            config_data["database_url"] = database_url

        # Add custom settings
        for key, value in custom_settings.items():
            if isinstance(value, (str, bool)):
                config_data[key] = value
            elif isinstance(value, (int, float, type(None))):
                config_data[key] = str(value) if value is not None else "None"
            else:
                config_data[key] = str(value)

        # Create configuration using factory method - returns FlextConfig directly
        return FlextConfig.create(**config_data)

    # === Field Builder Methods (Unified) ===

    def create_test_field(
        self,
        field_type: str,
        field_id: str | None = None,
        field_name: str | None = None,
        *,
        required: bool = True,
        default_value: object = None,
        **constraints: object,
    ) -> object:
        """Create a test field object."""
        if not field_id:
            field_id = f"test_{field_type}_field"
        if not field_name:
            field_name = field_id

        config: FlextTypes.Core.Dict = {
            "required": required,
        }

        # Add constraints
        for key, value in constraints.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                config[key] = value
            else:
                config[key] = str(value)

        # Add default value if provided
        if default_value is not None:
            if isinstance(default_value, (str, int, float, bool, type(None))):
                config["default_value"] = default_value
            else:
                config["default_value"] = str(default_value)

        return {
            "type": field_type,
            "name": field_name,
            **config,
        }

    def create_string_field(
        self,
        field_id: str = "test_string",
        *,
        required: bool = True,
        **constraints: object,
    ) -> object:
        """Create a string field for testing."""
        return self.create_test_field(
            "string",
            field_id,
            None,
            required=required,
            **constraints,
        )

    # === File Builder Methods (Unified) ===

    def create_temp_file(
        self,
        content: str = "",
        suffix: str = FlextConstants.Platform.EXT_TXT,
        encoding: str = "utf-8",
    ) -> str:
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding=encoding,
        ) as f:
            f.write(content)
            return f.name

    def create_json_file(self, data: object, suffix: str = ".json") -> str:
        """Create a temporary JSON file for testing."""
        content = json.dumps(data, indent=2)
        return self.create_temp_file(content=content, suffix=suffix)

    def create_config_file(self, config: object) -> str:
        """Create a temporary configuration file for testing."""
        return self.create_json_file(config, suffix=".config.json")

    # === Test Double Methods (Unified) ===

    def create_test_callable(
        self,
        return_value: object = None,
        return_values: FlextTypes.Core.List | None = None,
        side_effect: BaseException | None = None,
        behavior: _TestCallableProtocol | None = None,
    ) -> _TestCallableProtocol:
        """Create a test callable object with direct implementation."""
        if return_value is not None and return_values is None:
            return_values = [return_value]

        side_effects = [side_effect] if side_effect else []
        call_count = 0

        def callable_impl(*args: object, **kwargs: object) -> object:
            """Test callable implementation."""
            nonlocal call_count

            if behavior:
                return behavior(*args, **kwargs)

            if side_effects:
                raise side_effects[0]

            if return_values:
                if len(return_values) == 1:
                    return return_values[0]
                if len(return_values) > 1:
                    # Cycle through return values
                    result_item: object = return_values[call_count % len(return_values)]
                    call_count += 1
                    return result_item

            return None

        return callable_impl

    def create_result_returning_callable(
        self,
        data: object = None,
        error: str | None = None,
    ) -> _TestCallableProtocol:
        """Create a callable that returns FlextResult objects."""
        if error:
            result_value: FlextResult[object] = FlextResult[object].fail(error)
        else:
            result_value = FlextResult[object].ok(data)

        return self.create_test_callable(return_value=result_value)

    # === Convenience Factory Methods ===

    @classmethod
    def create_instance(cls) -> FlextTestsBuilders:
        """Create a new instance of the unified builders service."""
        return cls()

    # DEPRECATED: Legacy methods for backward compatibility
    @classmethod
    def result(cls) -> FlextTestsBuilders:
        """Create a result builder (deprecated - use create_success_result/create_failure_result)."""
        instance = cls()
        # Ensure result building mode is enabled by setting result-specific attributes
        instance._result_success = True
        instance._result_data = None
        # Explicitly disable container building mode
        instance._building_container = False
        return instance

    def with_success_data(self, data: object) -> FlextTestsBuilders:
        """Set successful result data (deprecated method)."""
        self._result_data = data
        self._result_success = True
        return self

    def with_failure(
        self,
        error: str,
        error_code: str | None = None,
        error_data: FlextTypes.Core.Dict | None = None,
    ) -> FlextTestsBuilders:
        """Set failure result with error details (deprecated method)."""
        self._result_error = error
        self._result_error_code = error_code
        self._result_error_data = error_data
        self._result_success = False
        return self

    def build(self: object) -> FlextContainer | FlextResult[object]:
        """Build object (deprecated method - supports both container and result building)."""
        # Check for result building first (explicit result building mode)
        if (
            hasattr(self, "_result_success")
            or hasattr(self, "_result_error")
            or hasattr(self, "_result_data")
        ):
            # FlextResult building pattern (deprecated)
            if getattr(self, "_result_success", True):
                result_data = getattr(self, "_result_data", "test_data")
                data: FlextTypes.Core.Dict = (
                    cast("FlextTypes.Core.Dict", result_data)
                    if isinstance(result_data, dict)
                    else {"data": result_data}
                )
                return FlextResult[object].ok(data)

            error = getattr(self, "_result_error", "Test error")
            error_code = getattr(self, "_result_error_code", None)
            error_data_raw = getattr(self, "_result_error_data", None)
            error_data: FlextTypes.Core.Dict = (
                cast("FlextTypes.Core.Dict", error_data_raw)
                if isinstance(error_data_raw, dict)
                else {}
            )
            return FlextResult[object].fail(
                error,
                error_code=error_code,
                error_data=error_data,
            )

        # Container building pattern (deprecated)
        if (
            getattr(self, "_building_container", False)
            or hasattr(self, "_container_services")
            or hasattr(self, "_container_factories")
        ):
            container = FlextContainer()

            # Register services if any
            if hasattr(self, "_container_services"):
                container_services = getattr(self, "_container_services", {})
                for name, service in container_services.items():
                    register_result: FlextResult[None] = container.register(
                        name, service
                    )
                    if register_result.is_failure:
                        msg = f"Failed to register service {name}: {register_result.error}"
                        raise RuntimeError(msg)

            # Register factories if any
            if hasattr(self, "_container_factories"):
                container_factories = getattr(self, "_container_factories", {})
                for name, factory in container_factories.items():
                    factory_result: FlextResult[None] = container.register_factory(
                        name, factory
                    )
                    if factory_result.is_failure:
                        msg = (
                            f"Failed to register factory {name}: {factory_result.error}"
                        )
                        raise RuntimeError(msg)

            return container

        # Default to successful result if no specific mode is detected
        return FlextResult[object].ok("test_data")

    @classmethod
    def container(cls: type[FlextTestsBuilders]) -> FlextTestsBuilders:
        """Create a container builder (deprecated - use create_test_container)."""
        instance = cls()
        instance._building_container = True  # Flag to indicate container building mode
        return instance

    def with_service(self, name: str, service: object) -> FlextTestsBuilders:
        """Add a service to the container (deprecated method)."""
        if not hasattr(self, "_container_services"):
            self._container_services = {}
        self._container_services[name] = service
        return self

    def with_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextTestsBuilders:
        """Add a factory to the container (deprecated method)."""
        if not hasattr(self, "_container_factories"):
            self._container_factories = {}
        self._container_factories[name] = factory
        return self

    def with_database_service(self) -> FlextTestsBuilders:
        """Add a database service (deprecated method)."""
        return self.with_service(
            "database",
            {
                "host": FlextConstants.Platform.DEFAULT_HOST,
                "port": 5432,
                "name": "test_db",
                "connected": True,
            },
        )

    def with_cache_service(self) -> FlextTestsBuilders:
        """Add a cache service (deprecated method)."""
        return self.with_service(
            "cache",
            {
                "host": FlextConstants.Platform.DEFAULT_HOST,
                "port": 6379,
                "ttl": 3600,
                "connected": True,
            },
        )

    def with_logger_service(self) -> FlextTestsBuilders:
        """Add a logger service (deprecated method)."""
        return self.with_service(
            "logger",
            {
                "level": FlextConstants.Config.LogLevel.DEBUG,
                "format": "json",
                "handlers": ["console"],
            },
        )

    # Legacy class references for backward compatibility
    @classmethod
    def success_result(cls, data: object = "test_data") -> FlextResult[object]:
        """Build a successful result quickly (deprecated - use create_success_result)."""
        return cls().create_success_result(data)

    @classmethod
    def failure_result(cls, error: str = "test_error") -> FlextResult[object]:
        """Build a failure result quickly (deprecated - use create_failure_result)."""
        return cls().create_failure_result(error)

    @classmethod
    def test_container(cls: type[FlextTestsBuilders]) -> FlextContainer:
        """Build a container with common test services (deprecated - use create_test_container)."""
        return cls().create_test_container()

    @classmethod
    def string_field(
        cls,
        field_id: str = "test_string",
        *,
        required: bool = True,
        **constraints: object,
    ) -> object:
        """Build a string field quickly (deprecated - use create_string_field)."""
        return cls().create_string_field(
            field_id=field_id,
            required=required,
            **constraints,
        )


# Export only the unified class
__all__ = [
    "FlextTestsBuilders",
]
