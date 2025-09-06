"""Dependency injection container for service management.

This module provides FlextContainer, a type-safe DI container with factory
patterns and global singleton access using FlextResult error handling.
"""

from __future__ import annotations

import inspect
from collections import UserString
from collections.abc import Callable
from datetime import datetime
from typing import NotRequired, TypedDict, Unpack, cast, override
from zoneinfo import ZoneInfo

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T
from flext_core.utilities import FlextUtilities


class FlextContainer:
    """Dependency injection container with type-safe service management.

    Manages service lifecycles through registration, factory patterns, and
    type-safe retrieval. Supports singleton and instance patterns.
    """

    # Class-level global manager instance
    _global_manager: FlextContainer.GlobalManager | None = None

    # =========================================================================
    # NESTED CLASSES - Organized functionality following FLEXT patterns
    # =========================================================================

    class ServiceKey[T](UserString, FlextProtocols.Foundation.Validator[str]):
        """Typed service key for type-safe service resolution."""

        __slots__ = ()

        @property
        def name(self) -> str:
            """Get service key name."""
            return str(self)

        @classmethod
        def __class_getitem__(cls, _item: object) -> type[FlextContainer.ServiceKey[T]]:
            """Support generic subscription."""
            return cls

        def validate(self, data: str) -> object:
            """Validate service key name."""
            if not data or not data.strip():
                return FlextResult[str].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[str].ok(data.strip())

    class Commands:
        """Container operation commands."""

        class RegisterService:
            """Command to register a service instance."""

            # Python 3.13 Advanced: TypedDict for command parameters
            class RegisterServiceParams(TypedDict):
                """RegisterService command parameters."""

                service_name: NotRequired[str]
                service_instance: NotRequired[object]
                command_type: NotRequired[str]
                command_id: NotRequired[str]
                timestamp: NotRequired[datetime | None]
                user_id: NotRequired[str | None]
                correlation_id: NotRequired[str]

            def __init__(self, **params: Unpack[RegisterServiceParams]) -> None:
                """Initialize RegisterService command using Python 3.13 TypedDict."""
                # Apply defaults using TypedDict pattern
                self.service_name = params.get("service_name", "")
                self.service_instance = params.get("service_instance")
                self.command_type = params.get("command_type", "register_service")
                self.command_id = (
                    params.get("command_id")
                    or FlextUtilities.Generators.generate_uuid()
                )
                self.timestamp = params.get("timestamp") or datetime.now(
                    tz=ZoneInfo("UTC"),
                )
                self.user_id = params.get("user_id")
                self.correlation_id = (
                    params.get("correlation_id")
                    or FlextUtilities.Generators.generate_uuid()
                )

            @classmethod
            def create(
                cls,
                service_name: str,
                service_instance: object,
            ) -> FlextContainer.Commands.RegisterService:
                """Create command with default values."""
                return cls(
                    service_name=service_name,
                    service_instance=service_instance,
                    command_type="register_service",
                    command_id=FlextUtilities.Generators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextUtilities.Generators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate service registration command."""
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    )
                return FlextResult[None].ok(None)

        class RegisterFactory:
            """Command to register a service factory."""

            # Python 3.13 Advanced: TypedDict for command parameters
            class RegisterFactoryParams(TypedDict):
                """TypedDict for RegisterFactory command - ELIMINATES BOILERPLATE."""

                service_name: NotRequired[str]
                factory: NotRequired[object]
                command_type: NotRequired[str]
                command_id: NotRequired[str]
                timestamp: NotRequired[datetime | None]
                user_id: NotRequired[str | None]
                correlation_id: NotRequired[str]

            def __init__(self, **params: Unpack[RegisterFactoryParams]) -> None:
                """Initialize RegisterFactory command using Python 3.13 TypedDict."""
                self.service_name = params.get("service_name", "")
                self.factory = params.get("factory")
                self.command_type = params.get("command_type", "register_factory")
                self.command_id = (
                    params.get("command_id")
                    or FlextUtilities.Generators.generate_uuid()
                )
                self.timestamp = params.get("timestamp") or datetime.now(
                    tz=ZoneInfo("UTC"),
                )
                self.user_id = params.get("user_id")
                self.correlation_id = (
                    params.get("correlation_id")
                    or FlextUtilities.Generators.generate_uuid()
                )

            @classmethod
            def create(
                cls,
                service_name: str,
                factory: object,
            ) -> FlextContainer.Commands.RegisterFactory:
                """Create command with default values."""
                return cls(
                    service_name=service_name,
                    factory=factory,
                    command_type="register_factory",
                    command_id=FlextUtilities.Generators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextUtilities.Generators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate factory registration command."""
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    )
                if not callable(self.factory):
                    return FlextResult[None].fail("Factory must be callable")
                return FlextResult[None].ok(None)

        class UnregisterService:
            """Command to unregister a service."""

            def __init__(
                self,
                service_name: str = "",
                command_type: str = "unregister_service",
                command_id: str = "",
                timestamp: datetime | None = None,
                user_id: str | None = None,
                correlation_id: str = "",
            ) -> None:
                """Initialize UnregisterService command."""
                self.service_name = service_name
                self.command_type = command_type
                self.command_id = (
                    command_id or FlextUtilities.Generators.generate_uuid()
                )
                self.timestamp = timestamp or datetime.now(tz=ZoneInfo("UTC"))
                self.user_id = user_id
                self.correlation_id = (
                    correlation_id or FlextUtilities.Generators.generate_uuid()
                )

            @classmethod
            def create(
                cls,
                service_name: str,
            ) -> FlextContainer.Commands.UnregisterService:
                """Create command with default values."""
                return cls(
                    service_name=service_name,
                    command_type="unregister_service",
                    command_id=FlextUtilities.Generators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextUtilities.Generators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate service unregistration command."""
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    )
                return FlextResult[None].ok(None)

    class Queries:
        """Container query definitions following CQRS pattern.

        Nested class containing all query definitions for container operations
        following the Command Query Responsibility Segregation pattern.
        """

        class GetService:
            """Query to retrieve a service."""

            def __init__(
                self,
                service_name: str = "",
                expected_type: str | None = None,
                query_type: str = "get_service",
                query_id: str = "",
                page_size: int = FlextConstants.Defaults.PAGE_SIZE,
                page_number: int = 1,
                sort_by: str | None = None,
                sort_order: str = "asc",
            ) -> None:
                """Initialize GetService query."""
                self.service_name = service_name
                self.expected_type = expected_type
                self.query_type = query_type
                self.query_id = query_id
                self.page_size = page_size
                self.page_number = page_number
                self.sort_by = sort_by
                self.sort_order = sort_order

            @classmethod
            def create(
                cls,
                service_name: str,
                expected_type: str | None = None,
            ) -> FlextContainer.Queries.GetService:
                """Create query with default values."""
                return cls(
                    service_name=service_name,
                    expected_type=expected_type,
                    query_type="get_service",
                    query_id="",
                    page_size=100,
                    page_number=1,
                    sort_by=None,
                    sort_order="asc",
                )

            def validate_query(self) -> FlextResult[None]:
                """Validate service retrieval query."""
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    )
                return FlextResult[None].ok(None)

        class ListServices:
            """Query to list all services."""

            def __init__(
                self,
                *,
                include_factories: bool = True,
                service_type_filter: str | None = None,
                query_type: str = "list_services",
                query_id: str = "",
                page_size: int = FlextConstants.Defaults.PAGE_SIZE,
                page_number: int = 1,
                sort_by: str | None = None,
                sort_order: str = "asc",
            ) -> None:
                """Initialize ListServices query."""
                self.include_factories = include_factories
                self.service_type_filter = service_type_filter
                self.query_type = query_type
                self.query_id = query_id
                self.page_size = page_size
                self.page_number = page_number
                self.sort_by = sort_by
                self.sort_order = sort_order

            @classmethod
            def create(
                cls,
                *,
                include_factories: bool = True,
                service_type_filter: str | None = None,
            ) -> FlextContainer.Queries.ListServices:
                """Create query with default values."""
                return cls(
                    include_factories=include_factories,
                    service_type_filter=service_type_filter,
                    query_type="list_services",
                    query_id="",
                    page_size=100,
                    page_number=1,
                    sort_by=None,
                    sort_order="asc",
                )

    class ServiceRegistrar:
        """Service registration component implementing Single Responsibility Principle.

        This nested class handles all service and factory registration operations with validation
        and error handling. It's separated from service retrieval to follow SRP and
        provide clear separation of concerns in the dependency injection system.

        """

        def __init__(self) -> None:
            """Initialize service registrar with empty registry."""
            self._services: FlextTypes.Service.ServiceDict = {}
            self._factories: FlextTypes.Service.FactoryDict = {}

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name."""
            if not FlextContainer.flext_validate_service_name(name):
                return FlextResult[str].fail(FlextConstants.Messages.SERVICE_NAME_EMPTY)
            return FlextResult[str].ok(name)

        def register_service(
            self,
            name: str,
            service: object,
        ) -> FlextResult[None]:
            """Register service instance."""
            # Fast path: simple validation without FlextResult overhead
            if not name or not name.strip():
                return FlextResult[None].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name = name.strip()

            # Store service in registry
            self._services[validated_name] = service
            return FlextResult[None].ok(None)

        def register_factory(
            self,
            name: str,
            factory: object,
        ) -> FlextResult[None]:
            """Register service factory."""
            validation_result = self._validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name: str = validation_result.value

            # Runtime validation for callability
            if not callable(factory):
                return FlextResult[None].fail("Factory must be callable")

            # Verify factory signature to ensure it can be called without parameters
            try:
                sig = inspect.signature(factory)
                required_params = sum(
                    1
                    for p in sig.parameters.values()
                    if p.default == p.empty
                    and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
                )
                if required_params > 0:
                    msg = (
                        f"Factory must be callable without parameters, "
                        f"but requires {required_params} parameter(s)"
                    )
                    return FlextResult[None].fail(msg)
            except (ValueError, TypeError, OSError) as e:
                return FlextResult[None].fail(
                    f"Could not inspect factory signature: {e}",
                )

            if validated_name in self._services:
                del self._services[validated_name]

            # Safe assignment after signature verification
            factory_callable = cast("Callable[[], object]", factory)
            self._factories[validated_name] = factory_callable
            return FlextResult[None].ok(None)

        def unregister_service(self, name: str) -> FlextResult[None]:
            """Unregister a service."""
            validation_result = self._validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name: str = validation_result.value

            if validated_name in self._services:
                del self._services[validated_name]
                return FlextResult[None].ok(None)

            if validated_name in self._factories:
                del self._factories[validated_name]
                return FlextResult[None].ok(None)

            return FlextResult[None].fail(f"Service '{validated_name}' not found")

        def clear_all(self) -> FlextResult[None]:
            """Clear all registered services and factories."""
            self._services.clear()
            self._factories.clear()
            return FlextResult[None].ok(None)

        def get_service_names(self) -> list[str]:
            """Get all registered service names."""
            return list(self._services.keys()) + list(self._factories.keys())

        def get_service_count(self) -> int:
            """Get total service count."""
            return len(self._services) + len(self._factories)

        def has_service(self, name: str) -> bool:
            """Check if a service exists."""
            return name in self._services or name in self._factories

        def get_services_dict(self) -> FlextTypes.Service.ServiceDict:
            """Get services dictionary (internal use)."""
            return self._services

        def get_factories_dict(self) -> FlextTypes.Service.FactoryDict:
            """Get factories dictionary (internal use)."""
            return self._factories

    class ServiceRetriever:
        """Service retrieval component implementing single responsibility principle.

        This nested class handles service retrieval operations including service
        resolution, factory instantiation, and service information queries.
        """

        def __init__(
            self,
            services: FlextTypes.Service.ServiceDict,
            factories: FlextTypes.Service.FactoryDict,
        ) -> None:
            """Initialize service retriever with references.

            Args:
                services: Services dictionary reference.
                factories: Factories dictionary reference.

            """
            super().__init__()
            self._services = services
            self._factories = factories

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name."""
            if not FlextContainer.flext_validate_service_name(name):
                return FlextResult[str].fail(FlextConstants.Messages.SERVICE_NAME_EMPTY)
            return FlextResult[str].ok(name)

        def get_service(self, name: str) -> FlextResult[object]:
            """Retrieve a registered service - Performance optimized."""
            if not name or not name.strip():
                return FlextResult[object].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name = name.strip()

            # Check direct service registration first (the most common case)
            if validated_name in self._services:
                return FlextResult[object].ok(self._services[validated_name])

            # Check factory registration
            if validated_name in self._factories:
                try:
                    factory = self._factories[validated_name]
                    service = factory()

                    # Cache the factory result as a service for singleton behavior
                    self._services[validated_name] = service
                    # Remove from factories since it's now cached as a service
                    del self._factories[validated_name]

                    return FlextResult[object].ok(service)
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    FlextContainer._get_exception_class("FlextError"),
                ) as e:
                    return FlextResult[object].fail(
                        f"Factory for '{validated_name}' failed: {e!s}",
                    )

            return FlextResult[object].fail(f"Service '{validated_name}' not found")

        def get_service_info(self, name: str) -> FlextResult[dict[str, object]]:
            """Get detailed information about a registered service or factory."""
            if not name or not name.strip():
                return FlextResult[dict[str, object]].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name = name.strip()

            # Check if a service is registered as instance
            if validated_name in self._services:
                service = self._services[validated_name]
                service_class = service.__class__
                return FlextResult[dict[str, object]].ok(
                    {
                        "name": validated_name,
                        "type": "instance",
                        "class": service_class.__name__,
                        "module": service_class.__module__,
                    },
                )

            # Check if a service is registered as factory
            if validated_name in self._factories:
                factory = self._factories[validated_name]
                factory_name = getattr(factory, "__name__", str(factory))
                factory_module = getattr(factory, "__module__", "unknown")
                return FlextResult[dict[str, object]].ok(
                    {
                        "name": validated_name,
                        "type": "factory",
                        "factory": factory_name,
                        "module": factory_module,
                    },
                )

            return FlextResult[dict[str, object]].fail(
                f"Service '{validated_name}' not found",
            )

        def list_services(self) -> FlextTypes.Service.ServiceListDict:
            """List all services with their types."""
            services_info: FlextTypes.Service.ServiceListDict = {}

            for name in self._services:
                services_info[name] = "instance"

            for name in self._factories:
                services_info[name] = "factory"

            return services_info

    class GlobalManager:
        """Simple global container manager for singleton pattern.

        This nested class manages the global container instance following
        the singleton pattern for ecosystem-wide service sharing.
        """

        def __init__(self) -> None:
            """Initialize with default container."""
            self._container = FlextContainer()

        def get_container(self) -> FlextContainer:
            """Get the global container instance."""
            return self._container

        def set_container(self, container: FlextContainer) -> None:
            """Set the global container instance."""
            self._container = container

    # =========================================================================
    # CONTAINER IMPLEMENTATION - Main container functionality
    # =========================================================================

    def __init__(self, config: FlextTypes.Config.ConfigDict | None = None) -> None:
        """Initialize container with internal components and FlextTypes.Config support.

        Args:
            config: Optional configuration dictionary using FlextTypes.Config.

        """
        super().__init__()

        # Initialize container configuration with FlextTypes.Config
        self._config: FlextTypes.Config.ConfigDict = config or {
            "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
            "log_level": FlextConstants.Config.LogLevel.INFO.value,
            "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
            "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
            "max_services": 1000,
            "service_timeout": 30000,  # milliseconds
            "enable_auto_wire": True,
            "enable_factory_cache": True,
        }

        # SRP: Delegate to focused internal components
        self._registrar = self.ServiceRegistrar()

        # DIP: Retriever depends on registrar's data abstractions
        services_dict = self._registrar.get_services_dict()
        factories_dict = self._registrar.get_factories_dict()
        self._retriever = self.ServiceRetriever(services_dict, factories_dict)

        # Specialized configuration support
        self._database_config: FlextModels.DatabaseConfig | None = None
        self._security_config: FlextModels.SecurityConfig | None = None
        self._logging_config: FlextModels.LoggingConfig | None = None

        # Simplified command bus - can be extended later
        self._command_bus = None

    # =========================================================================
    # SPECIALIZED CONFIGURATION PROPERTIES
    # =========================================================================

    @property
    def database_config(self) -> FlextModels.DatabaseConfig | None:
        """Access database configuration if available."""
        return self._database_config

    @property
    def security_config(self) -> FlextModels.SecurityConfig | None:
        """Access security configuration if available."""
        return self._security_config

    @property
    def logging_config(self) -> FlextModels.LoggingConfig | None:
        """Access logging configuration if available."""
        return self._logging_config

    def configure_database(self, config: FlextModels.DatabaseConfig) -> None:
        """Configure database settings for this container."""
        self._database_config = config

    def configure_security(self, config: FlextModels.SecurityConfig) -> None:
        """Configure security settings for this container."""
        self._security_config = config

    def configure_logging(self, config: FlextModels.LoggingConfig) -> None:
        """Configure logging settings for this container."""
        self._logging_config = config

    # =========================================================================
    # CONFIGURATION MANAGEMENT WITH FlextTypes.Config - Massive Integration
    # =========================================================================

    def configure_container(
        self,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[None]:
        """Configure container with FlextTypes.Config and StrEnum validation."""
        try:
            # Validate environment using FlextConstants.Config.ConfigEnvironment
            if "environment" in config:
                env_value = config["environment"]
                valid_environments = [
                    e.value for e in FlextConstants.Config.ConfigEnvironment
                ]
                if env_value not in valid_environments:
                    return FlextResult[None].fail(
                        f"Invalid environment: {env_value}. Must be one of: {valid_environments}",
                    )

            # Validate log level using FlextConstants.Config.LogLevel
            if "log_level" in config:
                log_level = config["log_level"]
                valid_levels = [level.value for level in FlextConstants.Config.LogLevel]
                if log_level not in valid_levels:
                    return FlextResult[None].fail(
                        f"Invalid log_level: {log_level}. Must be one of: {valid_levels}",
                    )

            # Validate validation level using FlextConstants.Config.ValidationLevel
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_val_levels = [
                    v.value for v in FlextConstants.Config.ValidationLevel
                ]
                if val_level not in valid_val_levels:
                    return FlextResult[None].fail(
                        f"Invalid validation_level: {val_level}. Must be one of: {valid_val_levels}",
                    )

            # Validate config source using FlextConstants.Config.ConfigSource
            if "config_source" in config:
                source = config["config_source"]
                valid_sources = [s.value for s in FlextConstants.Config.ConfigSource]
                if source not in valid_sources:
                    return FlextResult[None].fail(
                        f"Invalid config_source: {source}. Must be one of: {valid_sources}",
                    )

            # Validate numeric constraints
            if "max_services" in config:
                max_services = config["max_services"]
                if not isinstance(max_services, int) or max_services < 1:
                    return FlextResult[None].fail(
                        "max_services must be a positive integer",
                    )

            if "service_timeout" in config:
                timeout = config["service_timeout"]
                if not isinstance(timeout, (int, float)) or timeout < 0:
                    return FlextResult[None].fail(
                        "service_timeout must be a non-negative number",
                    )

            # Update internal configuration
            self._config.update(config)

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Container configuration failed: {e}")

    def get_container_config(self) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current container configuration using FlextTypes.Config."""
        try:
            # Return copy to prevent external modification
            config_copy: FlextTypes.Config.ConfigDict = dict(self._config)
            return FlextResult[FlextTypes.Config.ConfigDict].ok(config_copy)
        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Config retrieval failed: {e}",
            )

    def create_environment_scoped_container(
        self,
        environment: FlextTypes.Config.Environment,
    ) -> FlextResult[FlextContainer]:
        """Create a new container scoped to a specific environment using FlextTypes.Config."""
        try:
            # Validate environment is a valid StrEnum value
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextContainer].fail(
                    f"Invalid environment: {environment}. Must be one of: {valid_environments}",
                )

            # Create environment-specific configuration
            env_config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "log_level": self._config.get(
                    "log_level",
                    FlextConstants.Config.LogLevel.INFO.value,
                ),
                "validation_level": (
                    FlextConstants.Config.ValidationLevel.STRICT.value
                    if environment
                    == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                    else FlextConstants.Config.ValidationLevel.NORMAL.value
                ),
                "config_source": FlextConstants.Config.ConfigSource.ENVIRONMENT.value,
                "max_services": self._config.get("max_services", 1000),
                "service_timeout": (
                    60000
                    if environment
                    == FlextConstants.Config.ConfigEnvironment.PRODUCTION.value
                    else 30000
                ),
                "enable_auto_wire": self._config.get("enable_auto_wire", True),
                "enable_factory_cache": self._config.get("enable_factory_cache", True),
            }

            # Create new container with environment-specific config
            scoped_container = FlextContainer(config=env_config)
            return FlextResult[FlextContainer].ok(scoped_container)

        except Exception as e:
            return FlextResult[FlextContainer].fail(
                f"Environment scoped container creation failed: {e}",
            )

    def get_configuration_summary(self) -> FlextResult[dict[str, object]]:
        """Get efficient configuration summary with FlextTypes.Config integration."""
        try:
            # Build summary with proper object typing
            summary: dict[str, object] = {
                "container_config": dict(self._config),
                "service_statistics": {
                    "total_services": self.get_service_count(),
                    "service_names": self.get_service_names(),
                    "max_services_limit": self._config.get("max_services", 1000),
                    "factory_cache_enabled": self._config.get(
                        "enable_factory_cache",
                        True,
                    ),
                },
                "environment_info": {
                    "current_environment": self._config.get("environment", "unknown"),
                    "log_level": self._config.get("log_level", "INFO"),
                    "validation_level": self._config.get("validation_level", "normal"),
                    "config_source": self._config.get("config_source", "environment"),
                },
                "performance_settings": {
                    "service_timeout_ms": self._config.get("service_timeout", 30000),
                    "auto_wire_enabled": self._config.get("enable_auto_wire", True),
                },
                "available_enum_values": {
                    "environments": [
                        e.value for e in FlextConstants.Config.ConfigEnvironment
                    ],
                    "log_levels": [
                        level.value for level in FlextConstants.Config.LogLevel
                    ],
                    "validation_levels": [
                        v.value for v in FlextConstants.Config.ValidationLevel
                    ],
                    "config_sources": [
                        s.value for s in FlextConstants.Config.ConfigSource
                    ],
                },
            }

            return FlextResult[dict[str, object]].ok(summary)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Configuration summary generation failed: {e}",
            )

    # -------------------------------------------------------------------------
    # Registration API - Simplified without command bus for now
    # -------------------------------------------------------------------------

    def register(self, name: str, service: object) -> FlextResult[None]:
        """Register a service instance."""
        return self._registrar.register_service(name, service)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register a service factory."""
        return self._registrar.register_factory(name, factory)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service."""
        return self._registrar.unregister_service(name)

    # -------------------------------------------------------------------------
    # Retrieval API - Simplified without command bus
    # -------------------------------------------------------------------------

    def get(self, name: str) -> FlextResult[object]:
        """Get a service by name."""
        return self._retriever.get_service(name)

    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type checking - Performance optimized."""
        result = self.get(name)
        if result.is_failure:
            return FlextResult[T].fail(result.error or "Service not found")

        service: object = result.value

        # Simple isinstance check instead of complex type guards
        if not isinstance(service, expected_type):
            actual_type = type(service).__name__
            return FlextResult[T].fail(
                f"Service '{name}' is {actual_type}, expected {expected_type.__name__}",
            )

        return FlextResult[T].ok(service)

    # -------------------------------------------------------------------------
    # Container management
    # -------------------------------------------------------------------------

    def clear(self) -> FlextResult[None]:
        """Clear all services."""
        return self._registrar.clear_all()

    def has(self, name: str) -> bool:
        """Check if a service exists."""
        return self._registrar.has_service(name)

    def list_services(self) -> FlextTypes.Service.ServiceListDict:
        """List all services."""
        return self._retriever.list_services()

    def get_service_names(self) -> list[str]:
        """Get service names."""
        return self._registrar.get_service_names()

    def get_service_count(self) -> int:
        """Get service count."""
        return self._registrar.get_service_count()

    @property
    def command_bus(self) -> object:
        """Access to the internal command bus for operations."""
        return self._command_bus

    def get_info(self, name: str) -> FlextResult[dict[str, object]]:
        """Return basic info about a registered service or factory."""
        try:
            if name in self._registrar.get_services_dict():
                service = self._registrar.get_services_dict()[name]
                service_info: dict[str, object] = {
                    "name": name,
                    "kind": "instance",
                    "type": "instance",
                    "class": type(service).__name__,
                }
                return FlextResult[dict[str, object]].ok(service_info)
            if name in self._registrar.get_factories_dict():
                factory = self._registrar.get_factories_dict()[name]
                factory_info: dict[str, object] = {
                    "name": name,
                    "kind": "factory",
                    "type": "factory",
                    "class": type(factory).__name__,
                }
                return FlextResult[dict[str, object]].ok(factory_info)
            return FlextResult[dict[str, object]].fail(f"Service '{name}' not found")
        except (KeyError, AttributeError, TypeError) as e:
            return FlextResult[dict[str, object]].fail(f"Info retrieval failed: {e}")

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[object]:
        """Get existing service or register-and-return a newly created one."""
        existing = self.get(name)
        if existing.is_success:
            return existing
        try:
            # Register factory and immediately resolve
            reg = self.register_factory(name, factory)
            if reg.is_failure:
                return FlextResult[object].fail(
                    reg.error or "Factory registration failed",
                )

            # Try to get the service
            service_result = self.get(name)
            if service_result.is_failure:
                error = service_result.error or ""
                if "Factory for" in error and "failed:" in error:
                    return FlextResult[object].fail(
                        f"Factory failed for service '{name}'",
                    )
                return service_result

            return service_result
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult[object].fail(f"get_or_create failed: {e}")

    def auto_wire(
        self,
        service_class: type[T],
        service_name: str | None = None,
    ) -> FlextResult[T]:
        """Auto-wire service dependencies and register the service."""
        try:
            # Use class name as default service name
            name = service_name or service_class.__name__

            # Get constructor signature
            sig = inspect.signature(service_class.__init__)
            args: list[object] = []
            kwargs: dict[str, object] = {}

            # Skip 'self' parameter
            params = list(sig.parameters.values())[1:]

            for param in params:
                if param.default is not inspect.Parameter.empty:
                    # Parameter has default, skip dependency resolution
                    continue

                # Try to resolve dependency from registered services
                dependency_result = self.get(param.name)
                if dependency_result.is_failure:
                    return FlextResult[T].fail(
                        f"Required dependency '{param.name}' not found "
                        f"for {service_class.__name__}",
                    )

                kwargs[param.name] = dependency_result.value

            # Instantiate service with resolved dependencies
            service_instance = service_class(*args, **kwargs)

            # Register the service
            register_result = self.register(name, service_instance)
            if register_result.is_failure:
                return FlextResult[T].fail(
                    f"Auto-wiring failed during registration: {register_result.error}",
                )

            return FlextResult[T].ok(service_instance)

        except (TypeError, ValueError, AttributeError, RuntimeError, OSError) as e:
            return FlextResult[T].fail(f"Auto-wiring failed: {e}")

    def batch_register(
        self,
        registrations: dict[str, object],
    ) -> FlextResult[list[str]]:
        """Register many services/factories atomically; roll back on failure."""
        # Snapshot current state for rollback
        services_snapshot = dict(self._registrar.get_services_dict())
        factories_snapshot = dict(self._registrar.get_factories_dict())
        registered_names: list[str] = []
        try:
            for key, value in registrations.items():
                if callable(value):
                    result = self.register_factory(key, value)
                else:
                    result = self.register(key, value)
                if result.is_failure:
                    # Rollback
                    self._registrar.get_services_dict().clear()
                    self._registrar.get_services_dict().update(services_snapshot)
                    self._registrar.get_factories_dict().clear()
                    self._registrar.get_factories_dict().update(factories_snapshot)
                    return FlextResult[list[str]].fail("Batch registration failed")
                registered_names.append(key)
            return FlextResult[list[str]].ok(registered_names)
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
            # Rollback on unexpected errors
            self._registrar.get_services_dict().clear()
            self._registrar.get_services_dict().update(services_snapshot)
            self._registrar.get_factories_dict().clear()
            self._registrar.get_factories_dict().update(factories_snapshot)
            return FlextResult[list[str]].fail(f"Batch registration crashed: {e}")

    # =========================================================================
    # CLASS METHODS FOR GLOBAL CONTAINER MANAGEMENT - Architectural Tier 1
    # =========================================================================

    @classmethod
    def _ensure_global_manager(cls) -> FlextContainer.GlobalManager:
        """Ensure global manager is initialized."""
        if cls._global_manager is None:
            cls._global_manager = cls.GlobalManager()
        # Type assertion since we just ensured it's not None
        return cls._global_manager

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Get global container instance (class method)."""
        manager = cls._ensure_global_manager()
        return manager.get_container()

    @classmethod
    def configure_global(
        cls,
        container: FlextContainer | None = None,
    ) -> FlextContainer:
        """Configure global container (class method)."""
        if container is None:
            container = cls()
        manager = cls._ensure_global_manager()
        manager.set_container(container)
        return container

    @classmethod
    def get_global_typed(cls, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get typed service from global container (class method)."""
        container = cls.get_global()
        return container.get_typed(name, expected_type)

    @classmethod
    def register_global(cls, name: str, service: object) -> FlextResult[None]:
        """Register service in global container (class method)."""
        container = cls.get_global()
        return container.register(name, service)

    @classmethod
    def create_module_utilities(cls, module_name: str) -> dict[str, object]:
        """Create standardized container helpers for a module namespace (class method)."""

        def _get_container() -> FlextContainer:
            return cls.get_global()

        def _configure_dependencies() -> FlextResult[None]:
            # Intentionally a no-op default. Modules can replace this function
            # to perform actual registrations when needed.
            return FlextResult[None].ok(None)

        def _get_service(name: str) -> FlextResult[object]:
            container = cls.get_global()
            direct = container.get(name)
            if direct.is_success:
                return direct
            # Fallback to namespaced key
            return container.get(f"{module_name}.{name}")

        return {
            "get_container": _get_container,
            "configure_dependencies": _configure_dependencies,
            "get_service": _get_service,
        }

    @override
    def __repr__(self) -> str:
        """Return string representation of container."""
        count = self.get_service_count()
        return f"FlextContainer(services: {count})"

    # Define validate_service_name locally to avoid circular import
    @staticmethod
    def flext_validate_service_name(name: object) -> FlextResult[None]:
        """Validate service name string."""
        if not isinstance(name, str):
            return FlextResult[None].fail("Service name must be a non-empty string")
        if not name:
            return FlextResult[None].fail("Service name must be a non-empty string")
        if not name.strip():
            return FlextResult[None].fail("Service name cannot be only whitespace")
        return FlextResult[None].ok(None)

    @staticmethod
    def _get_exception_class(name: str) -> type[Exception]:
        """Get exception class by name from FlextExceptions."""
        return cast("type[Exception]", getattr(FlextExceptions, name))


__all__: list[str] = [
    "FlextContainer",
]
