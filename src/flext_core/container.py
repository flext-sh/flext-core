"""Dependency injection container with service management.

Uses Command pattern for service registration operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections import UserString
from collections.abc import Callable
from datetime import datetime
from typing import NotRequired, TypedDict, Unpack, cast, override
from zoneinfo import ZoneInfo

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T
from flext_core.utilities import FlextUtilities


class FlextContainer:
    """Dependency injection container with type-safe service management."""

    # Class-level global manager instance
    _global_manager: FlextContainer.GlobalManager | None = None

    # =========================================================================
    # NESTED CLASSES - Organized functionality following FLEXT patterns
    # =========================================================================

    class ServiceKey[T](UserString, FlextProtocols.Foundation.Validator[str]):
        """Typed service key for type-safe service resolution.

        Provides string-based service keys with type information for improved
        type safety in dependency injection scenarios.
        """

        __slots__ = ()

        @property
        def name(self) -> str:
            """Get service key name."""
            return str(self)  # Service key name as string property

        @classmethod
        def __class_getitem__(cls, _item: object) -> type[FlextContainer.ServiceKey[T]]:
            """Support generic subscription."""
            return cls  # Generic subscription support for service keys

        def validate(self, data: str) -> FlextResult[str]:
            """Validate service key name."""
            # String key validation with empty check
            if not data or not data.strip():
                return FlextResult[str].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[str].ok(data.strip())

    class Commands:
        """Container operation commands."""

        # COMMAND PATTERN HELL: Commands for dependency injection registration!
        # This is CQRS madness for simple service registration - just use dict["name"] = service!

        class RegisterService:
            """Command to register a service instance."""

            # Command object for service registration with TypedDict parameters
            # Normal people: services["database"] = db_instance - DONE!

            # Python 3.13 Advanced: TypedDict for command parameters
            class RegisterServiceParams(TypedDict):
                """RegisterService command parameters."""

                # Service registration parameters with metadata tracking
                # Includes user_id, correlation_id, timestamp for audit trail

                service_name: NotRequired[str]
                service_instance: NotRequired[object]
                command_type: NotRequired[str]  # Command type identifier
                command_id: NotRequired[str]  # Unique command identifier
                timestamp: NotRequired[datetime | None]  # Command timestamp
                user_id: NotRequired[str | None]  # User context identifier
                correlation_id: NotRequired[str]  # Request correlation identifier

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
        """Container query definitions following CQRS pattern."""

        class GetService:
            """Query to retrieve a service."""

            def __init__(
                self,
                service_name: str | None = None,
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
        """Service registration component implementing Single Responsibility Principle."""

        def __init__(self) -> None:
            """Initialize service registrar with empty registry."""
            self._services: FlextTypes.Service.ServiceDict = {}
            self._factories: FlextTypes.Service.FactoryDict = {}

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name."""
            validation_result = FlextContainer.flext_validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[str].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY
                )
            return FlextResult[str].ok(name.strip())

        def register_service(
            self,
            name: str,
            service: object,
        ) -> FlextResult[None]:
            """Register service instance."""
            # Fast path: simple validation without FlextResult overhead
            if not FlextUtilities.Validation.is_non_empty_string(name):
                return FlextResult[None].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name = name.strip()

            # Store service in registry (overwrites are allowed)
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

        def get_service_names(self) -> FlextTypes.Core.StringList:
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
        """Service retrieval component implementing single responsibility principle."""

        def __init__(
            self,
            services: FlextTypes.Service.ServiceDict,
            factories: FlextTypes.Service.FactoryDict,
        ) -> None:
            """Initialize service retriever with references."""
            super().__init__()
            self._services = services
            self._factories = factories

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name."""
            validation_result = FlextContainer.flext_validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[str].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY
                )
            return FlextResult[str].ok(name.strip())

        def get_service(self, name: str) -> FlextResult[object]:
            """Retrieve a registered service - Performance optimized."""
            if not FlextUtilities.Validation.is_non_empty_string(name):
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

        def get_service_info(self, name: str) -> FlextResult[FlextTypes.Core.Dict]:
            """Get detailed information about a registered service or factory.

            Args:
                name: The name of the service or factory to get information about.

            Returns:
                A FlextResult containing the information about the service or factory.

            """
            if not FlextUtilities.Validation.is_non_empty_string(name):
                return FlextResult[FlextTypes.Core.Dict].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                )

            validated_name = name.strip()

            # Check if a service is registered as instance
            if validated_name in self._services:
                service = self._services[validated_name]
                service_class = service.__class__
                return FlextResult[FlextTypes.Core.Dict].ok(
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
                return FlextResult[FlextTypes.Core.Dict].ok(
                    {
                        "name": validated_name,
                        "type": "factory",
                        "factory": factory_name,
                        "module": factory_module,
                    },
                )

            return FlextResult[FlextTypes.Core.Dict].fail(
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
        """Simple global container manager for singleton pattern."""

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

    def __init__(self) -> None:
        """Initialize container."""
        super().__init__()

        # Get global configuration singleton
        self._global_config = FlextConfig.get_global_instance()

        # SRP: Delegate to focused internal components
        self._registrar = self.ServiceRegistrar()

        # DIP: Retriever depends on registrar's data abstractions
        services_dict = self._registrar.get_services_dict()
        factories_dict = self._registrar.get_factories_dict()
        self._retriever = self.ServiceRetriever(services_dict, factories_dict)

        # Use FlextConfig singleton for all configuration needs
        # Domain-specific configs belong in your application, not flext-core!

        # Keep these for backward compatibility with tests
        self._database_config: FlextTypes.Core.Dict | None = None
        self._security_config: FlextTypes.Core.Dict | None = None
        self._logging_config: FlextTypes.Core.Dict | None = None
        self._flext_config: FlextConfig | None = None

        # Simplified command bus - can be extended later
        self._command_bus = None

    # =========================================================================
    # SPECIALIZED CONFIGURATION PROPERTIES
    # =========================================================================

    # Keep for backward compatibility with tests
    @property
    def database_config(self) -> FlextTypes.Core.Dict | None:
        """Access database configuration if available."""
        return self._database_config

    @property
    def security_config(self) -> FlextTypes.Core.Dict | None:
        """Access security configuration if available."""
        return self._security_config

    @property
    def logging_config(self) -> FlextTypes.Core.Dict | None:
        """Access logging configuration if available."""
        return self._logging_config

    def configure_database(self, config: FlextTypes.Core.Dict) -> None:
        """Configure database settings for this container."""
        self._database_config = config

    def configure_security(self, config: FlextTypes.Core.Dict) -> None:
        """Configure security settings for this container."""
        self._security_config = config

    def configure_logging(self, config: FlextTypes.Core.Dict) -> None:
        """Configure logging settings for this container."""
        self._logging_config = config

    def configure_container(self, config: dict[str, object]) -> FlextResult[object]:
        """Configure container settings using FlextConfig."""
        try:
            # Map container fields to FlextConfig fields
            flext_config_data = {}
            for key, value in config.items():
                if key == "max_services":
                    flext_config_data["max_workers"] = value
                elif key == "service_timeout":
                    flext_config_data["timeout_seconds"] = value
                elif key in {"environment", "log_level", "config_source", "debug"}:
                    flext_config_data[key] = value
                # Ignore unknown fields for now

            # Use FlextConfig.create to handle validation and configuration
            flext_config_result = FlextConfig.create(constants=flext_config_data)
            if flext_config_result.is_failure:
                return FlextResult[object].fail(
                    f"Configuration validation failed: {flext_config_result.error}"
                )

            # Store the validated configuration
            self._flext_config = flext_config_result.value

            return FlextResult[object].ok("Container configured successfully")
        except Exception as e:
            return FlextResult[object].fail(f"Failed to configure container: {e}")

    def get_container_config(self) -> FlextResult[dict[str, object]]:
        """Get current container configuration using FlextConfig."""
        try:
            # Handle case when no config is set
            if self._flext_config is None:
                default_config: dict[str, object] = {
                    "environment": "development",
                    "max_services": 1000,  # Map to max_services for compatibility
                    "max_workers": 4,
                    "timeout_seconds": 30,
                    "debug": False,
                    "log_level": "INFO",
                }
                return FlextResult[dict[str, object]].ok(default_config)

            # This code only executes when self._flext_config is not None
            # Use FlextConfig's to_dict method and map fields for compatibility
            config_dict = self._flext_config.to_dict()

            # Map FlextConfig fields to container expected fields
            mapped_config = {
                "environment": config_dict.get("environment", "development"),
                "max_services": config_dict.get(
                    "max_workers", 4
                ),  # Map max_workers to max_services
                "max_workers": config_dict.get("max_workers", 4),
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "debug": config_dict.get("debug", False),
                "log_level": config_dict.get("log_level", "INFO"),
                "config_source": config_dict.get("config_source", "default"),
                "service_timeout": config_dict.get(
                    "timeout_seconds", 30
                ),  # Map timeout_seconds to service_timeout
                "enable_auto_wire": False,  # Default value
                "enable_factory_cache": True,  # Default value
            }

            return FlextResult[dict[str, object]].ok(mapped_config)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get container config: {e}"
            )

    def get_configuration(self) -> FlextResult[dict[str, object]]:
        """Get container configuration (alias for get_container_config)."""
        return self.get_container_config()

    def get_configuration_summary(self) -> FlextResult[dict[str, object]]:
        """Get a summary of container configuration and status."""
        try:
            # Handle case when no config is set
            if self._flext_config is None:
                summary: dict[str, object] = {
                    "container_config": {"status": "default"},
                    "environment_info": {
                        "environment": "development",
                        "container_id": id(self),
                    },
                    "service_statistics": {
                        "total_services": len(self._registrar._services)
                    },
                    "has_database_config": self._database_config is not None,
                    "has_security_config": self._security_config is not None,
                    "has_logging_config": self._logging_config is not None,
                }
                return FlextResult[dict[str, object]].ok(summary)

            # This code only executes when self._flext_config is not None
            # Use FlextConfig for configuration summary
            config_dict = self._flext_config.to_dict()

            container_config = {
                "environment": config_dict.get("environment", "development"),
                "max_workers": config_dict.get("max_workers", 4),
                "timeout_seconds": config_dict.get("timeout_seconds", 30),
                "debug": config_dict.get("debug", False),
                "log_level": config_dict.get("log_level", "INFO"),
            }

            environment_info = {
                "environment": config_dict.get("environment", "development"),
                "container_id": id(self),
                "current_services": len(self._registrar._services),
            }

            service_statistics = {
                "total_services": len(self._registrar._services),
                "total_factories": len(self._registrar._factories),
                "service_names": list(self._registrar._services.keys()),
                "factory_names": list(self._registrar._factories.keys()),
            }

            summary = {
                "container_config": container_config,
                "environment_info": environment_info,
                "service_statistics": service_statistics,
                "has_database_config": self._database_config is not None,
                "has_security_config": self._security_config is not None,
                "has_logging_config": self._logging_config is not None,
            }
            return FlextResult[dict[str, object]].ok(summary)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Failed to get configuration summary: {e}"
            )

    def create_scoped_container(
        self, config: FlextConfig | None = None
    ) -> FlextResult[FlextContainer]:
        """Create a new container with injected configuration."""
        try:
            # Create new container
            scoped_container = FlextContainer()

            # Use injected config if provided
            if config:
                scoped_container._flext_config = config

            return FlextResult[FlextContainer].ok(scoped_container)
        except Exception as e:
            return FlextResult[FlextContainer].fail(
                f"Failed to create scoped container: {e}"
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
        factory: Callable[[], object] | object,
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

    def get_service_names(self) -> FlextTypes.Core.StringList:
        """Get service names."""
        return self._registrar.get_service_names()

    def get_service_count(self) -> int:
        """Get service count."""
        return self._registrar.get_service_count()

    @property
    def command_bus(self) -> object:
        """Access to the internal command bus for operations."""
        return self._command_bus

    def get_info(self, name: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Return basic info about a registered service or factory."""
        try:
            if name in self._registrar.get_services_dict():
                service = self._registrar.get_services_dict()[name]
                service_info: FlextTypes.Core.Dict = {
                    "name": name,
                    "kind": "instance",
                    "type": "instance",
                    "class": type(service).__name__,
                }

                # If service is a dict, include its data
                if isinstance(service, dict):
                    service_info.update(service)

                return FlextResult[FlextTypes.Core.Dict].ok(service_info)
            if name in self._registrar.get_factories_dict():
                factory = self._registrar.get_factories_dict()[name]
                factory_info: FlextTypes.Core.Dict = {
                    "name": name,
                    "kind": "factory",
                    "type": "factory",
                    "class": type(factory).__name__,
                }
                return FlextResult[FlextTypes.Core.Dict].ok(factory_info)
            return FlextResult[FlextTypes.Core.Dict].fail(f"Service '{name}' not found")
        except (KeyError, AttributeError, TypeError) as e:
            return FlextResult[FlextTypes.Core.Dict].fail(f"Info retrieval failed: {e}")

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], object] | None = None,
    ) -> FlextResult[object]:
        """Get existing service or register-and-return a newly created one."""
        existing = self.get(name)
        if existing.is_success:
            return existing

        if factory is None:
            return FlextResult[object].fail("Factory is required for get_or_create")

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
        service_class: type[T] | str,
        service_name: str | None = None,
    ) -> FlextResult[T]:
        """Auto-wire service dependencies and register the service."""
        try:
            # Handle string service_class parameter (for test compatibility)
            if isinstance(service_class, str):
                return FlextResult[T].fail(
                    f"Service class must be a type, not string: {service_class}"
                )

            # Use class name as default service name
            name = service_name or service_class.__name__

            # Get constructor signature
            sig = inspect.signature(service_class.__init__)
            args: FlextTypes.Core.List = []
            kwargs: FlextTypes.Core.Dict = {}

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
        registrations: FlextTypes.Core.Dict,
    ) -> FlextResult[FlextTypes.Core.StringList]:
        """Register many services/factories atomically; roll back on failure."""
        # Snapshot current state for rollback
        services_snapshot = dict(self._registrar.get_services_dict())
        factories_snapshot = dict(self._registrar.get_factories_dict())
        registered_names: FlextTypes.Core.StringList = []
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
                    return FlextResult[FlextTypes.Core.StringList].fail(
                        "Batch registration failed"
                    )
                registered_names.append(key)
            return FlextResult[FlextTypes.Core.StringList].ok(registered_names)
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
            # Rollback on unexpected errors
            self._registrar.get_services_dict().clear()
            self._registrar.get_services_dict().update(services_snapshot)
            self._registrar.get_factories_dict().clear()
            self._registrar.get_factories_dict().update(factories_snapshot)
            return FlextResult[FlextTypes.Core.StringList].fail(
                f"Batch registration crashed: {e}"
            )

    # =========================================================================
    # CLASS METHODS FOR GLOBAL CONTAINER MANAGEMENT - Architectural Tier 1
    # =========================================================================

    @classmethod
    def _ensure_global_manager(cls) -> FlextContainer.GlobalManager:
        """Ensure global manager is initialized."""
        if cls._global_manager is None:
            cls._global_manager = cls.GlobalManager()

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
    def create_module_utilities(cls, module_name: str) -> FlextTypes.Core.Dict:
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


__all__: FlextTypes.Core.StringList = [
    "FlextContainer",
]
