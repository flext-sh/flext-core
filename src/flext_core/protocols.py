"""Protocol definitions and interface contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Type variables for generic protocols
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

# Type variables for generic protocols with correct variance
T_co = TypeVar("T_co", covariant=True)  # For output types
T_contra = TypeVar("T_contra", contravariant=True)  # For input types
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TOutput_co = TypeVar("TOutput_co", covariant=True)


class FlextProtocols:
    """Hierarchical protocol architecture with composition patterns."""

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols - core building blocks."""

        class Callable(Protocol, Generic[T_co]):
            """Generic callable protocol with type safety."""

            def __call__(self, *args: object, **kwargs: object) -> T_co:
                """Execute callable with arguments."""
                ...

        @runtime_checkable
        class DecoratedCallable(Protocol, Generic[T_co]):
            """Callable protocol with function attributes for decorators."""

            def __call__(self, *args: object, **kwargs: object) -> T_co:
                """Execute callable with arguments."""
                ...

            __name__: str
            __module__: str
            __doc__: str | None
            __qualname__: str
            __annotations__: FlextTypes.Core.Dict
            __dict__: FlextTypes.Core.Dict
            __wrapped__: object | None  # Can be any callable

        class SupportsRichComparison(Protocol):
            """Protocol for objects supporting rich comparison."""

            def __lt__(self, other: object) -> bool:
                """Less than comparison."""
                ...

            def __le__(self, other: object) -> bool:
                """Less than or equal comparison."""
                ...

            def __gt__(self, other: object) -> bool:
                """Greater than comparison."""
                ...

            def __ge__(self, other: object) -> bool:
                """Greater than or equal comparison."""
                ...

            def __eq__(self, other: object) -> bool:
                """Equality comparison."""
                ...

            def __ne__(self, other: object) -> bool:
                """Not equal comparison."""
                ...

            def __hash__(self) -> int:
                """Hash support for rich comparison objects."""
                ...

        class Validator(Protocol, Generic[T_contra]):
            """Generic validator protocol."""

            def validate(self, data: T_contra) -> object:
                """Validate input data and return status."""
                ...

        class ErrorHandler(Protocol):
            """Error handler protocol."""

            def handle_error(self, error: Exception) -> str:
                """Transform exception to error message."""
                ...

        class Factory(Protocol, Generic[T_co]):
            """Type-safe factory protocol."""

            def create(self, **kwargs: object) -> T_co:
                """Create instance of type T."""
                ...

        class AsyncFactory(Protocol, Generic[T_co]):
            """Async factory protocol."""

            async def create_async(self, **kwargs: object) -> T_co:
                """Create instance asynchronously."""
                ...

        @runtime_checkable
        class HasToDictBasic(Protocol):
            """Protocol for objects exposing to_dict_basic."""

            def to_dict_basic(
                self,
            ) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
                """Convert object to basic dictionary."""
                ...

        @runtime_checkable
        class HasToDict(Protocol):
            """Protocol for objects exposing to_dict."""

            def to_dict(
                self,
            ) -> FlextTypes.Core.Dict:  # pragma: no cover - typing helper
                """Convert object to dictionary."""
                ...

        @runtime_checkable
        class SupportsDynamicAttributes(Protocol):
            """Protocol for objects that support dynamic attribute setting.

            This protocol allows mixins to set arbitrary attributes on objects
            without triggering MyPy errors for missing attributes.
            """

            def __setattr__(self, name: str, value: object, /) -> None:
                """Set attribute on object."""
                ...

            def __getattribute__(self, name: str, /) -> object:
                """Get attribute from object."""
                ...

        @runtime_checkable
        class HasModelDump(Protocol):
            """Protocol for Pydantic v2 models with model_dump method."""

            def model_dump(self) -> FlextTypes.Core.Dict:
                """Convert model to dictionary (Pydantic v2 style)."""
                ...

        @runtime_checkable
        class HasDict(Protocol):
            """Protocol for Pydantic v1 models with dict method."""

            def dict(self) -> FlextTypes.Core.Dict:
                """Convert model to dictionary (Pydantic v1 style)."""
                ...

        @runtime_checkable
        class HasModelValidate(Protocol):
            """Protocol for Pydantic v2 models with model_validate class method."""

            @classmethod
            def model_validate(cls, obj: object) -> object:
                """Validate and create model instance from object data."""
                ...

        @runtime_checkable
        class DataConstructor(Protocol):
            """Protocol for classes that can be constructed from data."""

            def __call__(self, data: object) -> object:
                """Construct instance from data object."""
                ...

        @runtime_checkable
        class SizedDict(Protocol):
            """Protocol for dict-like objects that support len()."""

            def __len__(self) -> int:
                """Return length of dict."""
                ...

        @runtime_checkable
        class SizedList(Protocol):
            """Protocol for list-like objects that support len()."""

            def __len__(self) -> int:
                """Return length of list."""
                ...

    # =========================================================================
    # DOMAIN LAYER - Business logic protocols
    # =========================================================================

    class Domain:
        """Domain layer protocols - business logic."""

        class Service(Protocol):
            """Domain service protocol with lifecycle management."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for service."""
                ...

            @abstractmethod
            def start(self) -> object:
                """Start the service."""
                ...

            @abstractmethod
            def stop(self) -> object:
                """Stop the service."""
                ...

            @abstractmethod
            def health_check(self) -> object:
                """Perform health check."""
                ...

        class Repository(Protocol, Generic[T_contra]):
            """Repository protocol for data access."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Get entity by ID."""
                ...

            @abstractmethod
            def save(self, entity: T_contra) -> object:
                """Save entity."""
                ...

            @abstractmethod
            def delete(self, entity_id: str) -> object:
                """Delete entity by ID."""
                ...

            @abstractmethod
            def find_all(self) -> object:
                """Find all entities."""
                ...

        class DomainEvent(Protocol):
            """Domain event protocol."""

            event_id: str
            event_type: str
            aggregate_id: str
            event_version: int
            timestamp: str

            def to_dict(self) -> FlextTypes.Core.Dict:
                """Convert event to dictionary."""
                ...

            @classmethod
            def from_dict(
                cls,
                data: FlextTypes.Core.Dict,
            ) -> FlextProtocols.Domain.DomainEvent:
                """Create event from dictionary."""
                ...

        class EventStore(Protocol):
            """Event store protocol."""

            @abstractmethod
            def save_events(
                self,
                aggregate_id: str,
                events: list[FlextProtocols.Domain.DomainEvent],
                expected_version: int,
            ) -> object:
                """Save events for aggregate."""
                ...

            @abstractmethod
            def get_events(self, aggregate_id: str) -> object:
                """Get events for aggregate."""
                ...

    # =========================================================================
    # APPLICATION LAYER - Use cases and handlers
    # =========================================================================

    class Application:
        """Application layer protocols - use cases and handlers."""

        class Handler(Protocol, Generic[TInput_contra, TOutput_co]):
            """Application handler with validation."""

            def __call__(self, input_data: TInput_contra) -> object:
                """Process input and return output."""
                ...

            def validate(self, data: TInput_contra) -> object:
                """Validate input before processing."""
                ...

        class MessageHandler(Protocol):
            """Message handler for CQRS patterns."""

            def handle(self, message: object) -> object:
                """Handle incoming message and return result."""
                ...

            def can_handle(self, message_type: type) -> bool:
                """Check if handler can process a message type."""
                ...

        class ValidatingHandler(MessageHandler, Protocol):
            """Handler with built-in validation capabilities."""

            def validate(self, message: object) -> object:
                """Validate message before processing (Foundation.Validator composition)."""
                ...

        class AuthorizingHandler(MessageHandler, Protocol):
            """Handler with authorization capabilities."""

            def authorize(
                self,
                message: object,
                context: FlextTypes.Core.Dict,
            ) -> object:
                """Check authorization for message processing."""
                ...

        class EventProcessor(Protocol):
            """Event processor for domain event handling."""

            def process_event(self, event: FlextTypes.Core.Dict) -> object:
                """Process domain event."""
                ...

            def can_process(self, event_type: str) -> bool:
                """Check if the processor can handle an event type."""
                ...

        class UnitOfWork(Protocol):
            """Unit of Work pattern for transaction management."""

            @abstractmethod
            def begin(self) -> object:
                """Begin transaction."""
                ...

            @abstractmethod
            def commit(self) -> object:
                """Commit transaction."""
                ...

            @abstractmethod
            def rollback(self) -> object:
                """Rollback transaction."""
                ...

    # =========================================================================
    # INFRASTRUCTURE LAYER - External concerns and integrations
    # =========================================================================

    class Infrastructure:
        """Infrastructure layer protocols - external systems."""

        class Connection(Protocol):
            """Connection protocol for external systems."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection."""
                ...

            def test_connection(self) -> object:
                """Test connection to external system."""
                ...

            def get_connection_string(self) -> str:
                """Get connection string for external system."""
                ...

            def close_connection(self) -> object:
                """Close connection to external system."""
                ...

        class LdapConnection(Connection, Protocol):
            """LDAP-specific connection protocol."""

            def connect(self, uri: str, bind_dn: str, password: str) -> object:
                """Connect to LDAP server."""
                ...

            def bind(self, bind_dn: str, password: str) -> object:
                """Bind with specific credentials."""
                ...

            def unbind(self) -> object:
                """Unbind from LDAP server."""
                ...

            def search(
                self,
                base_dn: str,
                search_filter: str,
                scope: str = "subtree",
            ) -> object:
                """Perform LDAP search operation."""
                ...

            def add(self, dn: str, attributes: FlextTypes.Core.Dict) -> object:
                """Add new LDAP entry."""
                ...

            def modify(self, dn: str, modifications: FlextTypes.Core.Dict) -> object:
                """Modify existing LDAP entry."""
                ...

            def delete(self, dn: str) -> object:
                """Delete LDAP entry."""
                ...

            def is_connected(self) -> bool:
                """Check if connection is active."""
                ...

        class Auth(Protocol):
            """Authentication and authorization protocol."""

            def authenticate(
                self,
                credentials: FlextTypes.Core.Dict,
            ) -> object:
                """Authenticate user with credentials."""
                ...

            def authorize(
                self,
                user_info: FlextTypes.Core.Dict,
                resource: str,
            ) -> object:
                """Authorize user access to resource."""
                ...

            def refresh_token(self, refresh_token: str) -> object:
                """Refresh authentication token."""
                ...

        @runtime_checkable
        class Configurable(Protocol):
            """Configurable component protocol."""

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with provided settings."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
                """Get current configuration."""
                ...

        @runtime_checkable
        class LoggerProtocol(Protocol):
            """Logger protocol with standard logging methods."""

            def trace(self, message: str, **kwargs: object) -> None:
                """Log trace message."""
                ...

            def debug(self, message: str, **kwargs: object) -> None:
                """Log debug message."""
                ...

            def info(self, message: str, **kwargs: object) -> None:
                """Log info message."""
                ...

            def warning(self, message: str, **kwargs: object) -> None:
                """Log warning message."""
                ...

            def error(self, message: str, **kwargs: object) -> None:
                """Log error message."""
                ...

            def critical(self, message: str, **kwargs: object) -> None:
                """Log critical message."""
                ...

            def exception(
                self,
                message: str,
                *,
                exc_info: bool = True,
                **kwargs: object,
            ) -> None:
                """Log exception message."""
                ...

    # =========================================================================
    # EXTENSIONS LAYER - Advanced patterns and plugins
    # =========================================================================

    class Extensions:
        """Extensions layer protocols - plugins and advanced patterns."""

        class Plugin(Protocol):
            """Plugin protocol with configuration."""

            def configure(self, config: FlextTypes.Core.Dict) -> object:
                """Configure component with settings."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
                """Get current configuration."""
                ...

            @abstractmethod
            def initialize(
                self,
                context: FlextProtocols.Extensions.PluginContext,
            ) -> object:
                """Initialize plugin."""
                ...

            @abstractmethod
            def shutdown(self) -> object:
                """Shutdown plugin and cleanup."""
                ...

            @abstractmethod
            def get_info(self) -> FlextTypes.Core.Dict:
                """Get plugin information."""
                ...

        class PluginContext(Protocol):
            """Plugin execution context."""

            def get_service(self, service_name: str) -> object:
                """Get service by name."""
                ...

            def get_config(self) -> FlextTypes.Core.Dict:
                """Get plugin configuration."""
                ...

            def flext_logger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
                """Get logger instance for plugin."""
                ...

        class Middleware(Protocol):
            """Middleware pipeline component protocol."""

            def process(
                self,
                request: object,
                next_handler: Callable[[object], object],
            ) -> object:
                """Process request with middleware logic."""
                ...

        class AsyncMiddleware(Protocol):
            """Async middleware component protocol."""

            async def process_async(
                self,
                request: object,
                next_handler: Callable[[object], Awaitable[object]],
            ) -> object:
                """Process request asynchronously."""
                ...

        @runtime_checkable
        class Observability(Protocol):
            """Observability and monitoring protocol."""

            def record_metric(
                self,
                name: str,
                value: float,
                tags: FlextTypes.Core.Headers | None = None,
            ) -> object:
                """Record metric value."""
                ...

            def start_trace(self, operation_name: str) -> object:
                """Start distributed trace."""
                ...

            def health_check(self) -> object:
                """Perform health check."""
                ...

    # =============================================================================
    # DECORATOR PROTOCOLS - Special function patterns
    # =============================================================================

    class DecoratedFunction(Protocol, Generic[T_co]):
        """Decorated function protocol returning FlextResult for railway-oriented programming."""

        def __call__(self, *args: object, **kwargs: object) -> T_co:
            """Execute the decorated function returning FlextResult."""
            ...

    # =========================================================================
    # COMPATIBILITY LAYER - Optimized aliases for hierarchical access
    # =========================================================================

    # NOTE: Aliases removed - use direct hierarchical access like FlextProtocols.Foundation.Callable

    # =========================================================================
    # CONFIG - Protocol system configuration
    # =========================================================================

    class Config:
        """Enterprise protocol system management with FlextTypes.Config integration."""

        @classmethod
        def configure_protocols_system(
            cls,
            config: FlextTypes.Core.Dict,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Configure protocols system using FlextConfig.ProtocolsSettings with single class model."""
            # Map protocol_level to validation_level for backward compatibility
            config_dict = dict(config)
            original_protocol_level: str | None = None
            if "protocol_level" in config_dict:
                # Validate protocol_level explicitly to preserve expected error message
                valid_levels = [e.value for e in FlextConstants.Config.ValidationLevel]
                level_value = config_dict.get("protocol_level")
                if level_value not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid protocol_level '{level_value}'",
                    )
                # Save original protocol_level value to restore later
                original_protocol_level = cast("str", level_value)
                # Remove protocol_level from config_dict since FlextConfig doesn't have this field
                config_dict.pop("protocol_level")

            # If environment is not explicitly provided, set default
            if "environment" not in config_dict:
                config_dict["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Use FlextConfig.ProtocolsSettings for validation with no env overrides
            # Pass empty env_overrides to prevent loading from environment variables
            settings_res = FlextConfig.create_from_environment(
                extra_settings=config_dict if isinstance(config_dict, dict) else None,
            )
            if settings_res.is_failure:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Protocol configuration failed: {settings_res.error}",
                )

            # Get validated config dict from FlextConfig instance
            validated_config = cast(
                "FlextTypes.Config.ConfigDict", settings_res.value.to_dict()
            )

            # Restore original protocol_level or use default
            if original_protocol_level is not None:
                validated_config["protocol_level"] = original_protocol_level
            else:
                validated_config["protocol_level"] = (
                    FlextConstants.Config.ValidationLevel.LOOSE.value
                )

            # Add backward compatibility fields
            validated_config.setdefault(
                "environment", FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
            )
            validated_config.setdefault("enable_runtime_checking", True)
            validated_config.setdefault("protocol_composition_mode", "HIERARCHICAL")
            validated_config.setdefault("enable_protocol_caching", True)

            # Preserve custom fields from original config that are not in validated_config
            for key, value in config.items():
                if key not in validated_config:
                    validated_config[key] = cast("FlextTypes.Config.ConfigValue", value)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        @classmethod
        def get_protocols_system_config(
            cls,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Get current protocols system configuration."""
            default_config = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "protocol_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "enable_runtime_checking": True,
                "protocol_composition_mode": "HIERARCHICAL",
                "enable_protocol_caching": True,
            }
            try:
                core_validation = {
                    "environment": default_config.get("environment"),
                    "log_level": default_config.get("log_level"),
                    "validation_level": default_config.get("protocol_level"),
                }
                _ = FlextModels.SystemConfigs.ProtocolsConfig.model_validate(
                    core_validation
                )
                return FlextResult[FlextTypes.Config.ConfigDict].ok(
                    cast("FlextTypes.Config.ConfigDict", default_config),
                )
            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid default protocol config: {e!s}",
                )

        @classmethod
        def create_environment_protocols_config(
            cls,
            environment: str,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Create environment-specific protocol configuration."""
            environment_configs = {
                "development": {
                    "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                    "protocol_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_runtime_checking": True,
                    "protocol_composition_mode": "HIERARCHICAL",
                    "enable_protocol_caching": False,
                },
                "production": {
                    "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION.value,
                    "protocol_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_runtime_checking": False,
                    "protocol_composition_mode": "HIERARCHICAL",
                    "enable_protocol_caching": True,
                },
                "test": {
                    "environment": FlextConstants.Config.ConfigEnvironment.TEST.value,
                    "protocol_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_runtime_checking": True,
                    "protocol_composition_mode": "HIERARCHICAL",
                    "enable_protocol_caching": False,
                },
            }

            if environment.lower() not in environment_configs:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Unknown environment '{environment}'. Valid options: {list(environment_configs.keys())}",
                )
            config = environment_configs[environment.lower()]
            try:
                core_validation = {
                    "environment": config.get("environment"),
                    "log_level": config.get("log_level"),
                    "validation_level": config.get("protocol_level"),
                }
                _ = FlextModels.SystemConfigs.ProtocolsConfig.model_validate(
                    core_validation
                )
                return FlextResult[FlextTypes.Config.ConfigDict].ok(
                    cast("FlextTypes.Config.ConfigDict", config),
                )
            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment protocol config: {e!s}",
                )

        @classmethod
        def optimize_protocols_performance(
            cls,
            performance_level: str = "balanced",
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Optimize protocol system performance."""
            optimization_configs = {
                "low": {
                    "enable_runtime_checking": True,
                    "enable_protocol_caching": False,
                    "protocol_composition_mode": "FLAT",
                },
                "balanced": {
                    "enable_runtime_checking": True,
                    "enable_protocol_caching": True,
                    "protocol_composition_mode": "HIERARCHICAL",
                },
                "high": {
                    "enable_runtime_checking": False,
                    "enable_protocol_caching": True,
                    "protocol_composition_mode": "HIERARCHICAL",
                },
            }

            if performance_level not in optimization_configs:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid performance level '{performance_level}'. Valid options: {list(optimization_configs.keys())}",
                )
            config = optimization_configs[performance_level]
            try:
                core_validation = {
                    "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                }
                _ = FlextModels.SystemConfigs.ProtocolsConfig.model_validate(
                    core_validation
                )
                return FlextResult[FlextTypes.Config.ConfigDict].ok(
                    cast("FlextTypes.Config.ConfigDict", config),
                )
            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid optimized protocol config: {e!s}",
                )


# Delayed imports to avoid circular dependencies at runtime
# Dead code removed - unused helper function


# The FlextProtocolsConfig class has been consolidated into FlextProtocols.Config

# Cleanup of old standalone FlextProtocolsConfig class completed
# All functionality moved to FlextProtocols.Config as nested class

# This section can be removed as the class is now nested within FlextProtocols


__all__: FlextTypes.Core.StringList = [
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
]
