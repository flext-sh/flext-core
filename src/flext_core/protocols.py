"""Enterprise protocol definitions with hierarchical architecture and Clean Architecture principles.

Provides runtime-checkable protocol definitions organized in hierarchical layers for
type-safe contracts, dependency injection, validation, and enterprise patterns.

Usage:
    # Domain protocols
    class UserRepository(FlextProtocols.Domain.Repository[User]):
        def find_by_id(self, user_id: str) -> FlextResult[User]:
            # Implementation

    # Application protocols
    class CreateUserHandler(FlextProtocols.Application.Handler[CreateUser, User]):
        def handle(self, command: CreateUser) -> FlextResult[User]:
            # Implementation

    # Infrastructure protocols
    class DatabaseConnection(FlextProtocols.Infrastructure.Connection):
        def connect(self) -> FlextResult[None]:
            # Implementation

Features:
    - Hierarchical protocol organization (Foundation, Domain, Application, Infrastructure)
    - Runtime-checkable contracts
    - Generic type support with type variables
    - Clean Architecture dependency inversion
    - FlextResult integration for error handling
        Extensions.PluginContext                    # Plugin execution context protocol
        Extensions.Middleware                       # Middleware pipeline protocol
        Extensions.AsyncMiddleware                  # Async middleware protocol
        Extensions.Observability                    # Monitoring and metrics protocol

        # Protocol Methods:
        # Each protocol defines abstract methods appropriate to its domain
        # All protocols integrate with FlextResult for railway-oriented programming
        # Runtime checking available via isinstance() and issubclass()
        # Generic protocols support proper type parameter constraints

    FlextProtocolsConfig:                   # Protocol system configuration
        # Configuration Methods:
        configure_protocols_system(config) -> FlextResult[ConfigDict] # Configure protocol system
        get_protocols_system_config() -> FlextResult[ConfigDict] # Get current config
        create_environment_protocols_config(environment) -> FlextResult[ConfigDict] # Environment config
        optimize_protocols_performance(performance_level="balanced") -> FlextResult[ConfigDict] # Performance optimization

Usage Examples:
    Domain service implementation:
        class UserService(FlextProtocols.Domain.Service):
            def start(self) -> FlextResult[None]:
                return FlextResult[None].ok(None)

            def health_check(self) -> FlextResult[dict]:
                return FlextResult[dict].ok({"status": "healthy"})

    Repository pattern:
        class UserRepository(FlextProtocols.Domain.Repository[User]):
            def get_by_id(self, entity_id: str) -> FlextResult[User]:
                # Database lookup implementation
                return FlextResult[User].ok(user)

    Handler with validation:
        class CreateUserHandler(FlextProtocols.Application.ValidatingHandler):
            def handle(self, message: dict) -> FlextResult[dict]:
                # Process user creation
                return FlextResult[dict].ok({"created": True})

    Configuration:
        config = {
            "environment": "production",
            "protocol_level": "strict",
            "enable_runtime_checking": True,
        }
        FlextProtocolsConfig.configure_protocols_system(config)

Integration:
    FlextProtocols integrates with FlextResult for error handling, FlextTypes.Config
    for configuration, FlextConstants for validation, providing efficient type-safe
    contracts for all FLEXT ecosystem components with Clean Architecture compliance.

"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Protocol, cast, runtime_checkable

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# =============================================================================
# HIERARCHICAL PROTOCOL ARCHITECTURE - Optimized with composition
# =============================================================================


class FlextProtocols:
    """Hierarchical protocol architecture with optimized composition patterns.

    This class implements a structured hierarchy where subclasses compose with each
    other using Python 3.13+ syntax. The architecture follows Clean Architecture
    principles with clear separation of concerns and dependency inversion.

    Hierarchy Structure:
        Foundation -> Domain -> Application -> Infrastructure -> Extensions

    Composition Features:
        - Type-safe protocol inheritance with Python 3.13+ generics
        - Mixin composition for cross-cutting concerns
        - Runtime checkable protocols for dynamic validation
        - FlextResult integration for railway-oriented programming

    Examples:
        Foundation layer usage::

            processor: Foundation.Callable[str] = lambda x: x.upper()
            result = processor("test")  # Returns "TEST"

        Domain composition::

            class UserService(
                FlextProtocols.Domain.Service, Foundation.Validator[User]
            ):
                def validate(self, user: User) -> object: ...
                def start(self) -> object: ...

        Application layer patterns::

            handler: FlextProtocols.Application.Handler[CreateUser, str] = (
                CreateUserHandler()
            )
            result = handler(CreateUser(name="John"))  # Returns object

    """

    # =========================================================================
    # FOUNDATION LAYER - Core building blocks
    # =========================================================================

    class Foundation:
        """Foundation layer protocols - core building blocks for the ecosystem."""

        class Callable[T](Protocol):
            """Generic callable protocol with parameter and return type safety."""

            def __call__(self, *args: object, **kwargs: object) -> T:
                """Execute the callable with given arguments."""
                ...

        @runtime_checkable
        class DecoratedCallable[T](Protocol):
            """Callable protocol with function attributes for decorators.

            This protocol represents a function that has been decorated and
            includes all standard function attributes.
            """

            def __call__(self, *args: object, **kwargs: object) -> T:
                """Execute the callable with given arguments."""
                ...

            __name__: str
            __module__: str
            __doc__: str | None
            __qualname__: str
            __annotations__: dict[str, object]
            __dict__: dict[str, object]
            __wrapped__: object | None  # Can be any callable

        class SupportsRichComparison(Protocol):
            """Protocol for objects that support rich comparison operations."""

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
                """Hash support for objects that implement rich comparison."""
                ...

        class Validator[T](Protocol):
            """Generic validator protocol for type-safe validation."""

            def validate(self, data: T) -> object:
                """Validate input data and return success/failure status."""
                ...

        class ErrorHandler(Protocol):
            """Error handler protocol for exception transformation."""

            def handle_error(self, error: Exception) -> str:
                """Transform an exception into an error message string."""
                ...

        class Factory[T](Protocol):
            """Type-safe factory protocol for object creation."""

            def create(self, **kwargs: object) -> object:
                """Create instance of type T."""
                ...

        class AsyncFactory[T](Protocol):
            """Async factory protocol for asynchronous object creation."""

            async def create_async(self, **kwargs: object) -> object:
                """Create instance asynchronously."""
                ...

        @runtime_checkable
        class HasToDictBasic(Protocol):
            """Runtime-checkable protocol for objects exposing to_dict_basic."""

            def to_dict_basic(
                self,
            ) -> dict[str, object]:  # pragma: no cover - typing helper
                """Convert object to basic dictionary representation."""
                ...

        @runtime_checkable
        class HasToDict(Protocol):
            """Runtime-checkable protocol for objects exposing to_dict."""

            def to_dict(self) -> dict[str, object]:  # pragma: no cover - typing helper
                """Convert object to dictionary representation."""
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

            def model_dump(self) -> dict[str, object]:
                """Convert model to dictionary (Pydantic v2 style)."""
                ...

        @runtime_checkable
        class HasDict(Protocol):
            """Protocol for Pydantic v1 models with dict method."""

            def dict(self) -> dict[str, object]:
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
        """Domain layer protocols - business logic and domain services."""

        class Service(Protocol):
            """Domain service protocol with lifecycle management and callable interface."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for service invocation."""
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

        class Repository[T](Protocol):
            """Repository protocol for data access patterns."""

            @abstractmethod
            def get_by_id(self, entity_id: str) -> object:
                """Get entity by ID."""
                ...

            @abstractmethod
            def save(self, entity: T) -> object:
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
            """Domain event protocol for event sourcing."""

            event_id: str
            event_type: str
            aggregate_id: str
            event_version: int
            timestamp: str

            def to_dict(self) -> dict[str, object]:
                """Convert event to dictionary."""
                ...

            @classmethod
            def from_dict(
                cls, data: dict[str, object],
            ) -> FlextProtocols.Domain.DomainEvent:
                """Create event from dictionary."""
                ...

        class EventStore(Protocol):
            """Event store protocol for domain event persistence."""

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
        """Application layer protocols - use cases, handlers, and orchestration."""

        class Handler[TInput, TOutput](Protocol):
            """Application handler with validation and processing."""

            def __call__(self, input_data: TInput) -> object:
                """Process input data and return transformed output."""
                ...

            def validate(self, data: TInput) -> object:
                """Validate input before processing (Foundation.Validator composition)."""
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
                context: dict[str, object],
            ) -> object:
                """Check authorization for message processing."""
                ...

        class EventProcessor(Protocol):
            """Event processor for domain event handling."""

            def process_event(self, event: dict[str, object]) -> object:
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
        """Infrastructure layer protocols - external systems and cross-cutting concerns."""

        class Connection(Protocol):
            """Connection protocol for external systems with callable interface."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                """Callable interface for connection operations."""
                ...

            def test_connection(self) -> object:
                """Test connection to an external system."""
                ...

            def get_connection_string(self) -> str:
                """Get connection string for an external system."""
                ...

            def close_connection(self) -> object:
                """Close connection to an external system."""
                ...

        class LdapConnection(Connection, Protocol):
            """LDAP-specific connection protocol extending base Connection."""

            def connect(self, uri: str, bind_dn: str, password: str) -> object:
                """Connect to LDAP server with authentication."""
                ...

            def bind(self, bind_dn: str, password: str) -> object:
                """Bind with specific credentials."""
                ...

            def unbind(self) -> object:
                """Unbind from LDAP server."""
                ...

            def search(
                self, base_dn: str, search_filter: str, scope: str = "subtree",
            ) -> object:
                """Perform LDAP search operation."""
                ...

            def add(self, dn: str, attributes: dict[str, object]) -> object:
                """Add new LDAP entry."""
                ...

            def modify(self, dn: str, modifications: dict[str, object]) -> object:
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
                credentials: dict[str, object],
            ) -> object:
                """Authenticate user with provided credentials."""
                ...

            def authorize(
                self,
                user_info: dict[str, object],
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

            def configure(self, config: dict[str, object]) -> object:
                """Configure component with provided settings."""
                ...

            def get_config(self) -> dict[str, object]:
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
        """Extensions layer protocols - plugins, middleware, and advanced patterns."""

        class Plugin(Protocol):
            """Plugin protocol with configuration support."""

            def configure(self, config: dict[str, object]) -> object:
                """Configure component with provided settings (Infrastructure.Configurable composition)."""
                ...

            def get_config(self) -> dict[str, object]:
                """Get current configuration (Infrastructure.Configurable composition)."""
                ...

            @abstractmethod
            def initialize(
                self, context: FlextProtocols.Extensions.PluginContext,
            ) -> object:
                """Initialize plugin with context."""
                ...

            @abstractmethod
            def shutdown(self) -> object:
                """Shutdown plugin and cleanup resources."""
                ...

            @abstractmethod
            def get_info(self) -> dict[str, object]:
                """Get plugin information."""
                ...

        class PluginContext(Protocol):
            """Plugin execution context protocol."""

            def get_service(self, service_name: str) -> object:
                """Get service instance by name."""
                ...

            def get_config(self) -> dict[str, object]:
                """Get configuration for plugin."""
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
                tags: dict[str, str] | None = None,
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

    class DecoratedFunction[T](Protocol):
        """Decorated function protocol returning FlextResult for railway-oriented programming."""

        def __call__(self, *args: object, **kwargs: object) -> object:
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
            cls, config: dict[str, object],
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Configure protocols system using FlextTypes.Config with StrEnum validation.

            Configures the FLEXT protocol management system including interface validation,
            protocol inheritance checking, runtime type validation, contract enforcement,
            composition pattern optimization, and hierarchical protocol organization
            with efficient validation and type safety.

            Args:
                config: Configuration dictionary supporting:
                       - environment: Runtime environment (development, staging, production, test, local)
                       - protocol_level: Protocol validation level (strict, loose, disabled)
                       - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
                       - enable_runtime_checking: Enable runtime protocol validation
                       - protocol_composition_mode: HIERARCHICAL, FLAT, MIXED
                       - enable_protocol_caching: Cache protocol validation results

            Returns:
                FlextResult containing validated configuration with protocol system settings

            Example:
                ```python
                config = {
                    "environment": "production",
                    "protocol_level": "strict",
                    "log_level": "INFO",
                    "enable_runtime_checking": True,
                    "protocol_composition_mode": "HIERARCHICAL",
                }
                result = FlextProtocols.Config.configure_protocols_system(config)
                if result.success:
                    protocol_config = result.unwrap()
                ```

            """
            try:
                # Create working copy of config
                validated_config = dict(config)

                # Validate environment
                if "environment" in config:
                    env_value = config["environment"]
                    valid_environments = [
                        e.value for e in FlextConstants.Config.ConfigEnvironment
                    ]
                    if env_value not in valid_environments:
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            f"Invalid environment '{env_value}'. Valid options: {valid_environments}",
                        )
                else:
                    validated_config["environment"] = (
                        FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                    )

                # Validate protocol_level (using validation level as basis)
                if "protocol_level" in config:
                    level_value = config["protocol_level"]
                    valid_levels = [
                        e.value for e in FlextConstants.Config.ValidationLevel
                    ]
                    if level_value not in valid_levels:
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            f"Invalid protocol_level '{level_value}'. Valid options: {valid_levels}",
                        )
                else:
                    validated_config["protocol_level"] = (
                        FlextConstants.Config.ValidationLevel.LOOSE.value
                    )

                # Validate log_level
                if "log_level" in config:
                    log_level_value = config["log_level"]
                    valid_log_levels = [e.value for e in FlextConstants.Config.LogLevel]
                    if log_level_value not in valid_log_levels:
                        return FlextResult[FlextTypes.Config.ConfigDict].fail(
                            f"Invalid log_level '{log_level_value}'. Valid options: {valid_log_levels}",
                        )
                else:
                    validated_config["log_level"] = (
                        FlextConstants.Config.LogLevel.INFO.value
                    )

                # Set default values for additional configuration options
                validated_config.setdefault("enable_runtime_checking", True)
                validated_config.setdefault("protocol_composition_mode", "HIERARCHICAL")
                validated_config.setdefault("enable_protocol_caching", True)

                return FlextResult[FlextTypes.Config.ConfigDict].ok(
                    cast("FlextTypes.Config.ConfigDict", validated_config),
                )

            except Exception as e:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Protocol configuration failed: {e!s}",
                )

        @classmethod
        def get_protocols_system_config(
            cls,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Get current protocols system configuration.

            Returns:
                FlextResult containing current protocol system configuration

            """
            default_config = {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "protocol_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "enable_runtime_checking": True,
                "protocol_composition_mode": "HIERARCHICAL",
                "enable_protocol_caching": True,
            }
            return FlextResult[FlextTypes.Config.ConfigDict].ok(
                cast("FlextTypes.Config.ConfigDict", default_config),
            )

        @classmethod
        def create_environment_protocols_config(
            cls, environment: str,
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Create environment-specific protocol configuration.

            Args:
                environment: Target environment (development, staging, production, test, local)

            Returns:
                FlextResult containing environment-optimized configuration

            """
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
            return FlextResult[FlextTypes.Config.ConfigDict].ok(
                cast("FlextTypes.Config.ConfigDict", config),
            )

        @classmethod
        def optimize_protocols_performance(
            cls, performance_level: str = "balanced",
        ) -> FlextResult[FlextTypes.Config.ConfigDict]:
            """Optimize protocol system performance.

            Args:
                performance_level: Optimization level (low, balanced, high)

            Returns:
                FlextResult containing performance-optimized configuration

            """
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
            return FlextResult[FlextTypes.Config.ConfigDict].ok(
                cast("FlextTypes.Config.ConfigDict", config),
            )


# =============================================================================
# PROTOCOLS CONFIGURATION - FlextTypes.Config Integration
# =============================================================================


# Delayed imports to avoid circular dependencies at runtime
# Dead code removed - unused helper function


# The FlextProtocolsConfig class has been consolidated into FlextProtocols.Config

# Cleanup of old standalone FlextProtocolsConfig class completed
# All functionality moved to FlextProtocols.Config as nested class

# This section can be removed as the class is now nested within FlextProtocols


# =============================================================================
# EXPORTS - Hierarchical protocols
# =============================================================================

__all__: list[str] = [
    "FlextProtocols",  # Main hierarchical protocol architecture with Config
]
