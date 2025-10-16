"""Protocol definitions for interface contracts and type safety.

This module provides FlextProtocols, a hierarchical collection of protocol
definitions that establish interface contracts and enable type-safe
implementations throughout the FLEXT ecosystem.

ARCHITECTURE:
    Layer 0: Foundation protocols (used within flext-core)
    Layer 1: Domain protocols (services, repositories)
    Layer 2: Application protocols (command/query patterns)
    Layer 3: Infrastructure protocols (external integrations)

PROTOCOL INHERITANCE:
    Protocols use inheritance to reduce duplication and create logical hierarchies.
    Example: HasModelFields extends HasModelDump, adding model_fields attribute.

USAGE IN PROJECTS:
    Domain libraries extend FlextProtocols with domain-specific protocols:

    >>> class FlextLdapProtocols(FlextProtocols):
    ...     class Ldap:
    ...         class LdapConnection(FlextProtocols.Service):
    ...             # LDAP-specific extensions
    ...             pass

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import (
    Generic,
    Protocol,
    overload,
    runtime_checkable,
)

from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    T_Repository_contra,
    T_ResultProtocol,
    T_Service_co,
    TInput_Handler_contra,
    TResult_Handler_co,
)


class FlextProtocols:
    """Hierarchical protocol definitions for FLEXT ecosystem.

    This class provides a complete protocol hierarchy for the FLEXT ecosystem,
    organized by architectural layers and using inheritance to reduce duplication.

    CORE PRINCIPLES:
        1. Protocols in flext-core are ONLY those used within flext-core
        2. Domain-specific protocols live in their respective projects
        3. Protocol inheritance creates logical hierarchies
        4. All protocols are @runtime_checkable for isinstance() validation

    ARCHITECTURAL LAYERS:
        - Foundation: Core building blocks (serialization, validation)
        - Domain: Business logic protocols (services, repositories)
        - Application: Use case patterns (handlers, command bus)
        - Infrastructure: External integrations (connections, logging)

    EXTENSION PATTERN:
        Domain libraries extend FlextProtocols:

        >>> class FlextAuthProtocols(FlextProtocols):
        ...     class Auth:
        ...         class UserProtocol(FlextProtocols.Service):
        ...             pass

    """

    # =========================================================================
    # FOUNDATION LAYER - Core protocols used within flext-core
    # =========================================================================

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for objects with model_dump method (Pydantic compatibility).

        Base protocol for Pydantic-like model serialization. Extended by other
        protocols to add additional capabilities.

        Used in: utilities.py (safe_serialize_to_dict, model_dump)

        Extensions:
            - HasModelFields: Adds model_fields attribute
            - ModelProtocol: Adds validation methods
        """

        def model_dump(self, mode: str = "python") -> FlextTypes.Dict:
            """Dump the model to a dictionary.

            Args:
                mode: Serialization mode ('python' or 'json')

            Returns:
                Dictionary representation of the model

            """
            ...

    @runtime_checkable
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for objects with model_fields attribute.

        Extends HasModelDump with model_fields attribute for Pydantic models.
        Inherits model_dump method from parent protocol.

        Used in: utilities.py (safe_serialize_to_dict)

        Inheritance: HasModelDump → HasModelFields
        """

        model_fields: FlextTypes.Dict

    @runtime_checkable
    class HasResultValue(Protocol):
        """Protocol for FlextResult-like objects.

        Minimal protocol for result types with value and success status.
        Used for type checking without importing FlextResult (breaks circular imports).

        Used in: processors.py (middleware processing)
        """

        value: object
        is_success: bool

    @runtime_checkable
    class HasValidateCommand(Protocol):
        """Protocol for commands with validate_command method.

        CQRS pattern support for command validation before execution.

        Used in: bus.py (command validation)
        """

        def validate_command(self) -> FlextResult[None]:
            """Validate command and return FlextResult."""
            ...

    @runtime_checkable
    class HasInvariants(Protocol):
        """Protocol for domain objects with business invariants.

        Domain-Driven Design pattern for aggregate root validation.

        Used in: models.py (validate_aggregate_consistency)
        """

        def check_invariants(self) -> None:
            """Check business invariants for the object.

            Raises:
                FlextExceptions.ValidationError: If any invariant is violated

            """
            ...

    @runtime_checkable
    class HasTimestamps(Protocol):
        """Protocol for objects with timestamp attributes.

        Provides standardized timestamp tracking for audit and versioning.
        Used for entities that need creation/modification timestamps.

        Used in: models with timestamp fields
        """

        created_at: str | int | float
        updated_at: str | int | float

    @runtime_checkable
    class HasHandlerType(Protocol):
        """Protocol for handlers with type identification.

        Allows handlers to declare their type (command/query) for routing
        and middleware processing.

        Used in: handler implementations
        """

        handler_type: str

    @runtime_checkable
    class Configurable(Protocol):
        """Protocol for configurable components.

        Infrastructure protocol for components that can be configured with
        dictionary-based settings. Returns FlextResult for error handling.

        Used in: container.py (FlextContainer configuration)

        Note: Replaces duplicate Configurable in Infrastructure namespace.
        """

        def configure(self, config: FlextTypes.Dict) -> FlextResult[None]:
            """Configure component with provided settings.

            Args:
                config: Configuration dictionary

            Returns:
                FlextResult[None]: Success if configured, failure with error details

            """
            ...

    # =========================================================================
    # CIRCULAR IMPORT PREVENTION PROTOCOLS
    # =========================================================================
    # These protocols prevent circular dependencies between core modules
    # by providing interfaces without importing concrete implementations.

    @runtime_checkable
    class ResultProtocol(Protocol, Generic[T_ResultProtocol]):
        """Protocol for FlextResult-like types (prevents circular imports).

        Defines the interface for result types without importing the concrete
        FlextResult class, preventing circular dependencies between config,
        models, utilities, and result modules.

        Note: For internal use only. Domain libraries should use FlextResult directly.
        """

        @property
        def is_success(self) -> bool:
            """Check if result represents success."""
            ...

        @property
        def is_failure(self) -> bool:
            """Check if result represents failure."""
            ...

        @property
        def value(self) -> T_ResultProtocol:
            """Get the success value (may raise if failure)."""
            ...

        @property
        def error(self) -> str | None:
            """Get the error message if failure, None otherwise."""
            ...

        def unwrap(self) -> T_ResultProtocol:
            """Extract value, raising exception if failure."""
            ...

        def unwrap_or(self, default: T_ResultProtocol) -> T_ResultProtocol:
            """Extract value or return default if failure."""
            ...

    @runtime_checkable
    class ConfigProtocol(Protocol):
        """Protocol for FlextConfig-like types (prevents circular imports).

        Defines the interface for configuration objects without importing the
        concrete FlextConfig class, preventing circular dependencies.

        Note: For internal use only. Domain libraries should use FlextConfig directly.
        """

        @property
        def debug(self) -> bool:
            """Check if debug mode is enabled."""
            ...

        @property
        def log_level(self) -> str:
            """Get logging level."""
            ...

    @runtime_checkable
    class ModelProtocol(HasModelDump, Protocol):
        """Protocol for domain model types (prevents circular imports).

        Extends HasModelDump with validation and JSON serialization.
        Prevents circular dependencies between models, config, and utilities.

        Inheritance: HasModelDump → ModelProtocol

        Note: For internal use only. Domain libraries should use FlextModels directly.
        """

        def validate(self) -> object:
            """Validate model business rules.

            Returns:
                FlextResult[None]: Success if valid, failure with error details

            """
            ...

        def model_dump_json(self, **kwargs: object) -> str:
            """Dump model to JSON string (Pydantic compatibility).

            Args:
                **kwargs: Additional serialization options

            Returns:
                JSON string representation of the model

            """
            ...

    # =========================================================================
    # DOMAIN LAYER - Service and Repository protocols
    # =========================================================================

    @runtime_checkable
    class Service(Protocol, Generic[T_Service_co]):
        """Base domain service protocol.

        Provides the foundation for all domain services in the FLEXT ecosystem.
        Domain libraries extend this protocol with specific service operations.

        Domain Extensions:
            - FlextLdapProtocols.Ldap.LdapConnectionProtocol
            - FlextAuthProtocols.Auth.ServiceProtocol
            - FlextGrpcProtocols.Grpc.ServerProtocol
        """

        @abstractmethod
        def execute(self) -> object:
            """Execute the main domain operation.

            Returns:
                FlextResult[T_Service_co]: Success with domain result or failure

            """
            ...

        def is_valid(self) -> bool:
            """Check if the domain service is in a valid state.

            Returns:
                bool: True if valid, False otherwise

            """
            ...

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate business rules for the domain service.

            Returns:
                FlextResult[None]: Success if valid, failure with error details

            """
            ...

        def validate_config(self) -> FlextResult[None]:
            """Validate service configuration.

            Returns:
                FlextResult[None]: Success if valid, failure with error details

            """
            ...

        def get_service_info(self: object) -> FlextTypes.Dict:
            """Get service information and metadata.

            Returns:
                FlextTypes.Dict: Service information dictionary

            """
            ...

    @runtime_checkable
    class Repository(Protocol, Generic[T_Repository_contra]):
        """Base repository protocol for data access.

        Provides the foundation for repository implementations following
        Domain-Driven Design patterns. Domain libraries extend with specific
        repository operations.

        Domain Extensions:
            - FlextLdapProtocols should define LDAP-specific repositories
            - FlextAuthProtocols should define user/session repositories
        """

        @abstractmethod
        def get_by_id(self, entity_id: str) -> object:
            """Retrieve an aggregate using the standardized identity lookup."""
            ...

        @abstractmethod
        def save(self, entity: T_Repository_contra) -> object:
            """Persist an entity following modernization consistency rules."""
            ...

        @abstractmethod
        def delete(self, entity_id: str) -> object:
            """Delete an entity while respecting modernization invariants."""
            ...

        @abstractmethod
        def find_all(self: object) -> object:
            """Enumerate entities for modernization-aligned queries."""
            ...

    # =========================================================================
    # APPLICATION LAYER - Command/Query patterns
    # =========================================================================

    @runtime_checkable
    class Handler(Protocol, Generic[TInput_Handler_contra, TResult_Handler_co]):
        """Application handler protocol for CQRS patterns.

        Used in: handlers.py (FlextHandlers implementation)

        Provides standardized interface for command and query handlers with
        validation and execution methods.
        """

        @abstractmethod
        def handle(self, message: TInput_Handler_contra) -> object:
            """Handle the message and return result.

            Args:
                message: The input message to process

            Returns:
                FlextResult[TResult_Handler_co]: Success with result or failure

            """
            ...

        def __call__(self, input_data: TInput_Handler_contra) -> object:
            """Process input and return a FlextResult containing the output."""
            ...

        def can_handle(self, message_type: object) -> bool:
            """Check if handler can process this message type."""
            ...

        def execute(self, message: TInput_Handler_contra) -> object:
            """Execute the handler with the given message."""
            ...

        def validate_command(self, command: TInput_Handler_contra) -> object:
            """Validate a command message."""
            ...

        def validate(self, _data: TInput_Handler_contra) -> object:
            """Validate input before processing."""
            ...

        def validate_query(self, query: TInput_Handler_contra) -> object:
            """Validate a query message."""
            ...

        @property
        def handler_name(self: object) -> str:
            """Get the handler name."""
            ...

        @property
        def mode(self: object) -> str:
            """Get the handler mode (command/query)."""
            ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Protocol for command bus routing and execution."""

        @overload
        def register_handler(
            self,
            handler: Callable[[object], object],
            /,
        ) -> FlextResult[None]: ...

        @overload
        def register_handler(
            self,
            command_type: type,
            handler: Callable[[object], object],
            /,
        ) -> FlextResult[None]: ...

        def register_handler(
            self,
            command_type_or_handler: type | Callable[[object], object],
            handler: Callable[[object], object] | None = None,
            /,
        ) -> FlextResult[None]:
            """Register command handler."""
            ...

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command and return result."""
            ...

    @runtime_checkable
    class Middleware(Protocol):
        """Middleware protocol for command/query processing pipeline."""

        def process(
            self,
            command_or_query: object,
            next_handler: Callable[[object], FlextResult[object]],
        ) -> FlextResult[object]:
            """Process command/query through middleware chain."""
            ...

    # =========================================================================
    # INFRASTRUCTURE LAYER - External integrations
    # =========================================================================

    @runtime_checkable
    class LoggerProtocol(Protocol):
        """Infrastructure logging protocol.

        Provides standardized logging interface for infrastructure components.

        Used in: infrastructure logging implementations
        """

        def log(
            self, level: str, message: str, context: FlextTypes.Dict | None = None
        ) -> None:
            """Log a message with optional context."""
            ...

        def debug(self, message: str, context: FlextTypes.Dict | None = None) -> None:
            """Log debug message."""
            ...

        def info(self, message: str, context: FlextTypes.Dict | None = None) -> None:
            """Log info message."""
            ...

        def warning(self, message: str, context: FlextTypes.Dict | None = None) -> None:
            """Log warning message."""
            ...

        def error(self, message: str, context: FlextTypes.Dict | None = None) -> None:
            """Log error message."""
            ...

    @runtime_checkable
    class Connection(Protocol):
        """Generic connection protocol for external systems.

        Base protocol for database, LDAP, API, and other connections.
        Domain libraries extend with specific connection operations.

        Domain Extensions:
            - FlextLdapProtocols.Ldap.LdapConnectionProtocol
            - Database connection protocols
            - API client connection protocols
        """

        def test_connection(self: object) -> object:
            """Test connection to external system."""
            ...

        def get_connection_string(self: object) -> str:
            """Get connection string for external system."""
            ...

        def close_connection(self: object) -> object:
            """Close connection to external system."""
            ...

    # =========================================================================
    # EXTENSIONS LAYER - Additional protocols for ecosystem extensions
    # =========================================================================

    @runtime_checkable
    class PluginContext(Protocol):
        """Protocol for plugin execution contexts.

        Provides context information for plugin execution, including
        configuration, runtime state, and execution metadata.

        Used in: plugin systems
        """

        config: FlextTypes.Dict
        runtime_id: str

    @runtime_checkable
    class Observability(Protocol):
        """Protocol for observability implementations.

        Provides standardized interfaces for metrics, logging, and monitoring
        across the FLEXT ecosystem.

        Used in: monitoring and metrics collection
        """

        def record_metric(self, name: str, value: float, tags: FlextTypes.Dict) -> None:
            """Record a metric with optional tags."""
            ...

        def log_event(self, level: str, message: str, context: FlextTypes.Dict) -> None:
            """Log an event with context."""
            ...


__all__ = [
    "FlextProtocols",  # Main hierarchical protocol architecture
]
