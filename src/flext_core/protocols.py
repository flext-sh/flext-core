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
    TYPE_CHECKING,
    Generic,
    Protocol,
    overload,
    runtime_checkable,
)

from flext_core.typings import (
    FlextTypes,
    T,
    T_co,
    TInput_Handler_Protocol_contra,
    TResult_Handler_Protocol,
)

# TYPE_CHECKING import pattern: protocols.py is Tier 0, result.py is Tier 1.
# Using TYPE_CHECKING prevents runtime circular imports while enabling type checking.
# This is the standard Python pattern (PEP 484) combined with PEP 563 (future annotations).
if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextProtocols:
    """Hierarchical protocol definitions for enterprise FLEXT ecosystem.

    ==============================================================================
    ARCHITECTURE LAYER 0 - PURE CONSTANTS LAYER
    ==============================================================================

    FlextProtocols provides the foundational protocol definitions for the entire
    FLEXT ecosystem, establishing interface contracts and enabling type-safe,
    structural typing (duck typing) compliance across all 32+ dependent projects.

    **Architecture Position**: Layer 0 (Pure Constants - no implementation)
    - Pure interface definitions using Python's Protocol and @runtime_checkable
    - No imports from higher layers (Layer 1-4)
    - Used by all higher layers for type checking and structural typing validation

    **Key Distinction**: These are PROTOCOL DEFINITIONS, not implementations.
    Actual implementations (FlextResult, FlextContainer, FlextService, etc.) live
    in their respective layers.

    ==============================================================================
    STRUCTURAL TYPING (DUCK TYPING) - CORE DESIGN PRINCIPLE
    ==============================================================================

    All FlextProtocols are @runtime_checkable, which means:

    1. **Method Signatures Matter**: Classes satisfy protocols by implementing
    required methods with correct signatures, not by explicit inheritance

    2. **isinstance() Works**: isinstance(obj, FlextProtocols.Service) returns True
    if obj implements all required methods with correct signatures

    3. **Duck Typing Philosophy**: "If it walks like a duck and quacks like a duck,
    it's a duck" - structural typing instead of nominal typing

    4. **Metaclass Conflicts Prevented**: @runtime_checkable protocols don't use
    ProtocolMeta with service metaclasses, avoiding inheritance conflicts

    5. **Type Safety**: Full mypy/pyright type checking without inheritance

    Example of structural typing:
    class UserService:
        '''Satisfies FlextProtocols.Service through method implementation.'''
        def execute(self, command: Command) -> FlextResult:
            '''Required method - protocol compliance verified.'''
            pass

    service = UserService()
    # ✅ isinstance(service, FlextProtocols.Service) → True (duck typing!)

    ==============================================================================
    PROTOCOL HIERARCHY (5 LAYERS)
    ==============================================================================

    **Layer 0: Foundation Protocols** (Core building blocks)
    - HasModelDump - Pydantic model serialization
    - HasModelFields - Model field access
    - HasResultValue - Result-like object interface
    - HasValidateCommand - CQRS command validation
    - HasInvariants - DDD aggregate invariant checking
    - HasTimestamps - Audit timestamp tracking
    - HasHandlerType - Handler type identification
    - Configurable - Component configuration interface

    **Layer 0.5: Circular Import Prevention Protocols**
    - ResultProtocol[T] - Result type interface (prevents circular imports)
    - ConfigProtocol - Configuration interface (prevents circular imports)
    - ModelProtocol - Model type interface (prevents circular imports)

    **Layer 1: Domain Protocols** (Business logic interfaces)
    - Service[T_co] - Base domain service interface
    * execute() - Main operation
    * validate_business_rules() - Business rule validation
    * is_valid() - State validation
    * get_service_info() - Service metadata
    - Repository[T_contra] - Data access interface
    * get_by_id(entity_id) - Entity retrieval
    * save(entity) - Entity persistence
    * delete(entity_id) - Entity deletion
    * find_all() - Entity enumeration

    **Layer 2: Application Protocols** (CQRS patterns)
    - Handler[TInput, TOutput] - Command/Query handler interface
    * handle(message) - Process message
    * validate_command(command) - Command validation
    * validate_query(query) - Query validation
    * can_handle(message_type) - Handler capability check
    - CommandBus - Command routing and execution
    * register_handler(...) - Register handler for command
    * execute(command) - Execute command
    - Middleware - Processing pipeline
    * process(command, next_handler) - Middleware processing

    **Layer 3: Infrastructure Protocols** (External integrations)
    - LoggerProtocol - Logging interface
    * log(level, message, context)
    * debug(), info(), warning(), error()
    - Connection - External system connection
    * test_connection()
    * get_connection_string()
    * close_connection()

    **Layer 4: Extensions** (Ecosystem extensions)
    - PluginContext - Plugin execution context
    * config - Configuration data
    * runtime_id - Runtime identification
    - Observability - Metrics and monitoring
    * record_metric(name, value, tags)
    * log_event(level, message, context)

    ==============================================================================
    CORE PRINCIPLES (4 FUNDAMENTAL RULES)
    ==============================================================================

    **1. Protocols in flext-core are ONLY those used within flext-core**
    - No unnecessary protocols for other projects
    - Domain-specific protocols live in their respective projects
    - flext-ldap has FlextLdapProtocols, flext-cli has FlextCliProtocols, etc.

    **2. Domain-specific protocols live in their respective projects**
    - Each project extends FlextProtocols with domain-specific extensions
    - Example: FlextLdapProtocols.Ldap.LdapConnection extends FlextProtocols.Service
    - Allows type-safe domain-specific interface definitions

    **3. Protocol inheritance creates logical hierarchies**
    - HasModelFields extends HasModelDump (adds model_fields attribute)
    - ModelProtocol extends HasModelDump (adds validation methods)
    - Reduces duplication, improves maintainability
    - Inheritance prevents circular imports between core modules

    **4. All protocols are @runtime_checkable for isinstance() validation**
    - isinstance(obj, FlextProtocols.Service) validates structural compliance
    - Used for runtime type checking and validation
    - Enables duck typing without metaclass conflicts

    ==============================================================================
    EXTENSION PATTERN - HOW DOMAIN LIBRARIES USE FLEXTPROTOCOLS
    ==============================================================================

    Domain libraries extend FlextProtocols with domain-specific protocols:

    **Example 1: LDAP Domain Library**
    class FlextLdapProtocols(FlextProtocols):
        class Ldap:
            class LdapConnectionProtocol(FlextProtocols.Service):
                '''LDAP-specific connection service.'''
                def bind(self, username: str, password: str) -> FlextResult:
                    '''Bind to LDAP server.'''
                    ...
            class LdapSearchRepositoryProtocol(FlextProtocols.Repository[LdapEntry]):
                '''LDAP search operations.'''
                def search(self, base_dn: str, filter_str: str):
                    '''Search LDAP directory.'''
                    ...

    **Example 2: CLI Domain Library**
    class FlextCliProtocols(FlextProtocols):
        class Cli:
            class CommandProtocol(FlextProtocols.Handler[CliArgs, CliOutput]):
                '''CLI command handler.'''
                def run(self, args: CliArgs) -> FlextResult[CliOutput]:
                    '''Execute CLI command.'''
                    ...

    **Example 3: Auth Domain Library**
    class FlextAuthProtocols(FlextProtocols):
        class Auth:
            class UserServiceProtocol(FlextProtocols.Service):
                '''User management service.'''
                def authenticate(self, username: str, pwd: str) -> FlextResult[User]:
                    '''Authenticate user.'''
                    ...

    ==============================================================================
    INTEGRATION POINTS WITH FLEXT ARCHITECTURE
    ==============================================================================

    **FlextResult Integration**:
    - All result-returning methods defined with FlextResult[T] return type
    - Enables railway pattern error handling throughout ecosystem
    - Type-safe success/failure composition

    **FlextContainer Integration**:
    - ServiceLocator protocol for dependency injection
    - Enables type-safe service registration and retrieval
    - Global container singleton access

    **FlextService Integration**:
    - Base domain service implementation follows Service protocol
    - Methods: execute(), validate_business_rules(), get_service_info()
    - Type-safe service lifecycle management

    **FlextModels Integration**:
    - Domain models satisfy HasModelDump, HasModelFields, ModelProtocol
    - Pydantic v2 integration through model_dump, model_fields, validate
    - Type-safe DDD pattern implementation

    **FlextLogger Integration**:
    - LoggerProtocol for structured logging
    - Context propagation through log methods
    - Level-based context binding

    ==============================================================================
    PROTOCOL INHERITANCE PATTERNS
    ==============================================================================

    **Pattern 1: Foundation Extension**
    HasModelDump (base)
        ↓ extends
    HasModelFields (adds model_fields)
        ↓ extends
    ModelProtocol (adds validation methods)

    **Pattern 2: Service Specialization**
    FlextProtocols.Service (generic)
        ↓ extended by
    FlextLdapProtocols.Ldap.LdapService (LDAP-specific)
        ↓ extended by
    FlextTapLdapProtocols.Tap.LdapTapService (Singer tap-specific)

    **Pattern 3: Repository Hierarchy**
    FlextProtocols.Repository[T] (generic CRUD)
        ↓ extended by
    FlextLdapProtocols.LdapRepository (LDAP search)
        ↓ extended by
    FlextTapLdapProtocols.LdapTapRepository (Singer integration)

    ==============================================================================
    CIRCULAR IMPORT PREVENTION
    ==============================================================================

    Three specialized protocols prevent circular imports between core modules:

    **ResultProtocol[T]**: Prevents circular imports between result.py and
    config.py/models.py/utilities.py by providing result interface without
    importing concrete FlextResult class

    **ConfigProtocol**: Prevents circular imports between config.py and other
    modules by providing config interface without importing FlextConfig

    **ModelProtocol**: Prevents circular imports between models.py and config.py
    by providing model interface without importing FlextModels

    These protocols enable type checking without concrete imports, breaking
    circular dependency chains.

    ==============================================================================
    USAGE PATTERNS WITH EXAMPLES
    ==============================================================================

    **Pattern 1: Type Checking with isinstance()**
    service = UserService()
    if isinstance(service, FlextProtocols.Service):
        # Safe to call service.execute(), validate_business_rules(), etc.
        result = service.execute()

    **Pattern 2: Function Signature with Protocol**
    def process_service(svc: FlextProtocols.Service) -> FlextResult:
        '''Process any domain service.'''
        return svc.execute()

    **Pattern 3: Generic Protocol Usage**
    def query_repository(repo: FlextProtocols.Repository[User]) -> FlextResult:
        '''Query any repository.'''
        users = repo.find_all()
        # NOTE: This example shows conceptual usage only.
        # In actual code, import FlextResult from flext_core.result
        return FlextResult.ok(users)  # Illustrative - requires import

    **Pattern 4: Handler Registration**
    bus: FlextProtocols.CommandBus
    handler: FlextProtocols.Handler[CreateUserCmd, User]
    bus.register_handler(CreateUserCmd, handler)
    result = bus.execute(CreateUserCmd(...))

    **Pattern 5: Domain Library Extension**
    class FlextAuthProtocols(FlextProtocols):
        class Auth:
            class UserService(FlextProtocols.Service):
                def authenticate(self, creds) -> FlextResult[AuthToken]:
                    ...

    **Pattern 6: Logger with Context**
    logger: FlextProtocols.LoggerProtocol
    logger.info("user created", context={"user_id": "123"})
    logger.error("auth failed", context={"reason": "invalid_password"})

    **Pattern 7: Protocol Composition**
    class AuditedService(FlextProtocols.Service, FlextProtocols.HasTimestamps):
        '''Service with audit timestamps.'''
        created_at: str  # From HasTimestamps
        def execute(self):  # From Service
            ...

    **Pattern 8: Circular Import Prevention**
    # In config.py, use ResultProtocol instead of importing FlextResult
    # def validate(self) -> ResultProtocol[bool]:
    #     return FlextResult[bool].ok(True)

    ==============================================================================
    PRODUCTION-READY CHARACTERISTICS
    ==============================================================================

    ✅ Type Safety: @runtime_checkable protocols work with mypy/pyright strict
    ✅ Circular Imports: Specialized protocols prevent import cycles
    ✅ Structural Typing: Duck typing validation without metaclass conflicts
    ✅ Extensibility: Domain libraries extend with domain-specific protocols
    ✅ Integration: All core implementations follow protocol definitions
    ✅ No Breaking Changes: Protocol additions backward compatible
    ✅ Documentation: Each protocol documents use cases and extensions
    ✅ Performance: isinstance() checks optimized for runtime use

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
    # FOUNDATION LAYER (Layer 0) - Core protocols used within flext-core
    # =========================================================================

    @runtime_checkable
    class Monad(Protocol, Generic[T_co]):
        """Protocol for monadic types enabling functional composition.

        Monads provide bind/flat_map operations for composable error handling.
        FlextResult[T] implements this protocol for railway-oriented patterns.

        Used in: result.py (FlextResult monad operations)

        Composition Example:
            result = FlextResult[User].ok(user)
            composed = (
                result
                .flat_map(save_to_db)
                .map(format_response)
                .map_error(handle_error)
            )
        """

        @abstractmethod
        def bind(
            self, func: Callable[[T_co], FlextProtocols.Monad[T]]
        ) -> FlextProtocols.Monad[T]:
            """Monadic bind operation (flat_map equivalent).

            Args:
                func: Function returning a monad

            Returns:
                Monad[T]: Result of applying function to wrapped value

            """
            ...

        @abstractmethod
        def flat_map(
            self, func: Callable[[T_co], FlextProtocols.Monad[T]]
        ) -> FlextProtocols.Monad[T]:
            """Monadic flat_map operation (bind equivalent).

            Args:
                func: Function returning a monad

            Returns:
                Monad[T]: Result of applying function to wrapped value

            """
            ...

        @abstractmethod
        def map(self, func: Callable[[T_co], T]) -> FlextProtocols.Monad[T]:
            """Functor map operation (transform wrapped value).

            Args:
                func: Function transforming wrapped value

            Returns:
                Monad[T]: Monad with transformed value

            """
            ...

        def filter(
            self, predicate: Callable[[T_co], bool]
        ) -> FlextProtocols.Monad[T_co]:
            """Filter monad based on predicate (optional)."""
            ...

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for objects with model_dump method (Pydantic compatibility).

        Base protocol for Pydantic-like model serialization. Extended by other
        protocols to add additional capabilities.

        Extensions:
            - HasModelFields: Adds model_fields attribute
            - ModelProtocol: Adds validation methods
        """

        def model_dump(self, mode: str = "python") -> dict[str, object]:
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

        Inheritance: HasModelDump → HasModelFields
        """

        model_fields: dict[str, object]

    @runtime_checkable
    class HasResultValue(Protocol, Generic[T]):
        """Protocol for FlextResult-like objects.

        Minimal protocol for result types with value and success status.
        Used for type checking without importing FlextResult (breaks circular imports).

        Used in: processors.py (middleware processing)
        """

        value: T
        is_success: bool

    @runtime_checkable
    class HasValidateCommand(Protocol):
        """Protocol for commands with validate_command method.

        CQRS pattern support for command validation before execution.

        Used in: bus.py (command validation)
        """

        def validate_command(self) -> FlextResult[bool]:
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
    class TypeValidator(Protocol):
        """Protocol for type validation with constraints.

        Provides type-safe validation with constraints for generic types.
        Used by FlextContainer for runtime type checking.

        Used in: container.py (type validation in get_typed)
        """

        @abstractmethod
        def validate_type(self, value: T, expected_type: type[T]) -> FlextResult[T]:
            """Validate value matches expected type.

            Args:
                value: Value to validate
                expected_type: Expected type for validation

            Returns:
                FlextResult[T]: Validated value or error

            """
            ...

        @abstractmethod
        def is_valid_type(self, value: T, expected_type: type[T]) -> bool:
            """Check if value is valid for expected type.

            Args:
                value: Value to check
                expected_type: Expected type

            Returns:
                bool: True if valid, False otherwise

            """
            ...

    @runtime_checkable
    class ServiceRegistry(Protocol):
        """Protocol for service registry with typed access.

        Provides type-safe service registration and retrieval with
        support for factory functions and dependency injection.

        Used in: container.py (FlextContainer service management)
        """

        @abstractmethod
        def register_service(self, name: str, service: object) -> FlextResult[bool]:
            """Register service with given name.

            Args:
                name: Service identifier
                service: Service instance

            Returns:
                FlextResult[bool]: Success or registration error

            """
            ...

        @abstractmethod
        def get_service(self, name: str) -> FlextResult[object]:
            """Retrieve registered service.

            Args:
                name: Service identifier

            Returns:
                FlextResult[object]: Service instance or error

            """
            ...

        @abstractmethod
        def has_service(self, name: str) -> bool:
            """Check if service is registered.

            Args:
                name: Service identifier

            Returns:
                bool: True if registered

            """
            ...

    @runtime_checkable
    class FactoryProvider(Protocol, Generic[T]):
        """Protocol for factory functions creating instances.

        Provides factory-based service creation with lazy instantiation.
        Used by FlextContainer for factory registration patterns.

        Used in: container.py (with_factory, factory retrieval)
        """

        @abstractmethod
        def create_instance(self) -> FlextResult[T]:
            """Create new instance using factory.

            Returns:
                FlextResult[T]: Created instance or error

            """
            ...

    @runtime_checkable
    class Configurable(Protocol):
        """Protocol for configurable components.

        Infrastructure protocol for components that can be configured with
        dictionary-based settings. Returns FlextResult for error handling.

        Used in: container.py (FlextContainer configuration)

        Note: Replaces duplicate Configurable in Infrastructure namespace.
        """

        def configure(self, config: dict[str, object]) -> FlextResult[bool]:
            """Configure component with provided settings.

            Args:
                config: Configuration dictionary

            Returns:
                FlextResult[bool]: Success if configured, failure with error details

            """
            ...

    # =========================================================================
    # DOMAIN LAYER (Layer 1) - Service and Repository protocols
    # =========================================================================

    @runtime_checkable
    class Service(Protocol, Generic[T]):
        """Base domain service protocol.

        Provides the foundation for all domain services in the FLEXT ecosystem.
        Domain libraries extend this protocol with specific service operations.

        Domain Extensions:
            - FlextLdapProtocols.Ldap.LdapConnectionProtocol
            - FlextAuthProtocols.Auth.ServiceProtocol
            - FlextGrpcProtocols.Grpc.ServerProtocol
        """

        @abstractmethod
        def execute(self) -> FlextResult[T]:
            """Execute the main domain operation.

            Returns:
                FlextResult[T]: Success with domain result or failure

            """
            ...

        def is_valid(self) -> bool:
            """Check if the domain service is in a valid state.

            Returns:
                bool: True if valid, False otherwise

            """
            ...

        def validate_business_rules(self) -> FlextResult[bool]:
            """Validate business rules for the domain service.

            Returns:
                FlextResult[bool]: Success if valid, failure with error details

            """
            ...

        def validate_config(self) -> FlextResult[bool]:
            """Validate service configuration.

            Returns:
                FlextResult[bool]: Success if valid, failure with error details

            """
            ...

        def get_service_info(self) -> dict[str, FlextTypes.JsonValue]:
            """Get service information and metadata.

            Returns:
                dict[str, object]: Service information dictionary

            """
            ...

    @runtime_checkable
    class Repository(Protocol, Generic[T]):
        """Base repository protocol for data access.

        Provides the foundation for repository implementations following
        Domain-Driven Design patterns. Domain libraries extend with specific
        repository operations.

        Domain Extensions:
            - FlextLdapProtocols should define LDAP-specific repositories
            - FlextAuthProtocols should define user/session repositories
        """

        @abstractmethod
        def get_by_id(self, entity_id: str) -> FlextResult[T]:
            """Retrieve an aggregate using the standardized identity lookup."""
            ...

        @abstractmethod
        def save(self, entity: T) -> FlextResult[bool]:
            """Persist an entity following modernization consistency rules."""
            ...

        @abstractmethod
        def delete(self, entity_id: str) -> FlextResult[bool]:
            """Delete an entity while respecting modernization invariants."""
            ...

        @abstractmethod
        def find_all(self) -> FlextResult[list[T]]:
            """Enumerate entities for modernization-aligned queries."""
            ...

    @runtime_checkable
    class ExecutableService(Protocol, Generic[T]):
        """Protocol for services with enhanced execution capabilities.

        Extends Service protocol with execution context management,
        timeout support, and error handling patterns.

        Used in: service.py (FlextService enhanced execution)
        """

        @abstractmethod
        def execute_operation(self) -> FlextResult[T]:
            """Execute operation with full validation and context.

            Returns:
                FlextResult[T]: Operation result with rich error context

            """
            ...

        @abstractmethod
        def execute_with_validation(self) -> FlextResult[T]:
            """Execute with comprehensive business rule validation.

            Returns:
                FlextResult[T]: Result or validation error

            """
            ...

    @runtime_checkable
    class ContextAware(Protocol):
        """Protocol for context-aware domain operations.

        Enables services to manage execution context including user,
        correlation IDs, and operation metadata.

        Used in: service.py (FlextService context management)
        """

        @abstractmethod
        def with_correlation_id(self, correlation_id: str) -> FlextResult[bool]:
            """Set correlation ID for distributed tracing.

            Args:
                correlation_id: Unique trace identifier

            Returns:
                FlextResult[bool]: Success or context error

            """
            ...

        @abstractmethod
        def with_user_context(self, user_id: str) -> FlextResult[bool]:
            """Set user context for audit trail.

            Args:
                user_id: User identifier

            Returns:
                FlextResult[bool]: Success or context error

            """
            ...

        @abstractmethod
        def get_context(self) -> dict[str, FlextTypes.JsonValue]:
            """Retrieve current execution context.

            Returns:
                dict: Context dictionary with all identifiers

            """
            ...

    @runtime_checkable
    class TimeoutSupport(Protocol):
        """Protocol for operations with timeout enforcement.

        Enables services to execute operations with timeout constraints
        and cancellation support.

        Used in: service.py (FlextService timeout handling)
        """

        @abstractmethod
        def with_timeout(self, seconds: float) -> FlextResult[bool]:
            """Set execution timeout in seconds.

            Args:
                seconds: Timeout duration

            Returns:
                FlextResult[bool]: Success or configuration error

            """
            ...

        @abstractmethod
        def get_remaining_time(self) -> float:
            """Get remaining execution time.

            Returns:
                float: Seconds remaining before timeout

            """
            ...

    # =========================================================================
    # APPLICATION LAYER (Layer 2) - Command/Query patterns
    # =========================================================================

    @runtime_checkable
    class Handler(
        Protocol, Generic[TInput_Handler_Protocol_contra, TResult_Handler_Protocol]
    ):
        """Application handler protocol for CQRS patterns.

        Used in: handlers.py (FlextHandlers implementation)

        Provides standardized interface for command and query handlers with
        validation and execution methods. Uses properly-varianced TypeVars
        per Pyright requirement: input contravariant, result invariant.
        """

        @abstractmethod
        def handle(
            self, message: TInput_Handler_Protocol_contra
        ) -> FlextResult[TResult_Handler_Protocol]:
            """Handle the message and return result.

            Args:
                message: The input message to process

            Returns:
                FlextResult[TResult_Handler_Protocol]: Success with result or failure

            """
            ...

        def __call__(
            self, input_data: TInput_Handler_Protocol_contra
        ) -> FlextResult[TResult_Handler_Protocol]:
            """Process input and return a FlextResult containing the output."""
            ...

        def can_handle(self, message_type: type) -> bool:
            """Check if handler can process this message type."""
            ...

        def execute(
            self, message: TInput_Handler_Protocol_contra
        ) -> FlextResult[TResult_Handler_Protocol]:
            """Execute the handler with the given message."""
            ...

        def validate_command(
            self, command: TInput_Handler_Protocol_contra
        ) -> FlextResult[bool]:
            """Validate a command message."""
            ...

        def validate(self, _data: TInput_Handler_Protocol_contra) -> FlextResult[bool]:
            """Validate input before processing."""
            ...

        def validate_query(
            self, query: TInput_Handler_Protocol_contra
        ) -> FlextResult[bool]:
            """Validate a query message."""
            ...

        @property
        def handler_name(self) -> str:
            """Get the handler name."""
            ...

        @property
        def mode(self) -> str:
            """Get the handler mode (command/query)."""
            ...

    @runtime_checkable
    class CommandBus(Protocol):
        """Protocol for command bus routing and execution."""

        @overload
        def register_handler(
            self,
            handler: FlextTypes.HandlerCallableType,
            /,
        ) -> FlextResult[bool]: ...

        @overload
        def register_handler(
            self,
            command_type: type,
            handler: FlextTypes.HandlerCallableType,
            /,
        ) -> FlextResult[bool]: ...

        def register_handler(
            self,
            command_type_or_handler: type | FlextTypes.HandlerCallableType,
            handler: FlextTypes.HandlerCallableType | None = None,
            /,
        ) -> FlextResult[bool]:
            """Register command handler."""
            ...

        def execute(self, command: T) -> FlextResult[T]:
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

    @runtime_checkable
    class MessageValidator(Protocol):
        """Protocol for command/query message validation.

        Provides comprehensive validation for CQRS messages before
        execution. Used by handlers to validate input.

        Used in: handlers.py (FlextHandlers message validation)
        """

        @abstractmethod
        def validate_message(self, message: object) -> FlextResult[bool]:
            """Validate message structure and content.

            Args:
                message: Message to validate

            Returns:
                FlextResult[bool]: Success or validation error details

            """
            ...

        @abstractmethod
        def get_validation_errors(self) -> list[str]:
            """Get detailed validation error messages.

            Returns:
                list[str]: List of validation error descriptions

            """
            ...

    @runtime_checkable
    class MetricsCollector(Protocol):
        """Protocol for collecting handler execution metrics.

        Tracks handler performance including execution time, error rates,
        and success rates for observability.

        Used in: handlers.py (FlextHandlers metrics collection)
        """

        @abstractmethod
        def record_execution(
            self,
            handler_name: str,
            duration_ms: float,
            *,
            success: bool,
        ) -> FlextResult[bool]:
            """Record handler execution metrics.

            Args:
                handler_name: Name of executed handler
                duration_ms: Execution time in milliseconds
                success: Whether execution succeeded

            Returns:
                FlextResult[bool]: Success or recording error

            """
            ...

        @abstractmethod
        def get_metrics(self) -> dict[str, FlextTypes.JsonValue]:
            """Get collected metrics summary.

            Returns:
                dict: Metrics including execution counts and durations

            """
            ...

    @runtime_checkable
    class ExecutionContextManager(Protocol):
        """Protocol for managing handler execution context.

        Manages execution context lifecycle including correlation IDs,
        user context, and operation tracking.

        Used in: handlers.py (FlextHandlers context management)
        """

        @abstractmethod
        def setup_context(self) -> FlextResult[bool]:
            """Set up execution context before handler execution.

            Returns:
                FlextResult[bool]: Success or setup error

            """
            ...

        @abstractmethod
        def cleanup_context(self) -> FlextResult[bool]:
            """Clean up execution context after handler execution.

            Returns:
                FlextResult[bool]: Success or cleanup error

            """
            ...

    @runtime_checkable
    class CacheManager(Protocol):
        """Protocol for handler execution result caching.

        Manages caching of handler results with TTL and invalidation
        support for performance optimization.

        Used in: bus.py (FlextBus command result caching)
        """

        @abstractmethod
        def put(self, key: str, value: object, ttl_seconds: int) -> FlextResult[bool]:
            """Store value in cache with TTL.

            Args:
                key: Cache key identifier
                value: Value to cache
                ttl_seconds: Time to live in seconds

            Returns:
                FlextResult[bool]: Success or cache error

            """
            ...

        @abstractmethod
        def get(self, key: str) -> FlextResult[object | None]:
            """Retrieve value from cache.

            Args:
                key: Cache key identifier

            Returns:
                FlextResult[T | None]: Cached value or None if expired

            """
            ...

        @abstractmethod
        def invalidate(self, key: str) -> FlextResult[bool]:
            """Invalidate cache entry.

            Args:
                key: Cache key identifier

            Returns:
                FlextResult[bool]: Success or invalidation error

            """
            ...

    @runtime_checkable
    class MiddlewareChain(Protocol):
        """Protocol for composable middleware pipeline.

        Provides middleware composition and execution chain management
        for processing commands/queries.

        Used in: bus.py (FlextBus middleware processing)
        """

        @abstractmethod
        def add_middleware(
            self, middleware: FlextProtocols.Middleware
        ) -> FlextResult[bool]:
            """Add middleware to processing chain.

            Args:
                middleware: Middleware to add

            Returns:
                FlextResult[bool]: Success or chain error

            """
            ...

        @abstractmethod
        def execute_chain(
            self,
            command_or_query: object,
            final_handler: Callable[[object], FlextResult[object]],
        ) -> FlextResult[object]:
            """Execute full middleware chain.

            Args:
                command_or_query: Message to process
                final_handler: Final handler in chain

            Returns:
                FlextResult: Result from processing chain

            """
            ...

    @runtime_checkable
    class RegistrationTracker(Protocol):
        """Protocol for tracking handler registrations.

        Tracks handler registration changes, updates, and provides
        registration history for debugging and auditing.

        Used in: registry.py (FlextRegistry tracking)
        """

        @abstractmethod
        def on_registered(
            self, message_type: type, handler: object
        ) -> FlextResult[bool]:
            """Track handler registration event.

            Args:
                message_type: Message type being handled
                handler: Registered handler

            Returns:
                FlextResult[bool]: Success or tracking error

            """
            ...

        @abstractmethod
        def get_registration_history(
            self,
        ) -> list[dict[str, FlextTypes.JsonValue]]:
            """Get registration event history.

            Returns:
                list: Registration events with timestamps

            """
            ...

    # =========================================================================
    # INFRASTRUCTURE LAYER (Layer 3-4) - External integrations
    # =========================================================================

    @runtime_checkable
    class CircuitBreaker(Protocol):
        """Protocol for circuit breaker resilience pattern.

        Implements circuit breaker pattern to prevent cascading failures
        by stopping requests to failing services.

        Used in: dispatcher.py (FlextDispatcher resilience)
        """

        @abstractmethod
        def call(self, operation: Callable[[], object]) -> FlextResult[object]:
            """Execute operation through circuit breaker.

            Args:
                operation: Callable to execute

            Returns:
                FlextResult: Result or circuit breaker error

            """
            ...

        @abstractmethod
        def get_state(self: object) -> str:
            """Get circuit breaker state (closed/open/half-open).

            Returns:
                str: Current state name

            """
            ...

    @runtime_checkable
    class RateLimiter(Protocol):
        """Protocol for rate limiting operations.

        Enforces rate limits on operations to prevent overload and
        ensure fair resource sharing.

        Used in: dispatcher.py (FlextDispatcher rate limiting)
        """

        @abstractmethod
        def allow_request(self) -> FlextResult[bool]:
            """Check if request is allowed under rate limit.

            Returns:
                FlextResult[bool]: True if allowed, False if rate limited

            """
            ...

        @abstractmethod
        def get_remaining_quota(self: object) -> int:
            """Get remaining requests in current period.

            Returns:
                int: Remaining request quota

            """
            ...

    @runtime_checkable
    class RetryPolicy(Protocol):
        """Protocol for retry policy configuration.

        Configures retry behavior including max attempts, backoff
        strategies, and retry conditions.

        Used in: dispatcher.py (FlextDispatcher retry logic)
        """

        @abstractmethod
        def should_retry(self, attempt: int, error: object) -> bool:
            """Determine if operation should be retried.

            Args:
                attempt: Attempt number
                error: Exception that occurred

            Returns:
                bool: True if should retry

            """
            ...

        @abstractmethod
        def get_delay_ms(self, attempt: int) -> float:
            """Get delay before next retry in milliseconds.

            Args:
                attempt: Attempt number

            Returns:
                float: Delay in milliseconds

            """
            ...

    @runtime_checkable
    class TimeoutEnforcer(Protocol):
        """Protocol for enforcing operation timeouts.

        Enforces strict timeout constraints on operations to prevent
        indefinite hangs and resource exhaustion.

        Used in: dispatcher.py (FlextDispatcher timeout enforcement)
        """

        @abstractmethod
        def execute_with_timeout(
            self, operation: Callable[[], object], timeout_ms: float
        ) -> FlextResult[object]:
            """Execute operation with timeout constraint.

            Args:
                operation: Callable to execute
                timeout_ms: Timeout in milliseconds

            Returns:
                FlextResult: Result or timeout error

            """
            ...

        @abstractmethod
        def get_timeout_errors(self: object) -> list[str]:
            """Get list of recent timeout errors.

            Returns:
                list[str]: Timeout error messages

            """
            ...

    @runtime_checkable
    class ObservabilityCollector(Protocol):
        """Protocol for collecting operation metrics and traces.

        Collects metrics, traces, and performance data for observability
        across distributed systems.

        Used in: dispatcher.py (FlextDispatcher observability)
        """

        @abstractmethod
        def start_trace(self, operation_name: str) -> str:
            """Start operation trace.

            Args:
                operation_name: Name of operation

            Returns:
                str: Trace identifier

            """
            ...

        @abstractmethod
        def record_metric(
            self, metric_name: str, value: float, tags: dict[str, str] | None
        ) -> FlextResult[bool]:
            """Record metric value with tags.

            Args:
                metric_name: Metric identifier
                value: Metric value
                tags: Optional metric tags

            Returns:
                FlextResult[bool]: Success or collection error

            """
            ...

    @runtime_checkable
    class BatchProcessor(Protocol):
        """Protocol for batch processing operations.

        Processes items in batches for performance optimization and
        efficient resource utilization.

        Used in: dispatcher.py, registry.py (batch operations)
        """

        @abstractmethod
        def process_batch(self, items: list[object]) -> FlextResult[list[object]]:
            """Process batch of items.

            Args:
                items: Items to process

            Returns:
                FlextResult[list[T]]: Processed items or error

            """
            ...

        @abstractmethod
        def get_batch_size(self: object) -> int:
            """Get optimal batch size.

            Returns:
                int: Batch size

            """
            ...

    @runtime_checkable
    class EventPublisher(Protocol):
        """Protocol for publishing events.

        Publishes domain events for event-driven architecture and
        inter-service communication.

        Used in: bus.py (FlextBus event publishing)
        """

        @abstractmethod
        def publish_event(self, event: object) -> FlextResult[bool]:
            """Publish domain event.

            Args:
                event: Event to publish

            Returns:
                FlextResult[bool]: Success or publishing error

            """
            ...

        @abstractmethod
        def subscribe(
            self, event_type: type, handler: Callable[[object], object]
        ) -> FlextResult[bool]:
            """Subscribe to event type.

            Args:
                event_type: Type of event to subscribe to
                handler: Event handler callable

            Returns:
                FlextResult[bool]: Success or subscription error

            """
            ...

    @runtime_checkable
    class ConfigurationValidator(Protocol):
        """Protocol for configuration validation.

        Validates configuration settings and ensures they meet
        constraints and requirements.

        Used in: config.py (FlextConfig validation)
        """

        @abstractmethod
        def validate_config(self, config: dict[str, object]) -> FlextResult[bool]:
            """Validate configuration dictionary.

            Args:
                config: Configuration to validate

            Returns:
                FlextResult[bool]: Success or validation error

            """
            ...

        @abstractmethod
        def get_validation_rules(self: object) -> dict[str, object]:
            """Get validation rules.

            Returns:
                dict: Validation rules for each config key

            """
            ...

    @runtime_checkable
    class DynamicUpdater(Protocol):
        """Protocol for dynamic configuration updates.

        Supports runtime configuration updates without restarting
        services, enabling hot-reload patterns.

        Used in: config.py (FlextConfig dynamic updates)
        """

        @abstractmethod
        def update_config(self, key: str, value: object) -> FlextResult[bool]:
            """Update configuration dynamically.

            Args:
                key: Configuration key
                value: New value

            Returns:
                FlextResult[bool]: Success or update error

            """
            ...

        @abstractmethod
        def get_update_history(
            self,
        ) -> list[dict[str, FlextTypes.JsonValue]]:
            """Get configuration update history.

            Returns:
                list: Update events with timestamps

            """
            ...

    @runtime_checkable
    class SingletonProvider(Protocol):
        """Protocol for singleton pattern implementation.

        Provides singleton pattern with lazy initialization and
        thread-safe instance management.

        Used in: config.py (FlextConfig singleton pattern)
        """

        @abstractmethod
        def get_instance(self) -> FlextResult[object]:
            """Get singleton instance.

            Returns:
                FlextResult[T]: Singleton instance or error

            """
            ...

        @abstractmethod
        def reset_instance(self) -> FlextResult[bool]:
            """Reset singleton instance (testing only).

            Returns:
                FlextResult[bool]: Success or reset error

            """
            ...

    @runtime_checkable
    class ContextBinder(Protocol):
        """Protocol for binding context to logs.

        Binds contextual information (correlation IDs, user info) to
        all log messages for distributed tracing.

        Used in: loggings.py (FlextLogger context binding)
        """

        @abstractmethod
        def bind_context(self, context: dict[str, object]) -> FlextResult[bool]:
            """Bind context dictionary to logger.

            Args:
                context: Context to bind

            Returns:
                FlextResult[bool]: Success or binding error

            """
            ...

        @abstractmethod
        def get_bound_context(self: object) -> dict[str, object]:
            """Get currently bound context.

            Returns:
                dict: Bound context data

            """
            ...

    @runtime_checkable
    class PerformanceTracker(Protocol):
        """Protocol for tracking performance metrics.

        Tracks performance metrics including latencies, throughput,
        and resource utilization.

        Used in: loggings.py (FlextLogger performance tracking)
        """

        @abstractmethod
        def record_latency(
            self, operation_name: str, duration_ms: float
        ) -> FlextResult[bool]:
            """Record operation latency.

            Args:
                operation_name: Name of operation
                duration_ms: Duration in milliseconds

            Returns:
                FlextResult[bool]: Success or recording error

            """
            ...

        @abstractmethod
        def get_performance_stats(self: object) -> dict[str, object]:
            """Get performance statistics summary.

            Returns:
                dict: Performance metrics (avg, min, max, p95, p99)

            """
            ...

    # =========================================================================
    # USED PROTOCOLS - Actually referenced in real code
    # =========================================================================

    @runtime_checkable
    class Command(Protocol):
        """Protocol for CQRS command objects.

        Commands represent intent to change system state. Implementations
        must define command type and optional validation.

        Used in: CommandBus, Dispatcher, all command handlers
        """

        command_type: str

        def get_command_metadata(self: object) -> dict[str, object]:
            """Get command metadata for routing and tracing."""
            ...

    @runtime_checkable
    class Decorator(Protocol, Generic[T]):
        """Protocol for decorator pattern implementations.

        Decorators wrap objects to add behavior. Must implement component
        interface while delegating to wrapped object.

        Used in: Cross-cutting concerns, aspect-oriented patterns
        """

        wrapped_component: T

        def decorate(self, component: T) -> T:
            """Decorate a component."""
            ...

    @runtime_checkable
    class Constants(Protocol):
        """Protocol for constants holder objects.

        Centralizes application constants. Satisfy FlextConstants
        structural interface for type-safe constant access.

        Used in: Configuration, domain constants
        """

        def get_constant(self, name: str) -> FlextTypes.JsonValue:
            """Retrieve constant by name."""
            ...

    @runtime_checkable
    class Logger(Protocol):
        """Protocol for logger implementations.

        Logger satisfies both this protocol and LoggerProtocol.
        Provides structured logging with context propagation.

        Used in: All logging operations
        """

        def get_logger_name(self) -> str:
            """Get logger name/identifier."""
            ...

        def set_log_level(self, level: str) -> None:
            """Set logging level dynamically."""
            ...

        def get_log_level(self) -> str:
            """Get current logging level."""
            ...

    @runtime_checkable
    class HandlerRegistry(Protocol):
        """Protocol for handler registry implementations.

        Registry maintains handlers for commands/queries. Enables
        dynamic handler registration and discovery.

        Used in: CommandBus, Dispatcher, plugin systems
        """

        def register(
            self, message_type: type, handler: FlextTypes.HandlerCallableType
        ) -> FlextResult[bool]:
            """Register handler for message type."""
            ...

        def get_handler(
            self, message_type: type
        ) -> FlextResult[FlextTypes.HandlerCallableType]:
            """Get handler for message type."""
            ...

        def list_handlers(self) -> dict[type, FlextTypes.HandlerCallableType]:
            """List all registered handlers."""
            ...


__all__ = [
    "FlextProtocols",  # Main hierarchical protocol architecture
]

# =========================================================================
# PROTOCOL SUMMARY - 40+ protocols across 5 layers
# =========================================================================
# Layer 0: Foundation (9 protocols)
#   - Monad[T], HasModelDump, HasModelFields, HasResultValue, HasValidateCommand
#   - HasInvariants, Configurable, TypeValidator[T], ServiceRegistry[T]
#   - FactoryProvider[T]
#
# Layer 1: Domain (5 protocols)
#   - Service[T_co], Repository[T_contra], ExecutableService[T_co]
#   - ContextAware, TimeoutSupport
#
# Layer 2: Application (10 protocols)
#   - Handler[TInput, TOutput], CommandBus, Middleware
#   - MessageValidator, MetricsCollector, ExecutionContextManager
#   - CacheManager[T_co], MiddlewareChain, RegistrationTracker
#
# Layer 3-4: Infrastructure (16 protocols)
#   - CircuitBreaker, RateLimiter, RetryPolicy, TimeoutEnforcer
#   - ObservabilityCollector, BatchProcessor[T_co], EventPublisher
#   - ConfigurationValidator, DynamicUpdater[T_co], SingletonProvider[T_co]
#   - ContextBinder, PerformanceTracker
#
# Cross-layer/Utility (6 protocols)
#   - Command, Decorator, Constants, Logger, HandlerRegistry
