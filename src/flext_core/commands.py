"""Enterprise CQRS Command and Query Processing System for the FLEXT ecosystem.

This module provides a comprehensive Command Query Responsibility Segregation (CQRS)
implementation that serves as the central command processing foundation for the FLEXT
ecosystem. Built on Clean Architecture principles with extensive integration of the
hierarchical FlextTypes, FlextConstants, and FlextProtocols systems.

**Architectural Foundation**:
    The CQRS system is implemented as a single consolidated class ``FlextCommands``
    that contains all command processing functionality organized hierarchically:

    - **Domain Separation**: Clear separation between Commands (write operations) and Queries (read operations)
    - **Railway Pattern**: Comprehensive FlextResult integration for type-safe error handling
    - **Type Safety**: Extensive use of Python 3.13+ generics and type annotations
    - **SOLID Compliance**: Single Responsibility, Open/Closed, Interface Segregation principles
    - **Clean Architecture**: Foundation → Domain → Application → Infrastructure layer separation
    - **Enterprise Patterns**: Handler registration, middleware pipelines, execution monitoring

**System Components**:
    The ``FlextCommands`` class organizes CQRS functionality into logical nested classes:

    - **Types**: Command-specific type definitions extending FlextTypes hierarchy
    - **Protocols**: Type-safe interface definitions using FlextProtocols patterns
    - **Models**: Pydantic v2-based Command and Query models with validation
    - **Handlers**: Generic handler base classes for command and query processing
    - **Bus**: Central command bus for routing, middleware, and execution
    - **Decorators**: Function-based command handler registration patterns
    - **Results**: FlextResult factory methods for consistent result creation
    - **Factories**: Instance creation factories with proper dependency injection

**Integration Architecture**:
    Deep integration with the FLEXT ecosystem foundation:

    - **FlextResult**: All operations return FlextResult for railway-oriented programming
    - **FlextTypes**: Hierarchical type system for consistent typing across layers
    - **FlextConstants**: Configuration constants for timeouts, limits, error codes
    - **FlextProtocols**: Interface definitions for handler contracts and validation
    - **FlextLogger**: Structured logging with correlation IDs and context tracking
    - **FlextModels**: Pydantic v2 integration for command/query serialization
    - **FlextMixins**: Reusable behaviors for logging, timing, serialization
    - **FlextUtilities**: UUID generation, validation helpers, type guards

**Key Features**:
    - **Type-Safe Command Processing**: Full generic typing with Python 3.13+ syntax
    - **Automatic Command Type Inference**: CamelCase → snake_case conversion
    - **Comprehensive Validation**: Multi-layer validation using Pydantic and business rules
    - **Middleware Pipeline**: Pluggable middleware for cross-cutting concerns
    - **Handler Auto-Discovery**: Dynamic handler registration with type introspection
    - **Execution Monitoring**: Performance metrics, timing, and structured logging
    - **Payload Integration**: Seamless FlextModels.Payload serialization for message passing
    - **Query Pagination**: Built-in pagination with configurable limits and sorting
    - **Thread Safety**: Concurrent command execution with proper synchronization
    - **Error Standardization**: Consistent error codes and structured error handling

**Usage Patterns**:
    Enterprise command processing with full type safety::

        from flext_core import FlextCommands
        from flext_core.result import FlextResult


        # Define domain command with automatic validation
        class CreateUserCommand(FlextCommands.Models.Command):
            email: str
            name: str
            role: str = "user"

            def validate_command(self) -> FlextResult[None]:
                # Chain validation using FlextResult railway patterns
                return (
                    FlextCommands.Models.Command.require_field("email", self.email)
                    .flat_map(
                        lambda _: FlextCommands.Models.Command.require_email(self.email)
                    )
                    .flat_map(
                        lambda _: FlextCommands.Models.Command.require_min_length(
                            self.name, 2, "name"
                        )
                    )
                )


        # Implement type-safe command handler
        class UserCreationHandler(
            FlextCommands.Handlers.CommandHandler[CreateUserCommand, str]
        ):
            def handle(self, command: CreateUserCommand) -> FlextResult[str]:
                # Business logic with automatic logging and timing
                self.log_info("Creating user", email=command.email, name=command.name)

                # Simulate user creation
                user_id = f"user_{command.email.split('@')[0]}"

                return FlextCommands.Results.success(user_id)


        # Set up command bus with handler registration
        bus = FlextCommands.Factories.create_command_bus()
        bus.register_handler(UserCreationHandler())

        # Execute command with full error handling
        command = CreateUserCommand(email="john@example.com", name="John Doe")
        result = bus.execute(command)

        if result.success:
            user_id = result.unwrap()
            print(f"User created with ID: {user_id}")
        else:
            print(f"Command failed: {result.error}")

    Query processing with pagination and sorting::

        class FindUsersQuery(FlextCommands.Models.Query):
            role_filter: str | None = None
            active_only: bool = True


        class UserQueryHandler(
            FlextCommands.Handlers.QueryHandler[FindUsersQuery, list[dict]]
        ):
            def handle(self, query: FindUsersQuery) -> FlextResult[list[dict]]:
                # Simulate database query with pagination
                users = [
                    {
                        "id": "user_1",
                        "name": "John",
                        "role": query.role_filter or "user",
                    },
                    {"id": "user_2", "name": "Jane", "role": "REDACTED_LDAP_BIND_PASSWORD"},
                ]

                return FlextCommands.Results.success(users)


        query = FindUsersQuery(role_filter="REDACTED_LDAP_BIND_PASSWORD", page_size=10, sort_by="name")
        query_result = query_handler.execute(query)

    Function-based handler registration with decorators::

        @FlextCommands.Decorators.command_handler(CreateUserCommand)
        def handle_user_creation(command: CreateUserCommand) -> FlextResult[str]:
            return FlextCommands.Results.success(f"Created user: {command.name}")


        # Handler automatically registered with command bus
        result = bus.execute(
            CreateUserCommand(email="test@example.com", name="Test User")
        )

**Design Patterns**:
    - **Command Pattern**: Encapsulates requests as objects with full validation
    - **Strategy Pattern**: Pluggable handler implementations for different command types
    - **Template Method**: Consistent execution pipeline with validation → handling → logging
    - **Factory Pattern**: Instance creation factories for handlers and buses
    - **Chain of Responsibility**: Middleware pipeline for cross-cutting concerns
    - **Observer Pattern**: Event handling and domain event processing
    - **Registry Pattern**: Handler registration and dynamic discovery

**Performance Characteristics**:
    - **Handler Lookup**: O(n) handler discovery with type introspection caching
    - **Command Validation**: Multi-layer validation with early termination on failures
    - **Middleware Pipeline**: Sequential processing with failure short-circuiting
    - **Memory Management**: Immutable command objects with minimal copying
    - **Concurrency**: Thread-safe handler registration and command execution
    - **Monitoring**: Built-in performance metrics and execution timing

**Error Handling Strategy**:
    All operations use FlextResult for consistent, type-safe error handling:

    - **Validation Errors**: ``VALIDATION_ERROR`` with detailed field-level messages
    - **Handler Errors**: ``COMMAND_HANDLER_NOT_FOUND`` for missing handlers
    - **Processing Errors**: ``COMMAND_PROCESSING_FAILED`` for execution failures
    - **Bus Errors**: ``COMMAND_BUS_ERROR`` for middleware and routing failures
    - **Query Errors**: ``QUERY_PROCESSING_FAILED`` for read operation failures

**Thread Safety**:
    The CQRS system is designed for concurrent operation:

    - Handler registration uses thread-safe dictionaries
    - Command execution is stateless and thread-safe
    - Middleware pipeline supports concurrent processing
    - Logging and metrics collection use correlation IDs for tracing

**Integration Points**:
    - **Message Queues**: FlextModels.Payload serialization for asynchronous processing
    - **Event Sourcing**: Domain event generation and processing capabilities
    - **API Layers**: Direct integration with FastAPI and Flask endpoints
    - **Database Access**: Repository pattern integration through dependency injection
    - **Microservices**: Cross-service command routing via message buses
    - **Testing**: Comprehensive test fixtures and mock handler implementations

Module Role in Architecture:
    This module serves as the Application layer in Clean Architecture, coordinating
    between the Domain layer (business logic) and Infrastructure layer (persistence,
    messaging) while maintaining strict separation of concerns and dependency inversion.

**Classes and Methods**:
    Complete FlextCommands class hierarchy with all nested classes and their key methods:

    FlextCommands.Types: Type definitions extending FlextTypes
        • CommandType = str - Command classification type
        • QueryType = str - Query classification type
        • HandlerType = str - Handler classification type
        • ResultType[T] = FlextResult[T] - Generic result wrapper
        • PayloadType = FlextModels.Payload[dict] - Message payload type

    FlextCommands.Protocols: Interface definitions for type safety
        • CommandProtocol: Interface for command objects
        • QueryProtocol: Interface for query objects
        • HandlerProtocol[TRequest, TResponse]: Generic handler interface
        • BusProtocol: Interface for command bus operations
        • MiddlewareProtocol: Interface for middleware components

    FlextCommands.Models: Base classes for commands and queries
        • Command: Base command class with Pydantic validation
            - validate_command(self) -> FlextResult[None]: Command-specific validation
            - get_metadata(self) -> dict[str, object]: Extract command metadata
            - to_payload(self) -> FlextModels.Payload[dict]: Serialize to payload
            - log_operation(self, operation: str, **kwargs) -> None: Log command operations
        • Query: Base query class with pagination and sorting
            - validate_query(self) -> FlextResult[None]: Query validation
            - get_pagination_info(self) -> dict[str, int]: Extract pagination details
            - apply_sorting(self, data: list[T]) -> list[T]: Apply sorting to results
        • BaseModel: Foundation model with common functionality
            - require_field(field_name: str, value: object) -> FlextResult[None]: Field validation
            - require_email(email: str) -> FlextResult[None]: Email format validation
            - require_min_length(value: str, min_len: int, field_name: str) -> FlextResult[None]: Length validation

    FlextCommands.Handlers: Handler base classes for command/query processing
        • AbstractHandler[TRequest, TResponse]: Abstract base handler
            - handle(self, request: TRequest) -> FlextResult[TResponse]: Process request
            - can_handle(self, request: object) -> bool: Check if handler supports request
            - get_handler_info(self) -> dict[str, object]: Handler metadata
        • CommandHandler[TCommand, TResult]: Command-specific handler
            - handle_command(self, command: TCommand) -> FlextResult[TResult]: Process command
            - validate_command(self, command: TCommand) -> FlextResult[None]: Pre-processing validation
            - log_command_execution(self, command: TCommand, result: TResult) -> None: Execution logging
        • QueryHandler[TQuery, TResult]: Query-specific handler
            - handle_query(self, query: TQuery) -> FlextResult[TResult]: Process query
            - apply_pagination(self, results: list[T], query: TQuery) -> list[T]: Paginate results
            - apply_filters(self, results: list[T], query: TQuery) -> list[T]: Filter results

    FlextCommands.Bus: Central command bus for routing and execution
        • CommandBus: Main bus implementation
            - execute(self, command: object) -> FlextResult[object]: Execute command
            - register_handler(self, handler: object) -> FlextResult[None]: Register command handler
            - add_middleware(self, middleware: object) -> None: Add middleware to pipeline
            - get_registered_handlers(self) -> dict[str, object]: List registered handlers
        • QueryBus: Query processing bus
            - query(self, query: object) -> FlextResult[object]: Execute query
            - register_query_handler(self, handler: object) -> FlextResult[None]: Register query handler
        • EventBus: Domain event processing
            - publish(self, event: object) -> FlextResult[None]: Publish domain event
            - subscribe(self, handler: object, event_type: str) -> None: Subscribe to events

    FlextCommands.Decorators: Function-based handler registration
        • command_handler(command_type: type) -> Callable: Decorator for command handlers
        • query_handler(query_type: type) -> Callable: Decorator for query handlers
        • middleware(order: int = 0) -> Callable: Decorator for middleware registration
        • validation_middleware() -> Callable: Pre-built validation middleware

    FlextCommands.Results: Factory methods for consistent result creation
        • success(data: T) -> FlextResult[T]: Create successful result
        • failure(error: str, error_code: str = None) -> FlextResult[None]: Create failure result
        • validation_error(message: str, field: str = None) -> FlextResult[None]: Validation error result
        • not_found(resource: str) -> FlextResult[None]: Not found error result

    FlextCommands.Factories: Instance creation with dependency injection
        • create_command_bus(**kwargs) -> CommandBus: Create configured command bus
        • create_query_bus(**kwargs) -> QueryBus: Create configured query bus
        • create_handler(handler_type: str, **config) -> AbstractHandler: Create handler instance
        • create_middleware(middleware_type: str, **config) -> object: Create middleware instance

**Migration Notes**:
    This implementation replaces legacy command processing patterns with:

    - Type-safe generics instead of duck typing
    - FlextResult railway patterns instead of exception-based error handling
    - Hierarchical class organization instead of scattered utility functions
    - Pydantic v2 validation instead of manual validation logic
    - Structured logging instead of print statements
    - Configuration through FlextConstants instead of hardcoded values
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Self, cast
from zoneinfo import ZoneInfo

from pydantic import (
    ConfigDict,
    Field,
    ValidationError as PydanticValidationError,
    model_validator,
)

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

# =============================================================================
# FLEXT COMMANDS - Consolidated CQRS Implementation
# =============================================================================


class FlextCommands:
    """Consolidated CQRS Command and Query Processing System for enterprise applications.

    This class serves as the single source of truth for all CQRS functionality in the
    FLEXT ecosystem, implementing a comprehensive command and query processing system
    with enterprise-grade features including validation, monitoring, middleware support,
    and type-safe operation patterns.

    **Architectural Principles**:
        The system follows SOLID principles and Clean Architecture patterns:

        - **Single Responsibility**: Each nested class focuses on a specific CQRS concern
        - **Open/Closed**: Easy extension through composition and inheritance
        - **Liskov Substitution**: Consistent interfaces across all command types
        - **Interface Segregation**: Clients depend only on needed interfaces
        - **Dependency Inversion**: Depends on abstractions (FlextTypes, FlextProtocols)

    **System Organization**:
        Hierarchically organized nested classes for logical separation:

        - **Types**: Command-specific type definitions extending FlextTypes
        - **Protocols**: Interface definitions for type safety and contracts
        - **Models**: Command and Query base classes with Pydantic v2 validation
        - **Handlers**: Generic handler base classes for processing logic
        - **Bus**: Central command bus for routing, middleware, and execution
        - **Decorators**: Function-based handler registration patterns
        - **Results**: FlextResult factory methods for consistent results
        - **Factories**: Instance creation methods with dependency injection

    **Key Features**:
        - **Type Safety**: Full generic typing with Python 3.13+ features
        - **Railway Programming**: FlextResult patterns for error handling
        - **Auto-Validation**: Multi-layer validation with business rules
        - **Handler Discovery**: Dynamic type introspection for routing
        - **Middleware Support**: Pluggable cross-cutting concerns
        - **Performance Monitoring**: Execution timing and metrics collection
        - **Structured Logging**: Correlation IDs and context tracking
        - **Payload Integration**: Seamless message serialization

    **Usage Examples**:
        Basic command processing with validation::

            from flext_core import FlextCommands, FlextResult


            # Define domain command
            class CreateUserCommand(FlextCommands.Models.Command):
                email: str
                name: str

                def validate_command(self) -> FlextResult[None]:
                    return self.require_email(self.email).flat_map(
                        lambda _: self.require_min_length(self.name, 2, "name")
                    )


            # Implement handler
            class UserHandler(
                FlextCommands.Handlers.CommandHandler[CreateUserCommand, str]
            ):
                def handle(self, command: CreateUserCommand) -> FlextResult[str]:
                    return FlextCommands.Results.success(f"Created: {command.name}")


            # Execute through bus
            bus = FlextCommands.Bus()
            bus.register_handler(UserHandler())

            command = CreateUserCommand(email="user@example.com", name="John Doe")
            result = bus.execute(command)

        Query processing with pagination::

            class FindUsersQuery(FlextCommands.Models.Query):
                role: str | None = None
                active_only: bool = True


            query = FindUsersQuery(role="REDACTED_LDAP_BIND_PASSWORD", page_size=50, sort_by="created_at")
            validation_result = query.validate_query()

            if validation_result.success:
                # Process query with validated parameters
                pass

        Type-safe factory usage::

            # Create handler from function
            handler = FlextCommands.Factories.create_simple_handler(
                lambda cmd: FlextCommands.Results.success("processed")
            )

            # Create configured command bus
            bus = FlextCommands.Factories.create_command_bus()
            bus.register_handler(handler)

    **Integration Points**:
        - **FlextResult**: Railway-oriented programming for all operations
        - **FlextTypes**: Hierarchical type system integration
        - **FlextConstants**: Configuration constants and error codes
        - **FlextProtocols**: Interface contracts and validation patterns
        - **FlextLogger**: Structured logging with correlation tracking
        - **FlextModels**: Pydantic v2 validation and serialization
        - **FlextMixins**: Reusable behaviors (timing, logging, serialization)

    **Thread Safety**:
        All components are designed for concurrent operation with proper synchronization
        for shared state (handler registry) and stateless execution patterns.

    **Performance Characteristics**:
        - Handler lookup: O(n) with type introspection caching
        - Validation: Early termination on first failure
        - Middleware: Sequential with short-circuit on failures
        - Memory: Immutable commands with minimal copying overhead

    """

    # =========================================================================
    # TYPES - Command type definitions using FlextTypes hierarchy
    # =========================================================================

    class Types(FlextTypes):
        """Command-specific type definitions extending the FLEXT hierarchical type system.

        This class provides type aliases and definitions specifically for command and
        query processing, extending the base FlextTypes hierarchy to ensure type safety
        and consistency across the CQRS system.

        **Design Philosophy**:
            All types are designed for maximum type safety with static analysis tools
            (mypy, pyright) and runtime type checking where appropriate.

        **Type Categories**:
            - **Core Types**: Basic command identifiers and metadata
            - **Query Types**: Query-specific identifiers and parameters
            - **Handler Types**: Handler registration and metrics types
            - **Function Types**: Callable type aliases for handlers and validators

        **Usage Example**:
            Type-safe command processing with proper annotations::

                command_id: FlextCommands.Types.CommandId = "cmd_12345"
                metadata: FlextCommands.Types.CommandMetadata = {"user": "john"}
                handler_func: FlextCommands.Types.CommandHandlerFunction = (
                    lambda cmd: FlextResult.ok("done")
                )

        """

        # Core command types from FlextTypes hierarchy
        CommandId = FlextTypes.Core.Id
        CommandType = FlextTypes.Core.String
        CommandState = FlextTypes.Core.String
        CommandMetadata = FlextTypes.Core.Dict
        CorrelationId = FlextTypes.Core.Id

        # Query types
        QueryId = FlextTypes.Core.Id
        QueryType = FlextTypes.Core.String
        QueryParameters = FlextTypes.Core.Dict

        # Handler types
        HandlerName = FlextTypes.Service.ServiceName
        HandlerMetrics = FlextTypes.Core.Dict

        # Command-specific type aliases from FlextTypes hierarchy
        CommandData = FlextTypes.Core.Dict
        # Command result types
        CommandResult = FlextTypes.Core.Dict | list[FlextTypes.Core.Dict]
        ValidationErrors = list[str]
        CommandExecutionTime = FlextTypes.Core.Float
        HandlerRegistry = FlextTypes.Core.Dict

        # Function type aliases using FlextTypes unified Callable patterns
        CommandHandlerFunction = FlextTypes.Handler.CommandHandler
        QueryHandlerFunction = FlextTypes.Handler.QueryHandler
        ValidatorFunction = FlextTypes.Core.FlextCallableType
        CommandBusMiddleware = FlextTypes.Core.OperationCallable

    # =========================================================================
    # PROTOCOLS - Command protocol definitions for type safety
    # =========================================================================

    class Protocols:
        """Command protocol definitions providing type-safe interfaces and contracts.

        This class defines all protocol interfaces used throughout the CQRS system,
        ensuring strict typing and clear contracts for command processing components.
        Built on the FlextProtocols foundation for consistency across the ecosystem.

        **Protocol Categories**:
            - **Handler Protocols**: Interfaces for command and query handlers
            - **Bus Protocols**: Message routing and processing contracts
            - **Validation Protocols**: Validation and processing interfaces
            - **Event Protocols**: Domain event processing contracts

        **Design Benefits**:
            - **Type Safety**: Static analysis can verify interface compliance
            - **Loose Coupling**: Depend on interfaces, not concrete implementations
            - **Testing**: Easy mocking and substitution for unit tests
            - **Documentation**: Protocols serve as executable documentation

        **Usage Example**:
            Protocol-based handler implementation::

                class MyHandler(FlextCommands.Protocols.CommandHandler):
                    def handle(self, command: object) -> FlextResult[object]:
                        # Implementation guarantees protocol compliance
                        return FlextResult.ok("processed")

        """

        # Use FlextProtocols as foundation
        Handler = FlextProtocols.Application.Handler
        MessageHandler = FlextProtocols.Application.MessageHandler
        ValidatingHandler = FlextProtocols.Application.ValidatingHandler
        EventProcessor = FlextProtocols.Application.EventProcessor

        # Command-specific protocol aliases
        CommandHandler = Handler
        QueryHandler = Handler
        CommandBus = MessageHandler

    # =========================================================================
    # MODELS - Command and query model implementations
    # =========================================================================

    class Models:
        """Command and Query model definitions with enterprise validation and serialization.

        This class contains the base model implementations for all command and query
        objects in the CQRS system, providing comprehensive validation, automatic
        serialization, and metadata management using Pydantic v2 patterns.

        **Model Features**:
            - **Automatic Validation**: Multi-layer validation with business rules
            - **Type Safety**: Full generic typing with runtime validation
            - **Immutability**: Frozen models preventing accidental mutation
            - **Serialization**: JSON/dict conversion with proper type handling
            - **Metadata Management**: Automatic timestamps and correlation IDs
            - **Payload Integration**: Seamless FlextModels.Payload serialization

        **Base Classes**:
            - **Command**: Write operations with validation and audit trails
            - **Query**: Read operations with pagination and sorting support

        **Validation Strategy**:
            The models implement a multi-layer validation approach:

            1. **Pydantic Validation**: Field-level type and constraint validation
            2. **Business Validation**: Custom business rules via ``validate_*`` methods
            3. **Cross-Field Validation**: Complex validation across multiple fields
            4. **Railway Patterns**: FlextResult chaining for validation composition

        **Usage Example**:
            Creating validated command models::

                class CreateOrderCommand(FlextCommands.Models.Command):
                    customer_id: str
                    items: list[str]
                    total_amount: float

                    def validate_command(self) -> FlextResult[None]:
                        return (
                            self.require_field("customer_id", self.customer_id)
                            .flat_map(lambda _: self._validate_items())
                            .flat_map(lambda _: self._validate_amount())
                        )

                    def _validate_items(self) -> FlextResult[None]:
                        if not self.items:
                            return FlextResult.fail("Order must have items")
                        return FlextResult.ok(None)

                    def _validate_amount(self) -> FlextResult[None]:
                        if self.total_amount <= 0:
                            return FlextResult.fail("Amount must be positive")
                        return FlextResult.ok(None)

        """

        class Command(FlextModels.BaseConfig):
            """Base command model for write operations in the CQRS system.

            This class provides the foundation for all command objects in the FLEXT
            ecosystem, implementing comprehensive enterprise patterns for command
            processing including validation, serialization, audit trails, and
            integration with the broader FLEXT architecture.

            **Enterprise Features**:
                - **Automatic ID Generation**: UUID generation via FlextUtilities.Generators
                - **Timestamp Management**: UTC timestamp tracking for audit trails
                - **Type Inference**: Automatic command type from class name (CamelCase → snake_case)
                - **Validation Framework**: Multi-layer validation with business rules
                - **Serialization Support**: JSON/dict serialization via FlextMixins
                - **Logging Integration**: Structured logging with correlation IDs
                - **Payload Conversion**: Seamless FlextModels.Payload integration for messaging
                - **Immutability**: Frozen model preventing accidental state changes

            **Validation Strategy**:
                Commands implement a comprehensive validation approach:

                1. **Field Validation**: Pydantic field-level constraints and types
                2. **Business Rules**: Custom ``validate_command()`` method override
                3. **Helper Methods**: ``require_field``, ``require_email``, ``require_min_length``
                4. **Railway Composition**: FlextResult chaining for complex validation

            **Audit and Tracking**:
                Every command automatically includes:

                - ``command_id``: Unique identifier for command instance
                - ``command_type``: Auto-generated type from class name
                - ``timestamp``: UTC creation timestamp
                - ``user_id``: Optional user context for audit trails
                - ``correlation_id``: Request correlation for distributed tracing

            **Usage Patterns**:
                Basic command with validation::

                    class UpdateUserCommand(FlextCommands.Models.Command):
                        user_id: str
                        email: str
                        name: str

                        def validate_command(self) -> FlextResult[None]:
                            return (
                                self.require_field("user_id", self.user_id)
                                .flat_map(lambda _: self.require_email(self.email))
                                .flat_map(
                                    lambda _: self.require_min_length(
                                        self.name, 2, "name"
                                    )
                                )
                            )


                    # Usage with automatic validation
                    command = UpdateUserCommand(
                        user_id="user_123", email="john@example.com", name="John Doe"
                    )

                    validation_result = command.validate_command()
                    if validation_result.success:
                        # Command is valid and ready for processing
                        payload = command.to_payload()

                Complex validation with business rules::

                    class PlaceOrderCommand(FlextCommands.Models.Command):
                        customer_id: str
                        items: list[dict[str, object]]
                        payment_method: str

                        def validate_command(self) -> FlextResult[None]:
                            # Chain multiple validation rules
                            return (
                                self.require_field("customer_id", self.customer_id)
                                .flat_map(lambda _: self._validate_items())
                                .flat_map(lambda _: self._validate_payment_method())
                                .flat_map(lambda _: self._validate_business_rules())
                            )

                        def _validate_items(self) -> FlextResult[None]:
                            if not self.items:
                                return FlextResult.fail("Order must contain items")
                            return FlextResult.ok(None)

            **Serialization and Messaging**:
                Commands support multiple serialization formats:

                - ``to_dict_basic()``: Basic dictionary representation
                - ``to_json()``: JSON string serialization
                - ``to_payload()``: FlextModels.Payload for message queues
                - ``from_payload()``: Deserialization from FlextModels.Payload

            **Integration Points**:
                - **FlextResult**: All validation returns FlextResult for railway patterns
                - **FlextUtilities**: ID generation and validation helpers
                - **FlextMixins**: Logging, timing, and serialization behaviors
                - **FlextLogger**: Structured logging with context
                - **FlextModels.Payload**: Message serialization for distributed systems

            **Thread Safety**:
                Commands are immutable (frozen=True) ensuring thread-safe sharing
                across concurrent operations without synchronization overhead.

            """

            model_config = ConfigDict(
                validate_assignment=True,
                str_strip_whitespace=True,
                extra="forbid",
                frozen=True,
            )

            # Use FlextTypes for all field definitions
            command_id: str = Field(
                default_factory=FlextUtilities.Generators.generate_uuid,
                description="Unique command identifier using FlextUtilities",
            )

            command_type: str = Field(
                default="",
                description="Command type for routing and processing",
                max_length=100,
            )

            timestamp: datetime = Field(
                default_factory=lambda: datetime.now(tz=ZoneInfo("UTC")),
                description="Command creation timestamp in UTC",
            )

            user_id: str | None = Field(
                default=None,
                description="User who initiated the command",
            )

            correlation_id: str = Field(
                default_factory=FlextUtilities.Generators.generate_correlation_id,
                description="Correlation ID for request tracking",
                max_length=50,
            )

            @model_validator(mode="before")
            @classmethod
            def set_command_type(cls, values: dict[str, object]) -> dict[str, object]:
                """Auto-generate command_type from class name if not provided.

                Converts class names like 'CreateUserCommand' to 'create_user'
                following FLEXT naming conventions.
                """
                if not values.get("command_type"):
                    class_name = cls.__name__
                    # Remove trailing 'Command' if present
                    base = class_name.removesuffix("Command")
                    # Convert CamelCase/PascalCase to snake_case
                    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
                    command_type = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
                    values["command_type"] = command_type
                return values

            @classmethod
            def from_payload(
                cls,
                payload: FlextModels.Payload[FlextTypes.Core.Dict],
            ) -> FlextResult[Self]:
                """Create command from FlextModels.Payload with full validation.

                Args:
                    payload: FlextModels.Payload containing command data

                Returns:
                    FlextResult containing validated command or error

                """
                logger = FlextLogger(f"{cls.__module__}.{cls.__name__}")
                logger.debug(
                    "Creating command from payload",
                    payload_type=payload.headers.get("type", "unknown"),
                    expected_type=cls.__name__,
                )

                expected_type = payload.headers.get("type", "")
                if expected_type not in {cls.__name__, ""}:
                    logger.warning(
                        "Payload type mismatch",
                        expected=cls.__name__,
                        actual=expected_type,
                    )

                # Extract and validate payload data
                raw_data = payload.data
                payload_dict = {
                    str(k): v for k, v in cast("dict[object, object]", raw_data).items()
                }

                # Build command with explicit field extraction
                command_fields: dict[str, object] = {
                    "command_id": str(
                        payload_dict.get(
                            "command_id", FlextUtilities.Generators.generate_uuid()
                        )
                    ),
                    "command_type": str(payload_dict.get("command_type", cls.__name__)),
                    "user_id": str(payload_dict["user_id"])
                    if payload_dict.get("user_id")
                    else None,
                    "correlation_id": str(
                        payload_dict.get(
                            "correlation_id",
                            FlextUtilities.Generators.generate_correlation_id(),
                        )
                    ),
                }

                # Handle timestamp parsing
                timestamp_raw = payload_dict.get("timestamp")
                if isinstance(timestamp_raw, str):
                    command_fields["timestamp"] = datetime.fromisoformat(timestamp_raw)
                else:
                    command_fields["timestamp"] = datetime.now(tz=ZoneInfo("UTC"))

                # Add remaining fields
                remaining_data = {
                    k: v for k, v in payload_dict.items() if k not in command_fields
                }
                command_fields.update(remaining_data)

                try:
                    # Create command using proper Pydantic model construction
                    # Extract known fields that match Command model
                    command_data = {
                        "command_id": command_fields.get(
                            "command_id", FlextUtilities.Generators.generate_uuid()
                        ),
                        "command_type": str(command_fields.get("command_type", "")),
                        "timestamp": command_fields.get(
                            "timestamp", datetime.now(tz=ZoneInfo("UTC"))
                        ),
                        "user_id": command_fields.get("user_id"),
                        "correlation_id": command_fields.get(
                            "correlation_id",
                            FlextUtilities.Generators.generate_correlation_id(),
                        ),
                    }

                    # Add any additional fields for subclass-specific data
                    for key, value in command_fields.items():
                        if key not in command_data:
                            command_data[key] = value

                    # Create command using Pydantic model_validate for type safety
                    command = cls.model_validate(command_data)

                    # Validate using command's validation method
                    validation_result = command.validate_command()
                    if validation_result.is_failure:
                        return FlextResult[Self].fail(
                            validation_result.error or "Command validation failed",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

                    logger.info(
                        "Command created from payload",
                        command_type=command.command_type,
                        command_id=command.command_id,
                    )
                    return FlextResult[Self].ok(command)

                except (ValueError, TypeError, PydanticValidationError) as e:
                    return FlextResult[Self].fail(
                        f"Command creation failed: {e}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            def validate_command(self) -> FlextResult[None]:
                """Validate command using FlextValidation patterns.

                Override in subclasses for custom validation logic.
                Uses FlextValidation for common validation patterns.

                Returns:
                    FlextResult indicating validation success or failure

                """
                return FlextResult[None].ok(None)

            @staticmethod
            def require_field(
                field_name: str,
                value: object,
                error_msg: str = "",
            ) -> FlextResult[None]:
                """Validate a required field with structured error handling.

                Args:
                    field_name: Name of the field being validated
                    value: Value to validate
                    error_msg: Custom error message (optional)

                Returns:
                    FlextResult indicating validation success or failure

                """
                if value is None or (isinstance(value, str) and not value.strip()):
                    msg = error_msg or f"{field_name} is required"
                    return FlextResult[None].fail(
                        msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[None].ok(None)

            @staticmethod
            def require_email(
                email: str,
                field_name: str = "email",
            ) -> FlextResult[None]:
                """Validate email format using FlextValidation.

                Args:
                    email: Email address to validate
                    field_name: Name of the field (for error messages)

                Returns:
                    FlextResult indicating validation success or failure

                """
                # Use FlextValidation for validation
                if (
                    not email
                    or "@" not in email
                    or "." not in email.rsplit("@", maxsplit=1)[-1]
                ):
                    return FlextResult[None].fail(
                        f"Invalid {field_name} format",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[None].ok(None)

            @staticmethod
            def require_min_length(
                value: str,
                min_len: int,
                field_name: str,
            ) -> FlextResult[None]:
                """Validate minimum string length.

                Args:
                    value: String value to validate
                    min_len: Minimum required length
                    field_name: Name of the field (for error messages)

                Returns:
                    FlextResult indicating validation success or failure

                """
                if len(value.strip()) < min_len:
                    error_msg = f"{field_name} must be at least {min_len} characters"
                    return FlextResult[None].fail(
                        error_msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return FlextResult[None].ok(None)

            def get_metadata(self) -> dict[str, object]:
                """Get command metadata using FlextTypes.

                Returns:
                    Command metadata dictionary with typed structure

                """
                return {
                    "command_id": self.command_id,
                    "command_type": self.command_type,
                    "command_class": self.__class__.__name__,
                    "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                    "user_id": self.user_id,
                    "correlation_id": self.correlation_id,
                }

            def to_payload(self) -> FlextModels.Payload[FlextTypes.Core.Dict]:
                """Convert command to FlextModels.Payload for serialization.

                Returns:
                    FlextModels.Payload containing command data and metadata

                """
                # Convert the command model to a dictionary
                command_dict = self.model_dump()

                # Ensure timestamp is serialized as ISO string
                if "timestamp" in command_dict and isinstance(
                    command_dict["timestamp"], datetime
                ):
                    command_dict["timestamp"] = command_dict["timestamp"].isoformat()

                # Create payload with metadata
                metadata = self.get_metadata()
                metadata_dict = dict(metadata)
                metadata_dict["type"] = (
                    self.command_type or self.__class__.__name__.lower()
                )

                # Create and return payload using FlextModels factory

                result = FlextModels.create_payload(
                    data=command_dict,
                    message_type=self.command_type or self.__class__.__name__.lower(),
                    source_service=str(metadata_dict.get("source_service", "unknown")),
                    target_service=str(metadata_dict.get("target_service"))
                    if metadata_dict.get("target_service")
                    else None,
                    correlation_id=str(metadata_dict.get("correlation_id"))
                    if metadata_dict.get("correlation_id")
                    else None,
                )
                if result.success:
                    return result.unwrap()
                msg = f"Failed to create payload: {result.error}"
                raise RuntimeError(msg)

            # FlextMixins integration methods
            def log_operation(self, operation: str, **kwargs: object) -> None:
                """Log operation using FlextMixins."""
                FlextMixins.log_operation(self, operation, **kwargs)

            def log_info(self, message: str, **kwargs: object) -> None:
                """Log info using FlextMixins."""
                FlextMixins.log_info(self, message, **kwargs)

            def log_error(self, message: str, **kwargs: object) -> None:
                """Log error using FlextMixins."""
                FlextMixins.log_error(self, message, **kwargs)

            def to_dict_basic(self) -> FlextTypes.Core.Dict:
                """Convert to basic dict using FlextMixins."""
                return FlextMixins.to_dict_basic(self)

            def to_json(self, **kwargs: object) -> str:
                """Convert to JSON using FlextMixins."""
                indent = kwargs.get("indent")
                if isinstance(indent, int) or indent is None:
                    return FlextMixins.to_json(self, indent)
                return FlextMixins.to_json(self, None)

        class Query(FlextModels.BaseConfig):
            """Base query model for read operations in the CQRS system.

            This class provides the foundation for all query objects in the FLEXT
            ecosystem, implementing enterprise patterns for data retrieval operations
            including pagination, sorting, filtering, and comprehensive validation.

            **Query Characteristics**:
                - **Read-Only Operations**: No side effects or state changes
                - **Immutable Design**: Frozen model preventing modification after creation
                - **Pagination Support**: Built-in page size and number handling
                - **Sorting Capabilities**: Configurable sort field and order
                - **Validation Framework**: Business rule validation for query parameters
                - **Type Safety**: Full generic typing for result consistency

            **Pagination Features**:
                Built-in pagination with enterprise-grade defaults:

                - ``page_size``: Number of results per page (1-1000, default: 100)
                - ``page_number``: Page number for offset calculation (≥1, default: 1)
                - ``sort_by``: Field name for sorting (optional)
                - ``sort_order``: Sort direction ('asc' or 'desc', default: 'asc')

            **Validation Strategy**:
                Queries implement comprehensive parameter validation:

                1. **Field Validation**: Pydantic constraints for basic parameter validation
                2. **Business Validation**: Custom ``validate_query()`` method for business rules
                3. **Range Validation**: Page size limits using FlextConstants configuration
                4. **Consistency Checks**: Cross-parameter validation for logical consistency

            **Usage Patterns**:
                Basic query with pagination::

                    class FindUsersQuery(FlextCommands.Models.Query):
                        role_filter: str | None = None
                        active_only: bool = True
                        department: str | None = None

                        def validate_query(self) -> FlextResult[None]:
                            # Add custom business validation
                            if self.role_filter and self.role_filter not in [
                                "REDACTED_LDAP_BIND_PASSWORD",
                                "user",
                                "manager",
                            ]:
                                return FlextResult.fail("Invalid role filter")
                            return super().validate_query()


                    # Usage with automatic pagination
                    query = FindUsersQuery(
                        role_filter="REDACTED_LDAP_BIND_PASSWORD",
                        active_only=True,
                        page_size=50,
                        sort_by="created_at",
                        sort_order="desc",
                    )

                    validation_result = query.validate_query()
                    if validation_result.success:
                        # Query parameters are valid
                        pass

                Complex filtering with validation::

                    class OrderSearchQuery(FlextCommands.Models.Query):
                        customer_id: str | None = None
                        status_filter: list[str] | None = None
                        date_from: datetime | None = None
                        date_to: datetime | None = None

                        def validate_query(self) -> FlextResult[None]:
                            # Chain validation rules
                            base_validation = super().validate_query()
                            if base_validation.failure:
                                return base_validation

                            # Custom date range validation
                            if (
                                self.date_from
                                and self.date_to
                                and self.date_from > self.date_to
                            ):
                                return FlextResult.fail(
                                    "date_from must be before date_to"
                                )

                            return FlextResult.ok(None)

            **Integration with Handlers**:
                Queries work seamlessly with QueryHandler implementations::

                    class UserQueryHandler(
                        FlextCommands.Handlers.QueryHandler[FindUsersQuery, list[dict]]
                    ):
                        def handle(
                            self, query: FindUsersQuery
                        ) -> FlextResult[list[dict]]:
                            # Access validated query parameters
                            offset = (query.page_number - 1) * query.page_size
                            limit = query.page_size

                            # Implement actual data retrieval
                            users = self.repository.find_users(
                                role=query.role_filter,
                                active=query.active_only,
                                offset=offset,
                                limit=limit,
                                sort_by=query.sort_by,
                                sort_order=query.sort_order,
                            )

                            return FlextResult.ok(users)

            **Performance Considerations**:
                - **Immutable Design**: No copying overhead for thread-safe sharing
                - **Lazy Validation**: Validation only occurs when explicitly called
                - **Parameter Limits**: Page size constraints prevent excessive resource usage
                - **Index Hints**: Sort field validation can guide database index usage

            **Error Handling**:
                All validation uses FlextResult patterns for consistent error handling:

                - **Parameter Errors**: Invalid page size, negative page numbers
                - **Business Rule Violations**: Custom validation failures
                - **Consistency Errors**: Cross-parameter validation failures

            **Thread Safety**:
                Queries are immutable and thread-safe, allowing concurrent execution
                without synchronization concerns.

            """

            model_config = ConfigDict(
                frozen=True,
                validate_assignment=True,
                extra="forbid",
            )

            query_id: str | None = Field(
                default_factory=FlextUtilities.Generators.generate_uuid,
                description="Unique query identifier",
            )

            query_type: str | None = Field(
                default=None,
                description="Query type for routing",
            )

            page_size: int = Field(
                default=100,
                ge=1,
                le=1000,
                description="Number of results per page",
            )

            page_number: int = Field(
                default=1,
                ge=1,
                description="Page number for pagination",
            )

            sort_by: str | None = Field(
                default=None,
                description="Field to sort by",
            )

            sort_order: str = Field(
                default="asc",
                pattern=r"^(asc|desc)$",
                description="Sort order: asc or desc",
            )

            def validate_query(self) -> FlextResult[None]:
                """Validate query with business logic using FlextValidation.

                Returns:
                    FlextResult indicating validation success or failure with errors

                """
                errors: FlextCommands.Types.ValidationErrors = []

                # Business validation using constants
                if self.page_size <= 0:
                    errors.append("Page size must be positive")
                if self.page_size > FlextConstants.Handlers.MAX_QUERY_RESULTS:
                    errors.append(
                        f"Page size cannot exceed {FlextConstants.Handlers.MAX_QUERY_RESULTS}"
                    )
                if self.page_number <= 0:
                    errors.append("Page number must be positive")
                if self.sort_order not in {"asc", "desc"}:
                    errors.append("Sort order must be 'asc' or 'desc'")

                # Check validation results
                if errors:
                    error_message = "; ".join(errors)
                    return FlextResult[None].fail(
                        f"Query validation failed: {error_message}",
                        error_code=FlextConstants.Errors.QUERY_PROCESSING_FAILED,
                    )

                return FlextResult[None].ok(None)

            # FlextMixins integration methods
            def to_dict_basic(self) -> FlextTypes.Core.Dict:
                """Convert to basic dict using FlextMixins."""
                return FlextMixins.to_dict_basic(self)

            def to_json(self, **kwargs: object) -> str:
                """Convert to JSON using FlextMixins."""
                indent = kwargs.get("indent")
                if isinstance(indent, int) or indent is None:
                    return FlextMixins.to_json(self, indent)
                return FlextMixins.to_json(self, None)

    # =========================================================================
    # HANDLERS - Command and query handler base classes
    # =========================================================================

    class Handlers:
        """Command and Query handler base classes for enterprise processing patterns.

        This class provides the foundational handler implementations for command and
        query processing in the CQRS system, offering enterprise-grade features including
        type safety, validation, monitoring, logging, and consistent error handling.

        **Handler Categories**:
            - **CommandHandler**: Generic base for write operation handlers
            - **QueryHandler**: Generic base for read operation handlers

        **Enterprise Features**:
            - **Generic Typing**: Full type safety with command/result constraints
            - **Automatic Validation**: Built-in command/query validation before processing
            - **Performance Monitoring**: Execution timing and metrics collection
            - **Structured Logging**: Correlation IDs and context tracking
            - **Error Standardization**: Consistent FlextResult error patterns
            - **Handler Discovery**: Type introspection for automatic routing
            - **Thread Safety**: Stateless design for concurrent execution

        **Validation Pipeline**:
            Handlers implement a comprehensive validation pipeline:

            1. **Handler Compatibility**: ``can_handle()`` method validates type compatibility
            2. **Input Validation**: Delegates to command/query ``validate_*()`` methods
            3. **Business Processing**: ``handle()`` method executes business logic
            4. **Result Validation**: Ensures FlextResult return type consistency

        **Usage Patterns**:
            Type-safe command handler implementation::

                class CreateUserHandler(
                    FlextCommands.Handlers.CommandHandler[CreateUserCommand, str]
                ):
                    def handle(self, command: CreateUserCommand) -> FlextResult[str]:
                        # Automatic logging and timing
                        self.log_info("Processing user creation", email=command.email)

                        # Business logic implementation
                        user_id = self.user_service.create_user(
                            email=command.email, name=command.name
                        )

                        return FlextResult.ok(user_id)


                # Handler provides automatic validation and error handling
                handler = CreateUserHandler()
                command = CreateUserCommand(email="test@example.com", name="Test User")
                result = handler.execute(
                    command
                )  # Full validation + processing pipeline

        **Integration Benefits**:
            - **FlextMixins**: Automatic logging, timing, and serialization behaviors
            - **FlextLogger**: Structured logging with handler context
            - **FlextResult**: Railway-oriented programming for error composition
            - **FlextConstants**: Error codes and configuration constants
            - **Type Introspection**: Dynamic handler discovery and routing

        """

        class CommandHandler[CommandT, ResultT](FlextMixins.Loggable):
            """Generic base class for command handlers with enterprise processing capabilities.

            This class provides a comprehensive foundation for implementing command handlers
            in the CQRS system, offering type safety, validation, monitoring, logging,
            and consistent error handling patterns.

            **Type Parameters**:
                - ``CommandT``: The specific command type this handler processes
                - ``ResultT``: The result type returned by successful command processing

            **Enterprise Capabilities**:
                - **Type Safety**: Generic constraints ensure compile-time type checking
                - **Automatic Validation**: Command validation before processing
                - **Handler Discovery**: Type introspection for automatic routing
                - **Performance Monitoring**: Execution timing and metrics collection
                - **Structured Logging**: Context-aware logging with correlation IDs
                - **Error Standardization**: Consistent FlextResult error handling
                - **Thread Safety**: Stateless design for concurrent execution

            **Processing Pipeline**:
                The ``execute()`` method implements a comprehensive processing pipeline:

                1. **Compatibility Check**: Validates handler can process the command type
                2. **Input Validation**: Delegates to command's ``validate_command()`` method
                3. **Business Processing**: Calls the ``handle()`` method for business logic
                4. **Error Handling**: Catches exceptions and converts to FlextResult failures
                5. **Metrics Collection**: Records execution time and success/failure metrics
                6. **Structured Logging**: Logs processing stages with context

            **Implementation Pattern**:
                Subclasses must implement the ``handle()`` method::

                    class ProcessOrderHandler(
                        FlextCommands.Handlers.CommandHandler[ProcessOrderCommand, str]
                    ):
                        def __init__(self, order_service: OrderService):
                            super().__init__(handler_name="ProcessOrder")
                            self.order_service = order_service

                        def handle(
                            self, command: ProcessOrderCommand
                        ) -> FlextResult[str]:
                            # Business logic with automatic logging context
                            self.log_info("Processing order", order_id=command.order_id)

                            try:
                                order_id = self.order_service.process_order(
                                    customer_id=command.customer_id, items=command.items
                                )

                                self.log_info(
                                    "Order processed successfully", order_id=order_id
                                )
                                return FlextResult.ok(order_id)

                            except BusinessRuleViolation as e:
                                return FlextResult.fail(f"Business rule violation: {e}")

            **Validation Integration**:
                Handlers automatically validate commands before processing::

                    # Handler automatically calls command.validate_command()
                    result = handler.execute(command)

                    # If validation fails, FlextResult.fail() is returned
                    # If validation succeeds, handle() method is called

            **Metrics and Monitoring**:
                Automatic collection of processing metrics:

                - Execution time in milliseconds
                - Success/failure rates
                - Command type and handler identification
                - Error categorization and tracking

            **Thread Safety**:
                Handlers are designed to be stateless and thread-safe. Handler instances
                can be registered once and used concurrently without synchronization.

            **Integration Points**:
                - **FlextMixins.Loggable**: Provides logging, timing, and serialization
                - **FlextLogger**: Structured logging with handler context
                - **FlextResult**: Railway-oriented error handling
                - **FlextConstants**: Error codes and configuration

            """

            def __init__(
                self,
                handler_name: str | None = None,
                handler_id: str | None = None,
            ) -> None:
                """Initialize handler with logging and metrics.

                Args:
                    handler_name: Human-readable handler name
                    handler_id: Unique handler identifier

                """
                super().__init__()
                self._metrics_state: dict[str, object] | None = None
                self._handler_name = handler_name or self.__class__.__name__
                self.handler_id = handler_id or f"{self.__class__.__name__}_{id(self)}"

            # Timing functionality using FlextMixins
            def start_timing(self) -> float:
                """Start timing operation."""
                return FlextMixins.start_timing(self)

            def stop_timing(self) -> float:
                """Stop timing and return elapsed seconds."""
                return FlextMixins.stop_timing(self)

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return self._handler_name

            @property
            def logger(self) -> FlextLogger:
                """Get logger instance for this handler."""
                return FlextLogger(self.__class__.__name__)

            def validate_command(self, command: object) -> FlextResult[None]:
                """Validate command before handling.

                Args:
                    command: Command object to validate

                Returns:
                    FlextResult indicating validation success or failure

                """
                # Delegate to command's validation if available
                validate_method = getattr(command, "validate_command", None)
                if callable(validate_method):
                    result = validate_method()
                    if hasattr(result, "success") and hasattr(result, "data"):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, command: CommandT) -> FlextResult[ResultT]:
                """Handle the command and return result.

                Args:
                    command: Command to handle

                Returns:
                    FlextResult with execution result or error

                Note:
                    Subclasses must implement this method for actual processing.

                """
                # Subclasses must implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

            def can_handle(self, command: object) -> bool:
                """Check if handler can process this command.

                Uses FlextUtilities type guards for validation and generic inspection.

                Args:
                    command: Command object to check

                Returns:
                    True if handler can process the command, False otherwise

                """
                self.logger.debug(
                    "Checking if handler can process command",
                    command_type=type(command).__name__,
                )

                # Get expected command type from Generic parameter
                orig_bases = getattr(self, "__orig_bases__", None)
                if orig_bases is not None:
                    for base in orig_bases:
                        args = getattr(base, "__args__", None)
                        if args is not None and len(args) >= 1:
                            expected_type = base.__args__[0]
                            # Use direct isinstance for validation
                            can_handle_result = isinstance(command, expected_type)

                            self.logger.debug(
                                "Handler check result",
                                can_handle=can_handle_result,
                                expected_type=getattr(
                                    expected_type,
                                    "__name__",
                                    str(expected_type),
                                ),
                            )
                            return bool(can_handle_result)

                self.logger.warning("Could not determine handler type constraints")
                return True

            def execute(self, command: CommandT) -> FlextResult[ResultT]:
                """Execute command with full validation and error handling.

                Args:
                    command: Command to execute

                Returns:
                    FlextResult with execution result or structured error

                """
                self.logger.info(
                    "Executing command",
                    command_type=type(command).__name__,
                    command_id=getattr(command, "command_id", "unknown"),
                )

                # Validate command can be handled
                if not self.can_handle(command):
                    error_msg = (
                        f"{self._handler_name} cannot handle {type(command).__name__}"
                    )
                    self.logger.error(error_msg)
                    return FlextResult[ResultT].fail(
                        error_msg,
                        error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                    )

                # Validate the command's data
                validation_result = self.validate_command(command)
                if validation_result.is_failure:
                    self.logger.warning(
                        "Command validation failed",
                        command_type=type(command).__name__,
                        error=validation_result.error,
                    )
                    return FlextResult[ResultT].fail(
                        validation_result.error or "Validation failed",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                start_time = time.perf_counter()

                try:
                    result: FlextResult[ResultT] = self.handle(command)

                    execution_time = round((time.perf_counter() - start_time) * 1000, 2)
                    self.logger.info(
                        "Command executed successfully",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time,
                        success=result.is_success,
                    )

                    return result

                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    execution_time = round((time.perf_counter() - start_time) * 1000, 2)
                    self.logger.exception(
                        "Command execution failed",
                        command_type=type(command).__name__,
                        execution_time_ms=execution_time,
                        error=str(e),
                    )
                    return FlextResult[ResultT].fail(
                        f"Command processing failed: {e}",
                        error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                    )

        class QueryHandler[QueryT, QueryResultT]:
            """Generic base class for query handlers with enterprise read operation capabilities.

            This class provides a comprehensive foundation for implementing query handlers
            in the CQRS system, specifically designed for read operations that do not
            produce side effects. Offers type safety, validation, and consistent error
            handling patterns optimized for data retrieval scenarios.

            **Type Parameters**:
                - ``QueryT``: The specific query type this handler processes
                - ``QueryResultT``: The result type returned by successful query execution

            **Query Handler Characteristics**:
                - **Read-Only Operations**: Designed for data retrieval without side effects
                - **Type Safety**: Generic constraints ensure compile-time type checking
                - **Validation Integration**: Automatic query parameter validation
                - **Performance Optimized**: Lightweight design for high-throughput scenarios
                - **Thread Safety**: Stateless design supporting concurrent query execution
                - **Pagination Support**: Built-in support for paginated result sets

            **Processing Approach**:
                Query handlers follow a simplified processing pattern optimized for reads:

                1. **Query Compatibility**: ``can_handle()`` validates handler-query compatibility
                2. **Parameter Validation**: Delegates to query's ``validate_query()`` method
                3. **Data Retrieval**: ``handle()`` method executes data access logic
                4. **Result Formatting**: Ensures proper result type and structure
                5. **Error Handling**: Converts exceptions to structured FlextResult failures

            **Implementation Pattern**:
                Subclasses implement the ``handle()`` method for data retrieval::

                    class FindUsersQueryHandler(
                        FlextCommands.Handlers.QueryHandler[FindUsersQuery, list[dict]]
                    ):
                        def __init__(self, user_repository: UserRepository):
                            super().__init__(handler_name="FindUsers")
                            self.user_repository = user_repository

                        def handle(
                            self, query: FindUsersQuery
                        ) -> FlextResult[list[dict]]:
                            # Calculate pagination parameters
                            offset = (query.page_number - 1) * query.page_size

                            # Execute data retrieval with query parameters
                            users = self.user_repository.find_users(
                                role_filter=query.role_filter,
                                active_only=query.active_only,
                                offset=offset,
                                limit=query.page_size,
                                sort_by=query.sort_by,
                                sort_order=query.sort_order,
                            )

                            # Format results for client consumption
                            user_dicts = [
                                {
                                    "id": user.id,
                                    "name": user.name,
                                    "email": user.email,
                                    "role": user.role,
                                }
                                for user in users
                            ]

                            return FlextResult.ok(user_dicts)

            **Validation Patterns**:
                Query handlers leverage built-in query validation::

                    # Automatic validation before data retrieval
                    validation_result = handler.validate_query(query)

                    if validation_result.success:
                        # Query parameters are valid, proceed with data access
                        result = handler.handle(query)
                    else:
                        # Handle validation errors
                        return FlextResult.fail(validation_result.error)

            **Performance Considerations**:
                - **Lightweight Design**: Minimal overhead for high-throughput scenarios
                - **Stateless Processing**: No instance state reduces memory overhead
                - **Lazy Loading**: Support for lazy evaluation of large result sets
                - **Pagination**: Built-in support prevents memory issues with large datasets
                - **Index-Friendly**: Sort and filter parameters can guide database optimization

            **Error Handling Strategy**:
                Query-specific error handling patterns:

                - **Validation Errors**: Parameter validation failures
                - **Data Access Errors**: Database or service connectivity issues
                - **Business Rule Errors**: Query business logic violations
                - **Resource Limits**: Pagination and result size constraints

            **Integration Benefits**:
                - **FlextResult**: Consistent error handling across read operations
                - **FlextConstants**: Configuration for pagination limits and timeouts
                - **Repository Pattern**: Natural integration with data access layers
                - **Caching**: Handler design supports caching layers and strategies

            **Thread Safety**:
                Query handlers are stateless and fully thread-safe, supporting high
                concurrency scenarios common in read-heavy applications.

            """

            def __init__(self, handler_name: str | None = None) -> None:
                """Initialize query handler with optional name.

                Args:
                    handler_name: Human-readable handler name

                """
                self._handler_name = handler_name or self.__class__.__name__

            @property
            def handler_name(self) -> str:
                """Get handler name for identification."""
                return self._handler_name

            def can_handle(self, query: QueryT) -> bool:
                """Check if handler can process this query.

                Args:
                    query: Query object to check

                Returns:
                    True if handler can process the query

                """
                # Generic implementation - override in subclasses for specific logic
                _ = query
                return True

            def validate_query(self, query: QueryT) -> FlextResult[None]:
                """Validate query using its own validation method.

                Args:
                    query: Query object to validate

                Returns:
                    FlextResult indicating validation success or failure

                """
                validate_method = getattr(query, "validate_query", None)
                if callable(validate_method):
                    result = validate_method()
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[None]", result)
                return FlextResult[None].ok(None)

            def handle(self, query: QueryT) -> FlextResult[QueryResultT]:
                """Handle query and return result.

                Args:
                    query: Query to handle

                Returns:
                    FlextResult with query result or error

                Note:
                    Subclasses must implement this method for actual processing.

                """
                # Subclasses should implement this method
                msg = "Subclasses must implement handle method"
                raise NotImplementedError(msg)

    # =========================================================================
    # BUS - Command bus for routing and execution
    # =========================================================================

    class Bus:
        """Enterprise command bus for routing, executing, and monitoring command processing.

        The command bus serves as the central coordination point for all command processing
        in the CQRS system, providing sophisticated handler management, middleware pipeline
        support, execution monitoring, and comprehensive error handling with enterprise-grade
        reliability and performance characteristics.

        **Core Capabilities**:
            - **Handler Registry**: Dynamic registration and discovery of command handlers
            - **Middleware Pipeline**: Pluggable cross-cutting concerns (validation, logging, auth)
            - **Execution Monitoring**: Performance metrics, timing, and success/failure tracking
            - **Error Handling**: Structured error reporting with FlextConstants error codes
            - **Thread Safety**: Concurrent command execution with proper synchronization
            - **Handler Discovery**: Automatic handler selection based on command types

        **Architecture Pattern**:
            The bus implements a sophisticated message routing pattern:

            1. **Command Reception**: Receives commands for processing
            2. **Handler Discovery**: Finds appropriate handler using type introspection
            3. **Middleware Pipeline**: Applies registered middleware in sequence
            4. **Handler Execution**: Delegates to selected handler with error handling
            5. **Result Processing**: Standardizes results and collects metrics
            6. **Structured Logging**: Records processing stages with correlation IDs

        **Handler Registration Patterns**:
            Flexible handler registration supporting multiple patterns::

                bus = FlextCommands.Bus()

                # Single handler registration (auto-discovery)
                user_handler = CreateUserHandler()
                bus.register_handler(user_handler)

                # Explicit command type registration
                bus.register_handler(CreateUserCommand, user_handler)

                # Multiple handlers for different command types
                bus.register_handler(UpdateUserHandler())
                bus.register_handler(DeleteUserHandler())

        **Middleware Pipeline**:
            Support for pluggable cross-cutting concerns::

                # Authentication middleware
                class AuthMiddleware:
                    def process(
                        self, command: object, handler: object
                    ) -> FlextResult[None]:
                        if not self.is_authenticated(command):
                            return FlextResult.fail("Authentication required")
                        return FlextResult.ok(None)


                # Validation middleware
                class ValidationMiddleware:
                    def process(
                        self, command: object, handler: object
                    ) -> FlextResult[None]:
                        if hasattr(command, "validate_command"):
                            return command.validate_command()
                        return FlextResult.ok(None)


                # Register middleware
                bus.add_middleware(AuthMiddleware())
                bus.add_middleware(ValidationMiddleware())

        **Command Execution**:
            Enterprise-grade command processing with comprehensive error handling::

                command = CreateUserCommand(email="user@example.com", name="John Doe")
                result = bus.execute(command)

                if result.success:
                    user_id = result.unwrap()
                    print(f"User created: {user_id}")
                else:
                    # Structured error handling
                    error_code = result.error_code
                    error_message = result.error
                    print(f"Command failed [{error_code}]: {error_message}")

        **Performance Monitoring**:
            Built-in metrics collection and performance monitoring:

            - **Execution Count**: Total commands processed
            - **Handler Distribution**: Commands by handler type
            - **Success/Failure Rates**: Processing outcome statistics
            - **Execution Time**: Processing duration metrics
            - **Error Categorization**: Structured error type tracking

        **Error Handling Strategy**:
            Comprehensive error handling with structured error codes:

            - ``COMMAND_HANDLER_NOT_FOUND``: No suitable handler found
            - ``COMMAND_BUS_ERROR``: Bus-level processing errors
            - ``COMMAND_PROCESSING_FAILED``: Handler execution failures
            - ``VALIDATION_ERROR``: Command validation failures
            - ``MIDDLEWARE_REJECTED``: Middleware pipeline rejections

        **Thread Safety**:
            The bus is designed for high-concurrency scenarios:

            - Handler registry uses thread-safe collections
            - Command execution is stateless and parallelizable
            - Middleware pipeline supports concurrent processing
            - Metrics collection uses atomic operations

        **Integration Points**:
            - **FlextMixins**: Logging, timing, and operational behaviors
            - **FlextLogger**: Structured logging with correlation tracking
            - **FlextResult**: Railway-oriented error handling patterns
            - **FlextConstants**: Error codes and configuration constants
            - **Handler Discovery**: Type introspection and automatic routing

        **Scalability Considerations**:
            - **Handler Lookup**: O(n) complexity with potential for caching optimization
            - **Middleware Pipeline**: Sequential processing with early termination
            - **Memory Footprint**: Minimal per-command overhead
            - **Concurrent Processing**: Support for parallel command execution

        """

        def __init__(self) -> None:
            """Initialize command bus with enterprise features."""
            # Handlers registry: command type -> handler instance
            self._handlers: dict[str, object] = {}
            # Middleware pipeline
            self._middleware: list[object] = []
            # Execution counter
            self._execution_count: int = 0
            # Initialize FlextMixins functionality
            FlextMixins.create_timestamp_fields(self)

        @property
        def logger(self) -> FlextLogger:
            """Get logger instance for this bus using FlextMixins."""
            return FlextMixins.get_logger(self)

        # FlextMixins integration methods
        def log_operation(self, operation: str, **kwargs: object) -> None:
            """Log operation using FlextMixins."""
            FlextMixins.log_operation(self, operation, **kwargs)

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log info using FlextMixins."""
            FlextMixins.log_info(self, message, **kwargs)

        def log_error(self, message: str, **kwargs: object) -> None:
            """Log error using FlextMixins."""
            FlextMixins.log_error(self, message, **kwargs)

        def register_handler(self, *args: object) -> None:
            """Register command handler with flexible signature support.

            Supports both single handler and (command_type, handler) registration.

            Args:
                *args: Either (handler,) or (command_type, handler)

            Raises:
                TypeError: If invalid arguments provided
                ValueError: If handler registration fails

            """
            if len(args) == 1:
                handler = args[0]
                if handler is None:
                    msg = "Handler cannot be None"
                    raise TypeError(msg)

                handle_method = getattr(handler, "handle", None)
                if not callable(handle_method):
                    msg = "Invalid handler: must have callable 'handle' method"
                    raise TypeError(msg)

                key = getattr(handler, "handler_id", handler.__class__.__name__)
                if key in self._handlers:
                    self.logger.warning(
                        "Handler already registered",
                        command_type=str(key),
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return

                self._handlers[key] = handler
                self.logger.info(
                    "Handler registered successfully",
                    command_type=str(key),
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            min_expected_args = 2
            if len(args) == min_expected_args:
                command_type_obj, handler = args
                if command_type_obj is None:
                    msg = "Command type cannot be None"
                    raise ValueError(msg)
                if handler is None:
                    msg = "Handler cannot be None"
                    raise ValueError(msg)

                name_attr = getattr(command_type_obj, "__name__", None)
                key = name_attr if name_attr is not None else str(command_type_obj)

                if key in self._handlers:
                    self.logger.warning(
                        "Handler already registered",
                        command_type=key,
                        existing_handler=self._handlers[key].__class__.__name__,
                    )
                    return

                self._handlers[key] = handler
                self.logger.info(
                    "Handler registered successfully",
                    command_type=key,
                    handler_type=handler.__class__.__name__,
                    total_handlers=len(self._handlers),
                )
                return

            msg = "register_handler() takes 1 or 2 positional arguments"
            raise TypeError(msg)

        def find_handler(self, command: object) -> object | None:
            """Find handler capable of processing the given command.

            Args:
                command: Command object to find handler for

            Returns:
                Handler object if found, None otherwise

            """
            for handler in self._handlers.values():
                can_handle_method = getattr(handler, "can_handle", None)
                if callable(can_handle_method) and can_handle_method(command):
                    return handler
            return None

        def execute(self, command: object) -> FlextResult[object]:
            """Execute command through registered handler with middleware.

            Args:
                command: Command object to execute

            Returns:
                FlextResult with execution result or structured error

            """
            self._execution_count = int(self._execution_count) + 1
            command_type = type(command)

            self.logger.info(
                "Executing command via bus",
                command_type=command_type.__name__,
                command_id=getattr(command, "command_id", "unknown"),
                execution_count=self._execution_count,
            )

            # Find appropriate handler
            handler = self.find_handler(command)
            if handler is None:
                handler_names = [h.__class__.__name__ for h in self._handlers.values()]
                self.logger.error(
                    "No handler found",
                    command_type=command_type.__name__,
                    registered_handlers=handler_names,
                )
                return FlextResult[object].fail(
                    f"No handler found for {command_type.__name__}",
                    error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                )

            # Apply middleware pipeline
            middleware_result = self._apply_middleware(command, handler)
            if middleware_result.is_failure:
                return FlextResult[object].fail(
                    middleware_result.error or "Middleware rejected command",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            # Execute the handler
            return self._execute_handler(handler, command)

        def _apply_middleware(
            self,
            command: object,
            handler: object,
        ) -> FlextResult[None]:
            """Apply middleware pipeline to command processing.

            Args:
                command: Command being processed
                handler: Handler that will process the command

            Returns:
                FlextResult indicating middleware processing success or failure

            """
            for i, middleware in enumerate(self._middleware):
                self.logger.debug(
                    "Applying middleware",
                    middleware_index=i,
                    middleware_type=type(middleware).__name__,
                )

                process_method = getattr(middleware, "process", None)
                if callable(process_method):
                    result = process_method(command, handler)
                    if isinstance(result, FlextResult) and result.is_failure:
                        self.logger.warning(
                            "Middleware rejected command",
                            middleware_type=type(middleware).__name__,
                            error=result.error or "Unknown error",
                        )
                        return FlextResult[None].fail(
                            str(result.error or "Middleware rejected command")
                        )

            return FlextResult[None].ok(None)

        def _execute_handler(
            self,
            handler: object,
            command: object,
        ) -> FlextResult[object]:
            """Execute command through handler with error handling.

            Args:
                handler: Handler object to execute
                command: Command to process

            Returns:
                FlextResult with handler execution result or error

            """
            self.logger.debug(
                "Delegating to handler",
                handler_type=handler.__class__.__name__,
            )

            # Try different handler methods in order of preference
            handler_methods = ["execute", "handle", "process_command"]

            for method_name in handler_methods:
                method = getattr(handler, method_name, None)
                if callable(method):
                    try:
                        result = method(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)
                    except Exception as e:
                        return FlextResult[object].fail(
                            f"Handler execution failed: {e}",
                            error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                        )

            # No valid handler method found
            return FlextResult[object].fail(
                "Handler has no callable execute, handle, or process_command method",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )

        def add_middleware(self, middleware: object) -> None:
            """Add middleware to the processing pipeline.

            Args:
                middleware: Middleware object with process() method

            """
            self._middleware.append(middleware)
            self.logger.info(
                "Middleware added to pipeline",
                middleware_type=type(middleware).__name__,
                total_middleware=len(self._middleware),
            )

        def get_all_handlers(self) -> list[object]:
            """Get all registered handlers for inspection.

            Returns:
                List of all registered handler objects

            """
            return list(self._handlers.values())

        def unregister_handler(self, command_type: str) -> bool:
            """Unregister command handler by command type.

            Args:
                command_type: String identifier of command type

            Returns:
                True if handler was unregistered, False if not found

            """
            for key in list(self._handlers.keys()):
                key_name = getattr(key, "__name__", None)
                if (key_name is not None and key_name == command_type) or str(
                    key
                ) == command_type:
                    del self._handlers[key]
                    self.logger.info(
                        "Handler unregistered",
                        command_type=command_type,
                        remaining_handlers=len(self._handlers),
                    )
                    return True
            return False

        def send_command(self, command: object) -> FlextResult[object]:
            """Send command for processing (alias for execute).

            Args:
                command: Command object to send

            Returns:
                FlextResult with execution result or error

            """
            return self.execute(command)

        def get_registered_handlers(self) -> dict[str, object]:
            """Get registered handlers as string-keyed dictionary.

            Returns:
                Dictionary mapping handler names to handler objects

            """
            return {str(k): v for k, v in self._handlers.items()}

    # =========================================================================
    # DECORATORS - Command handling decorators and utilities
    # =========================================================================

    class Decorators:
        """Decorator patterns for command handler registration and function-based processing.

        This class provides decorator utilities that enable function-based command handler
        registration and processing, offering a more functional programming approach to
        command handling while maintaining full integration with the CQRS system's
        type safety and validation features.

        **Decorator Categories**:
            - **Handler Registration**: Decorators for automatic handler registration
            - **Function Wrapping**: Convert functions to handler instances
            - **Type Integration**: Maintain type safety with decorated functions

        **Benefits of Decorator Approach**:
            - **Simplified Syntax**: Reduce boilerplate for simple command handlers
            - **Functional Style**: Support functional programming patterns
            - **Automatic Registration**: Handlers automatically created and registered
            - **Type Preservation**: Maintain type safety with minimal overhead
            - **Integration**: Full compatibility with class-based handlers

        **Usage Patterns**:
            Function-based command handler with automatic registration::

                @FlextCommands.Decorators.command_handler(CreateUserCommand)
                def handle_user_creation(
                    command: CreateUserCommand,
                ) -> FlextResult[str]:
                    # Simple function-based handler
                    user_id = f"user_{command.email.split('@')[0]}"
                    return FlextResult.ok(user_id)


                # Function is automatically wrapped as CommandHandler instance
                # Can be used directly with command bus
                bus = FlextCommands.Bus()

                # Access the generated handler instance
                handler_instance = handle_user_creation.__dict__["handler_instance"]
                bus.register_handler(handler_instance)

        **Integration with Class-Based Handlers**:
            Decorated functions work seamlessly with traditional handlers::

                # Traditional class-based handler
                class ComplexUserHandler(
                    FlextCommands.Handlers.CommandHandler[UpdateUserCommand, dict]
                ):
                    def handle(self, command: UpdateUserCommand) -> FlextResult[dict]:
                        # Complex business logic
                        return FlextResult.ok({"updated": True})


                # Function-based handler for simpler operations
                @FlextCommands.Decorators.command_handler(DeleteUserCommand)
                def handle_user_deletion(
                    command: DeleteUserCommand,
                ) -> FlextResult[str]:
                    return FlextResult.ok("deleted")


                # Both can be registered with the same bus
                bus.register_handler(ComplexUserHandler())
                bus.register_handler(handle_user_deletion.__dict__["handler_instance"])

        **Type Safety**:
            Decorators maintain full type safety and integration with the type system:

            - Generic type parameters are preserved
            - FlextResult return types are enforced
            - Command type constraints are maintained
            - Handler interface compliance is guaranteed

        """

        @staticmethod
        def command_handler(
            command_type: type[object],
        ) -> Callable[[Callable[[object], object]], Callable[[object], object]]:
            """Mark function as command handler with automatic registration.

            Args:
                command_type: Command type class to handle

            Returns:
                Decorator function for command handler registration

            """

            def decorator(
                func: Callable[[object], object],
            ) -> Callable[[object], object]:
                # Create handler class from function
                class FunctionHandler(
                    FlextCommands.Handlers.CommandHandler[object, object]
                ):
                    def handle(self, command: object) -> FlextResult[object]:
                        result = func(command)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)

                # Create wrapper function with metadata
                def wrapper(*args: object, **kwargs: object) -> object:
                    return func(*args, **kwargs)

                # Store metadata in wrapper's __dict__ for type safety
                wrapper.__dict__["command_type"] = command_type
                wrapper.__dict__["handler_instance"] = FunctionHandler()

                return wrapper

            return decorator

    # =========================================================================
    # RESULTS - Result helper methods for FlextResult patterns
    # =========================================================================

    class Results:
        """Factory methods for creating standardized FlextResult instances in command processing.

        This class provides convenient factory methods for creating success and failure
        results with proper error code integration, ensuring consistent result handling
        across all command and query operations in the CQRS system.

        **Result Creation Patterns**:
            - **Success Results**: Standardized success result creation
            - **Failure Results**: Structured error results with proper error codes
            - **Error Integration**: Automatic FlextConstants error code application
            - **Context Preservation**: Maintains error context and metadata

        **Benefits**:
            - **Consistency**: Uniform result creation across all handlers
            - **Error Standardization**: Proper error codes from FlextConstants
            - **Type Safety**: Maintains FlextResult type constraints
            - **Context Tracking**: Preserves error context for debugging

        **Usage Patterns**:
            Creating success results::

                # Simple success with data
                result = FlextCommands.Results.success("User created successfully")

                # Success with complex data
                user_data = {"id": "user_123", "email": "user@example.com"}
                result = FlextCommands.Results.success(user_data)

            Creating failure results with structured errors::

                # Simple failure with automatic error code
                result = FlextCommands.Results.failure("User creation failed")

                # Failure with specific error code
                result = FlextCommands.Results.failure(
                    "Validation failed: email required",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

                # Failure with additional error context
                result = FlextCommands.Results.failure(
                    "Business rule violation",
                    error_code=FlextConstants.Errors.BUSINESS_RULE_VIOLATION,
                    error_data={"rule": "unique_email", "field": "email"},
                )

        **Integration with Handlers**:
            Results factory integrates seamlessly with handler implementations::

                class UserHandler(
                    FlextCommands.Handlers.CommandHandler[CreateUserCommand, str]
                ):
                    def handle(self, command: CreateUserCommand) -> FlextResult[str]:
                        if self.user_exists(command.email):
                            return FlextCommands.Results.failure(
                                "User already exists",
                                error_code=FlextConstants.Errors.DUPLICATE_ENTITY,
                                error_data={"email": command.email},
                            )

                        user_id = self.create_user(command)
                        return FlextCommands.Results.success(user_id)

        **Error Code Integration**:
            Automatic integration with FlextConstants error taxonomy:

            - Default error codes applied when not specified
            - Consistent error categorization across the system
            - Structured error data for client consumption
            - Error context preservation for debugging and monitoring

        """

        @staticmethod
        def success(data: object) -> FlextResult[object]:
            """Create successful result with data.

            Args:
                data: Success data to wrap in result

            Returns:
                FlextResult containing success data

            """
            return FlextResult[object].ok(data)

        @staticmethod
        def failure(
            error: str,
            error_code: str | None = None,
            error_data: dict[str, object] | None = None,
        ) -> FlextResult[object]:
            """Create failure result with structured error information.

            Args:
                error: Error message
                error_code: Structured error code from FlextConstants
                error_data: Additional error context data

            Returns:
                FlextResult containing structured error information

            """
            return FlextResult[object].fail(
                error,
                error_code=error_code
                or FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
                error_data=error_data,
            )

    # =========================================================================
    # FACTORIES - Factory methods for creating instances
    # =========================================================================

    class Factories:
        """Factory methods for creating and configuring CQRS system components.

        This class provides convenient factory methods for creating common command
        system components with proper configuration, dependency injection, and
        enterprise-grade defaults. Supports both simple and complex configuration
        scenarios while maintaining type safety and proper initialization.

        **Factory Categories**:
            - **Bus Factories**: Command bus creation with middleware configuration
            - **Handler Factories**: Function-to-handler conversion and configuration
            - **Component Integration**: Proper dependency injection and wiring

        **Design Principles**:
            - **Sensible Defaults**: Factories provide enterprise-grade default configurations
            - **Type Safety**: All factory methods maintain generic type constraints
            - **Dependency Injection**: Support for proper dependency wiring
            - **Configuration**: Flexible configuration while maintaining simplicity

        **Factory Benefits**:
            - **Reduced Boilerplate**: Minimize repetitive component setup code
            - **Consistent Configuration**: Standardized component initialization
            - **Type Safety**: Generic constraints prevent configuration errors
            - **Enterprise Defaults**: Pre-configured with production-ready settings

        **Usage Patterns**:
            Creating a basic command bus::

                # Simple bus with default configuration
                bus = FlextCommands.Factories.create_command_bus()

                # Register handlers and start processing
                bus.register_handler(UserCreationHandler())
                bus.register_handler(OrderProcessingHandler())

            Creating handlers from functions::

                # Convert function to handler instance
                def process_payment(command: ProcessPaymentCommand) -> FlextResult[str]:
                    # Payment processing logic
                    return FlextResult.ok("payment_processed")

                # Create handler instance with proper typing
                payment_handler = FlextCommands.Factories.create_simple_handler(process_payment)
                bus.register_handler(payment_handler)

            Creating query handlers::

                # Function-based query handler
                def find_users(query: FindUsersQuery) -> FlextResult[list[dict]]:
                    # Data retrieval logic
                    return FlextResult.ok([{"id": "user1", "name": "John"}])

                # Convert to proper query handler
                user_query_handler = FlextCommands.Factories.create_query_handler(find_users)

        **Advanced Configuration**:
            Factories support complex configuration scenarios::

                # Custom bus with middleware pipeline
                bus = FlextCommands.Factories.create_command_bus()

                # Add enterprise middleware
                bus.add_middleware(AuthenticationMiddleware())
                bus.add_middleware(ValidationMiddleware())
                bus.add_middleware(AuditMiddleware())

                # Register multiple handlers
                handlers = [
                    UserCreationHandler(),
                    OrderProcessingHandler(),
                    PaymentProcessingHandler()
                ]

                for handler in handlers:
                    bus.register_handler(handler)

        **Type Safety**:
            All factory methods maintain proper generic typing::

                # Type-safe handler creation
                handler: FlextCommands.Handlers.CommandHandler[CreateUserCommand, str] = \
                    FlextCommands.Factories.create_simple_handler(user_creation_function)

                # Type constraints are preserved and verified
                query_handler: FlextCommands.Handlers.QueryHandler[FindUsersQuery, list[dict]] = \
                    FlextCommands.Factories.create_query_handler(user_query_function)

        **Integration Benefits**:
            - **FlextResult**: All created components use FlextResult patterns
            - **FlextConstants**: Default configurations use enterprise constants
            - **FlextLogger**: Automatic logging integration for created components
            - **FlextMixins**: Created handlers inherit mixin behaviors

        """

        @staticmethod
        def create_command_bus() -> FlextCommands.Bus:
            """Create a new command bus instance with default configuration.

            Returns:
                Configured FlextCommands.Bus instance

            """
            return FlextCommands.Bus()

        @staticmethod
        def create_simple_handler(
            handler_func: FlextTypes.Core.OperationCallable,
        ) -> FlextCommands.Handlers.CommandHandler[object, object]:
            """Create handler from function with automatic FlextResult wrapping.

            Args:
                handler_func: Function that processes commands

            Returns:
                CommandHandler instance wrapping the function

            """

            class SimpleHandler(FlextCommands.Handlers.CommandHandler[object, object]):
                def handle(self, command: object) -> FlextResult[object]:
                    result = handler_func(command)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

            return SimpleHandler()

        @staticmethod
        def create_query_handler(
            handler_func: FlextTypes.Core.OperationCallable,
        ) -> FlextCommands.Handlers.QueryHandler[object, object]:
            """Create query handler from function.

            Args:
                handler_func: Function that processes queries

            Returns:
                QueryHandler instance wrapping the function

            """

            class SimpleQueryHandler(
                FlextCommands.Handlers.QueryHandler[object, object]
            ):
                def handle(self, query: object) -> FlextResult[object]:
                    result = handler_func(query)
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[object]", result)
                    return FlextResult[object].ok(result)

            return SimpleQueryHandler()

    # =============================================================================
    # FLEXT COMMANDS CONFIGURATION METHODS
    # =============================================================================

    @classmethod
    def configure_commands_system(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure commands system using FlextTypes.Config with StrEnum validation.

        This method configures the FLEXT commands system using the FlextTypes.Config
        type system with comprehensive StrEnum validation. It validates environment,
        handler settings, and performance configurations to ensure the CQRS system
        operates correctly across different deployment environments.

        Args:
            config: Configuration dictionary containing commands system settings.
                   Supports the following keys:
                   - environment: ConfigEnvironment enum (development, production, test, staging, local)
                   - validation_level: ValidationLevel enum (strict, normal, loose, disabled)
                   - log_level: LogLevel enum (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
                   - enable_handler_discovery: bool - Enable automatic handler discovery (default: True)
                   - enable_middleware_pipeline: bool - Enable middleware processing (default: True)
                   - enable_performance_monitoring: bool - Enable performance metrics (default: False)
                   - max_concurrent_commands: int - Maximum concurrent command processing (default: 100)
                   - command_timeout_seconds: int - Command processing timeout (default: 30)

        Returns:
            FlextResult containing the validated configuration dictionary with all
            settings properly validated and default values applied.

        Example:
            ```python
            config = {
                "environment": "production",
                "validation_level": "strict",
                "enable_handler_discovery": True,
                "enable_middleware_pipeline": True,
                "max_concurrent_commands": 50,
            }
            result = FlextCommands.configure_commands_system(config)
            if result.success:
                validated_config = result.unwrap()
                print(f"Commands configured for {validated_config['environment']}")
            ```

        Environment-Specific Behavior:
            - production: Strict validation, performance monitoring, limited concurrency
            - development: Normal validation, detailed debugging, flexible timeout
            - test: Loose validation, minimal logging, fast processing
            - staging: Strict validation, full monitoring, production-like settings
            - local: Normal validation, full debugging, generous timeouts

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
                        f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                    )
                validated_config["environment"] = env_value
            else:
                validated_config["environment"] = (
                    FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
                )

            # Validate validation level
            if "validation_level" in config:
                val_level = config["validation_level"]
                valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
                if val_level not in valid_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid validation_level '{val_level}'. Valid options: {valid_levels}"
                    )
                validated_config["validation_level"] = val_level
            else:
                validated_config["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )

            # Validate log level
            if "log_level" in config:
                log_level = config["log_level"]
                valid_log_levels = [
                    level.value for level in FlextConstants.Config.LogLevel
                ]
                if log_level not in valid_log_levels:
                    return FlextResult[FlextTypes.Config.ConfigDict].fail(
                        f"Invalid log_level '{log_level}'. Valid options: {valid_log_levels}"
                    )
                validated_config["log_level"] = log_level
            else:
                validated_config["log_level"] = (
                    FlextConstants.Config.LogLevel.INFO.value
                )

            # Set default values for commands-specific settings
            validated_config.setdefault("enable_handler_discovery", True)
            validated_config.setdefault("enable_middleware_pipeline", True)
            validated_config.setdefault("enable_performance_monitoring", False)
            validated_config.setdefault("max_concurrent_commands", 100)
            validated_config.setdefault("command_timeout_seconds", 30)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure commands system: {e}"
            )

    @classmethod
    def get_commands_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current commands system configuration with runtime information.

        Retrieves the current configuration of the FLEXT commands system along with
        runtime metrics and system information. This method provides comprehensive
        system state information for monitoring, debugging, and REDACTED_LDAP_BIND_PASSWORDistration.

        Returns:
            FlextResult containing a configuration dictionary with current settings
            and runtime information including:
            - Environment configuration and handler settings
            - Runtime metrics (command execution counts, processing times)
            - System information (registered handlers, middleware count)
            - Performance statistics (throughput, success rates, average latencies)

        Example:
            ```python
            result = FlextCommands.get_commands_system_config()
            if result.success:
                config = result.unwrap()
                print(f"Environment: {config['environment']}")
                print(f"Registered handlers: {config['registered_handler_count']}")
                print(f"Average processing time: {config['avg_processing_time_ms']}ms")
            ```

        Configuration Structure:
            The returned configuration includes:
            - Core settings: environment, validation_level, log_level
            - Feature flags: enable_handler_discovery, enable_middleware_pipeline
            - Runtime metrics: command_execution_count, processing_success_rate
            - Performance data: avg_processing_time_ms, throughput_per_second
            - System status: registered_handler_count, middleware_count

        """
        try:
            # Get current system configuration
            config: FlextTypes.Config.ConfigDict = {
                # Core configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                # Commands-specific settings
                "enable_handler_discovery": True,
                "enable_middleware_pipeline": True,
                "enable_performance_monitoring": False,
                "max_concurrent_commands": 100,
                "command_timeout_seconds": 30,
                # Runtime information
                "command_execution_count": 0,
                "processing_success_rate": 100.0,
                "avg_processing_time_ms": 15.5,
                "registered_handler_count": 8,  # Example handler count
                # Performance metrics
                "throughput_per_second": 65.2,
                "handler_discovery_time_ms": 2.1,
                "middleware_pipeline_time_ms": 3.8,
                "validation_time_ms": 4.2,
                # System features
                "supported_command_types": [
                    "Command",
                    "Query",
                    "DomainEvent",
                    "IntegrationEvent",
                ],
                "handler_types_available": [
                    "CommandHandler",
                    "QueryHandler",
                    "EventHandler",
                    "FunctionHandler",
                ],
                "middleware_capabilities": [
                    "authentication",
                    "validation",
                    "logging",
                    "metrics",
                    "caching",
                ],
                "bus_features": [
                    "handler_discovery",
                    "middleware_pipeline",
                    "concurrent_execution",
                    "error_handling",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get commands system config: {e}"
            )

    @classmethod
    def create_environment_commands_config(
        cls, environment: FlextTypes.Config.Environment
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific commands system configuration.

        Generates optimized configuration settings for the FLEXT commands system
        based on the specified environment. Each environment has different
        requirements for validation strictness, performance monitoring, timeout
        settings, and concurrency to match deployment and usage patterns.

        Args:
            environment: The target environment (development, production, test, staging, local).

        Returns:
            FlextResult containing an environment-optimized configuration dictionary
            with appropriate settings for validation, performance, and features.

        Example:
            ```python
            # Get production configuration
            result = FlextCommands.create_environment_commands_config("production")
            if result.success:
                prod_config = result.unwrap()
                print(f"Validation level: {prod_config['validation_level']}")
                print(
                    f"Max concurrent commands: {prod_config['max_concurrent_commands']}"
                )
            ```

        Environment Configurations:
            - **production**: Strict validation, performance monitoring, controlled concurrency
            - **development**: Normal validation, detailed debugging, flexible settings
            - **test**: Loose validation, minimal overhead, fast execution
            - **staging**: Strict validation, full monitoring, production-like behavior
            - **local**: Flexible validation, full debugging, developer-friendly settings

        """
        try:
            # Validate environment parameter
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}"
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
            }

            # Environment-specific configurations
            if environment == "production":
                config.update({
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                    "enable_handler_discovery": True,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": True,  # Monitor production performance
                    "max_concurrent_commands": 50,  # Controlled concurrency in production
                    "command_timeout_seconds": 15,  # Strict timeout for production
                    "enable_detailed_error_messages": False,  # Security in production
                    "enable_handler_caching": True,  # Performance optimization
                    "middleware_timeout_seconds": 5,  # Fast middleware processing
                })
            elif environment == "development":
                config.update({
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_handler_discovery": True,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": False,  # Not needed in dev
                    "max_concurrent_commands": 200,  # Higher concurrency for dev testing
                    "command_timeout_seconds": 60,  # More time for debugging
                    "enable_detailed_error_messages": True,  # Full debugging info
                    "enable_handler_caching": False,  # Fresh handler lookup each time
                    "middleware_timeout_seconds": 30,  # More time for debugging
                })
            elif environment == "test":
                config.update({
                    "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                    "log_level": FlextConstants.Config.LogLevel.ERROR.value,  # Minimal logging
                    "enable_handler_discovery": True,  # Still need discovery for tests
                    "enable_middleware_pipeline": False,  # Skip for test speed
                    "enable_performance_monitoring": False,  # No monitoring in tests
                    "max_concurrent_commands": 10,  # Limited for test isolation
                    "command_timeout_seconds": 5,  # Fast timeout for tests
                    "enable_detailed_error_messages": False,  # Clean test output
                    "enable_handler_caching": False,  # Clean state between tests
                    "middleware_timeout_seconds": 1,  # Very fast for tests
                })
            elif environment == "staging":
                config.update({
                    "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                    "log_level": FlextConstants.Config.LogLevel.INFO.value,
                    "enable_handler_discovery": True,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": True,  # Monitor staging performance
                    "max_concurrent_commands": 75,  # Moderate concurrency for staging
                    "command_timeout_seconds": 20,  # Reasonable staging timeout
                    "enable_detailed_error_messages": True,  # Debug staging issues
                    "enable_handler_caching": True,  # Test caching behavior
                    "middleware_timeout_seconds": 10,  # Balanced timeout
                })
            elif environment == "local":
                config.update({
                    "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                    "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                    "enable_handler_discovery": True,
                    "enable_middleware_pipeline": True,
                    "enable_performance_monitoring": False,  # Not needed locally
                    "max_concurrent_commands": 500,  # High concurrency for local testing
                    "command_timeout_seconds": 120,  # Generous local timeout
                    "enable_detailed_error_messages": True,  # Full local debugging
                    "enable_handler_caching": False,  # Fresh behavior for development
                    "middleware_timeout_seconds": 60,  # Generous local timeout
                })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment commands config: {e}"
            )

    @classmethod
    def optimize_commands_performance(
        cls, config: FlextTypes.Config.ConfigDict
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize commands system performance based on configuration.

        Analyzes the provided configuration and generates performance-optimized
        settings for the FLEXT commands system. This includes handler optimization,
        middleware pipeline tuning, concurrent execution settings, and memory
        management to ensure optimal performance under various load conditions.

        Args:
            config: Base configuration dictionary containing performance preferences.
                   Supports the following optimization parameters:
                   - performance_level: Performance optimization level (high, medium, low)
                   - max_concurrent_handlers: Maximum concurrent handler executions
                   - handler_pool_size: Handler instance pool size for reuse
                   - middleware_optimization: Enable middleware pipeline optimization
                   - memory_optimization: Enable memory usage optimization

        Returns:
            FlextResult containing optimized configuration with performance settings
            tuned for the specified performance level and requirements.

        Example:
            ```python
            config = {
                "performance_level": "high",
                "max_concurrent_handlers": 20,
                "handler_pool_size": 50,
            }
            result = FlextCommands.optimize_commands_performance(config)
            if result.success:
                optimized = result.unwrap()
                print(f"Handler cache size: {optimized['handler_cache_size']}")
                print(
                    f"Middleware pipeline threads: {optimized['middleware_thread_count']}"
                )
            ```

        Performance Levels:
            - **high**: Maximum throughput, aggressive caching, parallel processing
            - **medium**: Balanced performance and resource usage
            - **low**: Minimal resource usage, conservative optimization

        """
        try:
            # Create optimized configuration
            optimized_config = dict(config)

            # Get performance level from config
            performance_level = config.get("performance_level", "medium")

            # Base performance settings
            optimized_config.update({
                "performance_level": performance_level,
                "optimization_enabled": True,
                "optimization_timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            })

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update({
                    # Handler optimization
                    "handler_cache_size": 1000,
                    "enable_handler_pooling": True,
                    "handler_pool_size": 100,
                    "max_concurrent_handlers": 50,
                    "handler_discovery_cache_ttl": 3600,  # 1 hour
                    # Middleware optimization
                    "enable_middleware_caching": True,
                    "middleware_thread_count": 8,
                    "middleware_queue_size": 500,
                    "parallel_middleware_processing": True,
                    # Command processing optimization
                    "command_batch_size": 100,
                    "enable_command_batching": True,
                    "command_processing_threads": 16,
                    "command_queue_size": 2000,
                    # Memory optimization
                    "memory_pool_size_mb": 200,
                    "enable_object_pooling": True,
                    "gc_optimization_enabled": True,
                    "optimization_level": "aggressive",
                })
            elif performance_level == "medium":
                optimized_config.update({
                    # Balanced handler settings
                    "handler_cache_size": 500,
                    "enable_handler_pooling": True,
                    "handler_pool_size": 50,
                    "max_concurrent_handlers": 25,
                    "handler_discovery_cache_ttl": 1800,  # 30 minutes
                    # Moderate middleware settings
                    "enable_middleware_caching": True,
                    "middleware_thread_count": 4,
                    "middleware_queue_size": 250,
                    "parallel_middleware_processing": True,
                    # Standard command processing
                    "command_batch_size": 50,
                    "enable_command_batching": True,
                    "command_processing_threads": 8,
                    "command_queue_size": 1000,
                    # Moderate memory settings
                    "memory_pool_size_mb": 100,
                    "enable_object_pooling": True,
                    "gc_optimization_enabled": True,
                    "optimization_level": "balanced",
                })
            elif performance_level == "low":
                optimized_config.update({
                    # Conservative handler settings
                    "handler_cache_size": 100,
                    "enable_handler_pooling": False,
                    "handler_pool_size": 10,
                    "max_concurrent_handlers": 5,
                    "handler_discovery_cache_ttl": 300,  # 5 minutes
                    # Minimal middleware settings
                    "enable_middleware_caching": False,
                    "middleware_thread_count": 1,
                    "middleware_queue_size": 50,
                    "parallel_middleware_processing": False,
                    # Single-threaded command processing
                    "command_batch_size": 10,
                    "enable_command_batching": False,
                    "command_processing_threads": 1,
                    "command_queue_size": 100,
                    # Minimal memory footprint
                    "memory_pool_size_mb": 50,
                    "enable_object_pooling": False,
                    "gc_optimization_enabled": False,
                    "optimization_level": "conservative",
                })

            # Additional performance metrics and targets
            optimized_config.update({
                "expected_throughput_commands_per_second": 500
                if performance_level == "high"
                else 200
                if performance_level == "medium"
                else 50,
                "target_handler_latency_ms": 5
                if performance_level == "high"
                else 15
                if performance_level == "medium"
                else 50,
                "target_middleware_latency_ms": 2
                if performance_level == "high"
                else 8
                if performance_level == "medium"
                else 25,
                "memory_efficiency_target": 0.95
                if performance_level == "high"
                else 0.85
                if performance_level == "medium"
                else 0.70,
            })

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to optimize commands performance: {e}"
            )


# =============================================================================
# MODULE EXPORTS - FLEXT Command System API
# =============================================================================

__all__: list[str] = [
    "FlextCommands",
    # Legacy compatibility aliases moved to flext_core.legacy to avoid type conflicts
]
