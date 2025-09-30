# FLEXT-CORE Comprehensive Docstring Template Guide

**Purpose**: Google-style docstring templates for all flext-core classes
**Standard**: PEP 8 (79 char line limit) + Google semantic style
**Sections**: Function, Uses, How to use, TODO

---

## Template Structure

```python
class ClassName:
    """Brief one-line summary describing primary purpose (max 79 chars).

    Extended multi-paragraph description explaining the class role in the
    FLEXT ecosystem, architectural patterns it implements, and key design
    decisions. Provides context for developers using this class.

    **Function**: What this class does
        - Primary responsibility and use case
        - Secondary features and capabilities
        - Integration points with ecosystem

    **Uses**: Dependencies and internal mechanisms
        - FlextResult[T] for railway-oriented error handling
        - FlextContainer for dependency injection
        - Internal caching/optimization strategies
        - Third-party libraries (if any)

    **How to use**: Practical usage examples
        ```python
        from flext_core import ClassName

        # Example 1: Basic instantiation and usage
        instance = ClassName(param="value")
        result = instance.primary_method()

        # Example 2: Error handling with FlextResult
        if result.is_success:
            data = result.unwrap()
            # Process successful data
        else:
            print(f"Error: {result.error}")

        # Example 3: Advanced usage pattern
        with instance.context_manager():
            # Perform operations
            pass
        ```

    **TODO**: Missing functionality and improvements
        - [ ] Add async/await support for concurrent operations
        - [ ] Enhance validation with custom rule engine
        - [ ] Implement caching layer for performance
        - [ ] Add telemetry and metrics collection
        - [ ] Improve error messages with context

    Args:
        param_name (type): Description of parameter purpose and
            constraints. Use newlines for long descriptions.
        optional_param (type, optional): Description. Defaults to None.

    Attributes:
        attribute_name (type): Description of instance attribute.
        _private_attr (type): Description of private attribute.

    Raises:
        ValueError: When invalid parameters are provided.
        FlextExceptions.ValidationError: When validation fails.

    Returns:
        ReturnType: Description of what methods return.

    Yields:
        YieldType: For generator methods, describe yielded values.

    Note:
        Important usage notes, limitations, or performance considerations.
        Multiple notes can be included.

    Warning:
        Critical warnings about misuse, breaking changes, or deprecated
        functionality.

    Example:
        Detailed example showing real-world usage:

        >>> instance = ClassName(param="test")
        >>> result = instance.method()
        >>> print(result.is_success)
        True

    See Also:
        RelatedClass: Brief description of relationship.
        AlternativeClass: When to use this alternative.
        https://docs.flext.io/classname: External documentation.

    """
```

---

## Module-Specific Templates

### 1. Foundation Layer: result.py - FlextResult[T]

```python
class FlextResult[T]:
    """Railway-oriented result type for type-safe error handling.

    FlextResult[T] implements the railway pattern (Either monad) for
    explicit error handling throughout the FLEXT ecosystem. Replaces
    try/except patterns with composable success/failure workflows.

    **Function**: Type-safe success/failure wrapper
        - Wraps operation results with explicit success/failure state
        - Provides monadic operations (map, flat_map, filter)
        - Maintains dual API (.data/.value) for compatibility
        - Enables railway-oriented programming patterns

    **Uses**: Pure Python foundation (no external dependencies)
        - Generic type variable T for type safety
        - Immutable result state
        - Internal caching for performance
        - Descriptor pattern for dual access

    **How to use**: Basic and advanced patterns
        ```python
        from flext_core import FlextResult

        # Basic usage: Create success/failure results
        success = FlextResult[str].ok("data")
        failure = FlextResult[str].fail("error occurred")

        # Check result state
        if success.is_success:
            value = success.unwrap()  # Safe extraction

        # Railway composition (monadic chaining)
        result = (
            validate_input(data)
            .flat_map(lambda d: process_data(d))
            .map(lambda d: format_output(d))
            .map_error(lambda e: log_error(e))
        )

        # Dual API compatibility (ecosystem requirement)
        assert result.value == result.data  # Both work
        ```

    **TODO**: Enhancements for 1.0.0+
        - [ ] Add async/await support for FlextResult[Awaitable[T]]
        - [ ] Implement result combination (sequence, traverse)
        - [ ] Add error context with stack traces
        - [ ] Support custom error types beyond strings
        - [ ] Add performance metrics collection
    """
```

### 2. Foundation Layer: container.py - FlextContainer

```python
class FlextContainer:
    """Global dependency injection container for FLEXT ecosystem.

    FlextContainer provides centralized service management using the
    singleton pattern. Access via FlextContainer.get_global() for
    consistent dependency injection throughout applications.

    **Function**: Service registry and dependency injection
        - Register services and factories globally
        - Resolve dependencies with type safety
        - Support auto-wiring of constructor dependencies
        - Provide batch operations with rollback
        - Enable thread-safe singleton access

    **Uses**: Core infrastructure components
        - FlextResult[T] for operation results
        - FlextConfig for container configuration
        - FlextModels.Validation for service name validation
        - threading.Lock for singleton thread safety
        - inspect module for dependency resolution

    **How to use**: Service registration and retrieval
        ```python
        from flext_core import FlextContainer, FlextLogger

        # Get global singleton instance
        container = FlextContainer.get_global()

        # Register services
        logger = FlextLogger(__name__)
        container.register("logger", logger)

        # Register factories (lazy instantiation)
        container.register_factory(
            "database",
            lambda: DatabaseService()
        )

        # Retrieve services with type safety
        logger_result = container.get_typed(
            "logger",
            FlextLogger
        )
        if logger_result.is_success:
            logger = logger_result.unwrap()

        # Auto-wire dependencies
        service_result = container.create_service(
            MyService,  # Auto-resolves constructor params
            service_name="my_service"
        )
        ```

    **TODO**: Advanced DI features
        - [ ] Add scoped lifetimes (request, transient, singleton)
        - [ ] Implement circular dependency detection
        - [ ] Support decorator-based registration
        - [ ] Add container hierarchies (parent/child)
        - [ ] Enhance auto-wiring with interface binding
    """
```

### 3. Domain Layer: models.py - FlextModels

```python
class FlextModels:
    """Domain-Driven Design (DDD) model patterns for FLEXT ecosystem.

    FlextModels provides base classes for implementing DDD patterns:
    Entities, Value Objects, and Aggregate Roots. Use these for domain
    modeling with built-in validation and event sourcing support.

    **Function**: DDD pattern implementations
        - Entity base with identity and lifecycle
        - Value Object base for immutable values
        - Aggregate Root for consistency boundaries
        - Domain event management
        - Validation utilities

    **Uses**: Pydantic for validation
        - BaseModel for all domain models
        - Field validators for business rules
        - model_config for Pydantic settings
        - FlextResult[T] for operation results

    **How to use**: Implement domain models
        ```python
        from flext_core import FlextModels, FlextResult

        # Value Object (immutable, compared by value)
        class Email(FlextModels.Value):
            address: str

            def validate(self) -> FlextResult[None]:
                if "@" not in self.address:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)

        # Entity (has identity)
        class User(FlextModels.Entity):
            name: str
            email: Email

            def activate(self) -> FlextResult[None]:
                if self.is_active:
                    return FlextResult[None].fail("Already active")
                self.is_active = True
                self.add_domain_event(
                    "UserActivated",
                    {"user_id": self.id}
                )
                return FlextResult[None].ok(None)

        # Aggregate Root (consistency boundary)
        class Account(FlextModels.AggregateRoot):
            owner: User
            balance: Decimal

            def withdraw(self, amount: Decimal) -> FlextResult[None]:
                if amount > self.balance:
                    return FlextResult[None].fail(
                        "Insufficient funds"
                    )
                self.balance -= amount
                self.add_domain_event(
                    "MoneyWithdrawn",
                    {"amount": str(amount)}
                )
                return FlextResult[None].ok(None)
        ```

    **TODO**: Enhanced DDD support
        - [ ] Add domain event versioning
        - [ ] Implement event store patterns
        - [ ] Support aggregate snapshots
        - [ ] Add saga orchestration
        - [ ] Enhance validation DSL
    """
```

### 4. Application Layer: bus.py - FlextBus

```python
class FlextBus:
    """Command/Query bus for CQRS pattern implementation.

    FlextBus provides message dispatching for Command Query
    Responsibility Segregation (CQRS) patterns. Routes commands and
    queries to registered handlers with middleware support.

    **Function**: Message bus with middleware pipeline
        - Register command and query handlers
        - Execute handlers with middleware chain
        - Support caching for query results
        - Provide handler discovery
        - Enable middleware-based cross-cutting concerns

    **Uses**: CQRS infrastructure
        - FlextHandlers for handler execution
        - FlextConfig for bus configuration
        - FlextLogger for operation logging
        - Internal cache for query results
        - Middleware pipeline for processing

    **How to use**: Command/query dispatch
        ```python
        from flext_core import FlextBus, FlextResult

        # Create bus instance
        bus = FlextBus()

        # Define command handler
        class CreateUserHandler:
            def handle(self, cmd: CreateUserCommand):
                # Process command
                return FlextResult[User].ok(user)

        # Register handler
        bus.register_handler(
            "CreateUser",
            CreateUserHandler()
        )

        # Execute command
        result = bus.execute(
            "CreateUser",
            CreateUserCommand(name="John")
        )

        # Add middleware for logging
        def logging_middleware(message, next_handler):
            logger.info(f"Executing: {message}")
            result = next_handler(message)
            logger.info(f"Completed: {result.is_success}")
            return result

        bus.add_middleware(logging_middleware)
        ```

    **TODO**: Bus enhancements
        - [ ] Add distributed bus support (Redis, RabbitMQ)
        - [ ] Implement message versioning
        - [ ] Support saga orchestration
        - [ ] Add priority queues
        - [ ] Enhance middleware composition
    """
```

---

## Implementation Checklist Per Class

For each class in every module, ensure:

- [ ] **One-line summary** (≤79 chars)
- [ ] **Extended description** (architectural context)
- [ ] **Function section** (what it does, 3-5 bullet points)
- [ ] **Uses section** (dependencies, mechanisms)
- [ ] **How to use section** (3+ code examples)
- [ ] **TODO section** (5+ actionable improvements)
- [ ] **Args/Attributes** (complete documentation)
- [ ] **Returns/Raises** (comprehensive coverage)
- [ ] **Note/Warning** (important caveats)
- [ ] **See Also** (cross-references)
- [ ] **Line length ≤79** (PEP 8 compliance)

---

## All 19 Modules Summary

### Foundation Layer (Layers 0-3)
1. **typings.py** - Type definitions and aliases
2. **constants.py** - Ecosystem-wide constants
3. **exceptions.py** - Exception hierarchy
4. **result.py** - Railway pattern implementation

### Domain Layer (Layer 4)
5. **models.py** - DDD patterns (Entity, Value, Aggregate)
6. **utilities.py** - Validation and processing utilities
7. **config.py** - Configuration management
8. **loggings.py** - Structured logging

### Application Layer (Layer 5)
9. **bus.py** - Command/Query bus
10. **handlers.py** - Handler base classes
11. **cqrs.py** - CQRS decorators and helpers
12. **dispatcher.py** - Message dispatching

### Infrastructure Layer (Layer 6)
13. **protocols.py** - Protocol definitions
14. **context.py** - Context management
15. **mixins.py** - Reusable mixins
16. **container.py** - Dependency injection

### Service Layer (Layer 7)
17. **registry.py** - Handler registry
18. **service.py** - Domain service base
19. **processors.py** - Processing pipelines

---

## Quick Reference: Common Patterns

### Pattern 1: FlextResult Operations
```python
# Always return FlextResult for operations
def operation(data: dict) -> FlextResult[ProcessedData]:
    if not data:
        return FlextResult[ProcessedData].fail("Data required")
    return FlextResult[ProcessedData].ok(processed_data)
```

### Pattern 2: Container Access
```python
# Always use global singleton
container = FlextContainer.get_global()
result = container.get("service_name")
```

### Pattern 3: Logger Usage
```python
# Direct instantiation
logger = FlextLogger(__name__)
logger.info("Operation completed")
```

### Pattern 4: Config Access
```python
# Global configuration
config = FlextConfig.get_global_instance()
timeout = config.timeout_seconds
```

---

**Next Steps**: Apply this template to all 19 modules systematically,
starting with Foundation Layer and progressing through each layer.