# FLEXT-Core Architecture

Technical architecture overview for the FLEXT-Core foundation library.

## Overview

FLEXT-Core implements Clean Architecture principles with domain-driven design patterns, providing a solid foundation for the FLEXT ecosystem's 33 interconnected projects.

### Architectural Principles

1. **Dependency Inversion**: Higher-level modules don't depend on lower-level modules
2. **Single Responsibility**: Each module has one reason to change
3. **Interface Segregation**: Clients depend only on interfaces they use
4. **Railway-Oriented Programming**: Explicit error handling without exceptions
5. **Domain-Driven Design**: Rich domain models with business logic

## Layer Architecture

```mermaid
graph TB
    subgraph "FLEXT-Core Foundation"
        subgraph "Foundation Layer"
            Result[FlextResult[T]]
            Container[FlextContainer]
            Types[FlextTypes]
            Constants[FlextConstants]
        end

        subgraph "Domain Layer"
            Models[FlextModels]
            DomainServices[FlextDomainService]
            Validations[FlextValidations]
        end

        subgraph "Application Layer"
            Commands[FlextCommands]
            Handlers[FlextHandlers]
            Guards[FlextGuards]
        end

        subgraph "Infrastructure Layer"
            Config[FlextConfig]
            Logging[FlextLogger]
            Protocols[FlextProtocols]
        end

        subgraph "Support Layer"
            Utilities[FlextUtilities]
            Decorators[FlextDecorators]
            Mixins[FlextMixins]
        end
    end

    subgraph "Ecosystem Projects"
        API[flext-api]
        CLI[flext-cli]
        DB[flext-db-oracle]
        LDAP[flext-ldap]
        Web[flext-web]
        Singer[Singer Projects]
    end

    Foundation --> Domain
    Domain --> Application
    Application --> Infrastructure

    Result --> API
    Container --> CLI
    Models --> DB
    Config --> LDAP
    Logging --> Web
    Commands --> Singer
```

## Module Organization

### Foundation Layer (Core Patterns)

**Purpose**: Provide fundamental patterns with no dependencies

```python
# src/flext_core/result.py - Railway pattern
class FlextResult[T]:
    """Monadic result type for railway-oriented programming."""
    def ok(value: T) -> FlextResult[T]: ...
    def fail(error: str) -> FlextResult[T]: ...
    def map(self, func: Callable[[T], U]) -> FlextResult[U]: ...
    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]: ...

# src/flext_core/container.py - Dependency injection
class FlextContainer:
    """Type-safe dependency injection container."""
    def register(self, key: str, service: Any) -> FlextResult[None]: ...
    def get(self, key: str) -> FlextResult[Any]: ...

    @classmethod
    def get_global(cls) -> 'FlextContainer': ...

# src/flext_core/typings.py - Type definitions
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
T_co = TypeVar('T_co', covariant=True)
```

**Key Files**:
- `result.py`: FlextResult[T] railway pattern (11,996 total lines across all modules)
- `container.py`: Dependency injection with singleton pattern
- `exceptions.py`: Exception hierarchy with error codes
- `constants.py`: System constants and enums
- `typings.py`: Type variables and aliases

### Domain Layer (Business Logic)

**Purpose**: Domain-driven design patterns and business rules

```python
# src/flext_core/models.py - Domain models
class FlextModels:
    class Entity(BaseModel):
        """Entity with identity and lifecycle."""
        id: str = Field(default_factory=lambda: generate_id("entity"))

        def add_domain_event(self, event_type: str, data: dict) -> None: ...

    class Value(BaseModel):
        """Immutable value object."""
        class Config:
            frozen = True

    class AggregateRoot(Entity):
        """Consistency boundary for domain operations."""
        _domain_events: list[dict] = PrivateAttr(default_factory=list)

# src/flext_core/domain_services.py - Service patterns
class FlextDomainService(BaseModel):
    """Base class for domain services."""
    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._container = FlextContainer.get_global()
```

**Key Files**:
- `models.py`: Entity, Value Object, Aggregate Root patterns
- `domain_services.py`: Domain service base classes
- `validations.py`: Business rule validation patterns

### Application Layer (Use Cases)

**Purpose**: Application-specific business logic and coordination

```python
# src/flext_core/commands.py - CQRS patterns
class FlextCommand(BaseModel):
    """Base command for CQRS operations."""
    command_id: str = Field(default_factory=lambda: generate_uuid())
    timestamp: datetime = Field(default_factory=datetime.now)

# src/flext_core/handlers.py - Command/query handlers
class FlextHandler[TRequest, TResponse]:
    """Generic handler pattern."""
    def handle(self, request: TRequest) -> FlextResult[TResponse]: ...
```

**Key Files**:
- `commands.py`: Command patterns for CQRS
- `processing.py`: Handler patterns and execution
- `guards.py`: Type guards and validation decorators

### Infrastructure Layer (External Concerns)

**Purpose**: External integrations and cross-cutting concerns

```python
# src/flext_core/config.py - Configuration management
class FlextConfig(BaseSettings):
    """Pydantic-based configuration with environment support."""
    class Config:
        env_file = ".env"
        case_sensitive = False

# src/flext_core/loggings.py - Structured logging
class FlextLogger:
    """Structured logger with contextual information."""
    def __init__(self, name: str) -> None: ...
    def info(self, message: str, **kwargs) -> None: ...
    def error(self, message: str, **kwargs) -> None: ...
```

**Key Files**:
- `config.py`: FlextConfig with Pydantic Settings
- `loggings.py`: Structured logging with Structlog
- `protocols.py`: Interface definitions and contracts
- `context.py`: Request/operation context management

## Design Patterns

### 1. Railway-Oriented Programming

**Problem**: Exception handling creates unpredictable control flow
**Solution**: Explicit success/failure paths with composable operations

```python
def process_user_registration(data: dict) -> FlextResult[User]:
    """Complete user registration with railway pattern."""
    return (
        validate_registration_data(data)
        .flat_map(lambda d: create_user_account(d))
        .flat_map(lambda u: send_welcome_email(u))
        .flat_map(lambda u: log_registration_event(u))
        .map(lambda u: enrich_user_profile(u))
    )

# Each step can fail, but success path flows naturally
# Failure at any step short-circuits the pipeline
```

**Benefits**:
- Explicit error handling
- Composable operations
- Type-safe error propagation
- No hidden exceptions

### 2. Dependency Injection Pattern

**Problem**: Tight coupling between services and their dependencies
**Solution**: Inversion of control with type-safe service container

```python
class UserService:
    def __init__(self):
        container = FlextContainer.get_global()

        # Services are injected, not created directly
        self._db = container.get("database").unwrap()
        self._email = container.get("email_service").unwrap()
        self._logger = container.get("logger").unwrap()

    def register_user(self, data: dict) -> FlextResult[User]:
        # Use injected dependencies
        user = self._create_user(data)
        self._db.save_user(user)
        self._email.send_welcome(user)
        self._logger.info("User registered", user_id=user.id)
        return FlextResult.ok(user)
```

**Benefits**:
- Loose coupling
- Testability (easy mocking)
- Configuration flexibility
- Single global container

### 3. Domain-Driven Design Patterns

**Problem**: Business logic scattered across services and controllers
**Solution**: Rich domain models with encapsulated business rules

```python
class Order(FlextModels.AggregateRoot):
    """Order aggregate root with business invariants."""
    customer_id: str
    items: list[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    total: Decimal = Decimal("0.00")

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item with business rule validation."""
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")

        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        if product.stock < quantity:
            return FlextResult.fail(f"Insufficient stock: {product.stock}")

        item = OrderItem(product=product, quantity=quantity)
        self.items.append(item)
        self.total += item.line_total()

        self.add_domain_event("ItemAdded", {
            "order_id": self.id,
            "product_id": product.id,
            "quantity": quantity
        })

        return FlextResult.ok(None)
```

**Benefits**:
- Business logic centralized in domain
- Invariants enforced at model level
- Rich, expressive domain language
- Clear boundaries and responsibilities

## Error Handling Strategy

### Railway Pattern Implementation

```python
class FlextResult[T]:
    """Result type with success/failure states."""

    def __init__(self, value: T | None = None, error: str | None = None):
        self._value = value
        self._error = error
        self._success = error is None

    @property
    def success(self) -> bool:
        return self._success

    @property
    def is_failure(self) -> bool:
        return not self._success

    def unwrap(self) -> T:
        """Extract value after success check."""
        if self._success:
            return self._value
        raise ValueError(f"Cannot unwrap failure: {self._error}")

    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value."""
        if self._success:
            try:
                return FlextResult.ok(func(self._value))
            except Exception as e:
                return FlextResult.fail(str(e))
        return FlextResult.fail(self._error)

    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return FlextResult."""
        if self._success:
            return func(self._value)
        return FlextResult.fail(self._error)
```

### Error Propagation

```python
def complex_business_operation(data: dict) -> FlextResult[ProcessedData]:
    """Multiple steps with error propagation."""

    # Each step can fail, errors propagate automatically
    validation_result = validate_input(data)
    if validation_result.is_failure:
        return validation_result

    # Railway pattern chains operations
    return (
        validation_result
        .flat_map(lambda d: enrich_data(d))
        .flat_map(lambda d: apply_business_rules(d))
        .flat_map(lambda d: persist_data(d))
        .map(lambda d: create_response(d))
    )

    # Success path: all operations succeed
    # Failure path: first failure terminates chain
```

## Performance Considerations

### Dependency Injection Container

- **Singleton Pattern**: Global container prevents multiple instances
- **Type Safety**: Compile-time type checking reduces runtime errors
- **Lazy Loading**: Services created only when first requested

### Railway Pattern Optimization

- **Early Returns**: Failures short-circuit operation chains
- **Memory Efficiency**: Only one result object per operation
- **Exception Avoidance**: No try/catch overhead in business logic

## Integration with Ecosystem

### Foundation for 33 Projects

FLEXT-Core provides consistent patterns used across:

1. **Infrastructure Projects** (8): Database, LDAP, gRPC, authentication
2. **Application Services** (5): API, CLI, web interfaces
3. **Singer Platform** (15): Data integration taps and targets
4. **Specialized Tools** (5): Quality tools, migration utilities

### API Compatibility

```python
# All ecosystem projects use consistent imports
from flext_core import FlextResult, FlextContainer, FlextModels

# Consistent error handling across ecosystem
def ecosystem_operation(data: dict) -> FlextResult[ProcessedData]:
    return FlextResult.ok(processed_data)

# Consistent dependency injection
container = FlextContainer.get_global()
container.register("service", service_instance)
```

### Ecosystem Standards

- **Error Handling**: All operations return FlextResult[T]
- **Service Management**: FlextContainer for dependency injection
- **Domain Modeling**: FlextModels for entities and value objects
- **Configuration**: FlextConfig for environment-aware settings
- **Logging**: FlextLogger for structured logging

## Testing Architecture

### Test Organization

```
tests/
├── unit/                    # Fast, isolated unit tests
│   ├── test_result.py       # Railway pattern tests
│   ├── test_container.py    # Dependency injection tests
│   └── test_models.py       # Domain model tests
├── integration/             # Cross-module integration tests
├── performance/             # Performance and load tests
└── conftest.py             # Shared fixtures and utilities
```

### Testing Patterns

```python
# Unit testing with FlextResult
def test_railway_pattern():
    """Test success and failure paths."""
    success = FlextResult[str].ok("success")
    assert success.success
    assert success.unwrap() == "success"

    failure = FlextResult[str].fail("error")
    assert failure.is_failure
    assert failure.error == "error"

# Integration testing with clean container
def test_service_integration(clean_container):
    """Test service with injected dependencies."""
    # Setup dependencies
    clean_container.register("database", MockDatabase())
    clean_container.register("logger", MockLogger())

    # Test service
    service = UserService()  # Uses injected dependencies
    result = service.create_user(valid_user_data)
    assert result.success
```

## Metrics and Monitoring

### Current Implementation Status

- **Modules**: 23 Python modules
- **Classes**: 22 total classes (unified class pattern)
- **Methods**: 700+ methods across all modules
- **Lines of Code**: 11,996 total lines
- **Test Coverage**: 84% (2,271 test cases)
- **Type Coverage**: Complete type annotations

### Quality Metrics

- **MyPy Strict**: Zero errors in src/ directory
- **PyRight**: Zero type errors
- **Ruff**: Zero code quality violations
- **Security**: Zero critical vulnerabilities (Bandit)

---

**Architecture Summary**: FLEXT-Core implements proven architectural patterns with railway-oriented programming, dependency injection, and domain-driven design to provide a solid foundation for the entire FLEXT ecosystem.