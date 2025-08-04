# FLEXT Core Architecture Overview

**The Comprehensive Architectural Foundation for Enterprise Data Integration**

FLEXT Core implements Clean Architecture and Domain-Driven Design principles to provide the foundational patterns used across all 33 projects in the FLEXT ecosystem. This architectural foundation enables enterprise-grade data integration solutions with consistent patterns, type safety, and reliability.

## 🏗️ Ecosystem Architecture Context

FLEXT Core serves as the architectural foundation for the complete FLEXT ecosystem, providing essential patterns that ensure consistency across all 33 projects:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    FLEXT ECOSYSTEM (33 projects)                 │
├─────────────────────────────────────────────────────────────────┤
│ 🎯 Services (3): FlexCore(Go) | client-a | client-b                │
├─────────────────────────────────────────────────────────────────┤
│ 📱 Applications (6): API | Auth | Web | CLI | Quality | Plugin  │
├─────────────────────────────────────────────────────────────────┤
│ 🔧 Infrastructure (6): Oracle | LDAP | LDIF | WMS | gRPC | Melt. │
├─────────────────────────────────────────────────────────────────┤
│ 🔄 Singer Ecosystem (15): 5 Taps | 5 Targets | 4 DBT | 1 Ext.  │
├─────────────────────────────────────────────────────────────────┤
│ ⚡ Go Binaries (4): flext | cli | server | demo                 │
├═════════════════════════════════════════════════════════════════┤
│              FLEXT CORE - ARCHITECTURAL FOUNDATION               │
│    FlextResult | FlextContainer | Domain Patterns | Config      │
└─────────────────────────────────────────────────────────────────┘
```

## 📐 Layered Architecture Implementation

FLEXT Core implements a **6-layer architectural hierarchy** that supports Clean Architecture, Domain-Driven Design, and railway-oriented programming across the entire ecosystem:

```text
┌─────────────────────────────────────────────────────────────────┐
│                 FLEXT CORE LAYERED ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│  FOUNDATION LAYER - Type System & Core Contracts               │
│  ├── flext_types.py    # Type system foundation                │
│  ├── constants.py      # Ecosystem-wide constants              │
│  ├── version.py        # Version management & compatibility     │
│  └── __init__.py       # Public API gateway                    │
├─────────────────────────────────────────────────────────────────┤
│  CORE PATTERN LAYER - Railway-Oriented Programming             │
│  ├── result.py         # FlextResult[T] - Railway pattern      │
│  ├── container.py      # FlextContainer - DI system            │
│  ├── exceptions.py     # Exception hierarchy                   │
│  └── utilities.py      # Pure utility functions                │
├─────────────────────────────────────────────────────────────────┤
│  CONFIGURATION LAYER - System Integration & Logging            │
│  ├── config.py         # FlextBaseSettings configuration       │
│  ├── loggings.py       # Structured logging with correlation   │
│  ├── payload.py        # Message/Event/Payload patterns        │
│  └── interfaces.py     # Protocol definitions                  │
├─────────────────────────────────────────────────────────────────┤
│  DOMAIN LAYER - Domain-Driven Design Patterns                 │
│  ├── entities.py       # FlextEntity - Rich domain entities    │
│  ├── value_objects.py  # FlextValueObject - Immutable values   │
│  ├── aggregate_root.py # FlextAggregateRoot - DDD aggregates    │
│  └── domain_services.py# FlextDomainService - Domain services  │
├─────────────────────────────────────────────────────────────────┤
│  CQRS LAYER - Command Query Responsibility Segregation         │
│  ├── commands.py       # Command pattern implementation        │
│  ├── handlers.py       # Handler pattern implementation        │
│  └── validation.py     # Input validation system               │
├─────────────────────────────────────────────────────────────────┤
│  EXTENSION LAYER - Reusable Behaviors & Cross-Cutting Concerns │
│  ├── mixins.py         # Reusable behavior mixins              │
│  ├── decorators.py     # Enterprise decorator patterns         │
│  ├── fields.py         # Field metadata system                 │
│  ├── guards.py         # Validation guards & builders          │
│  └── core.py           # FlextCore main class                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architectural Layer Responsibilities

### **Layer 1: Foundation Layer** - Type System & Core Contracts

**Module Role**: Establish foundational contracts that all other modules depend on.

**Key Components**:

- **flext_types.py**: Modern type system with generics, protocols, and type variables
- **constants.py**: Ecosystem-wide constants for ports, levels, and configurations
- **version.py**: Version management and compatibility checking across ecosystem
- \***\*init**.py\*\*: Public API gateway with comprehensive exports

```python
# Foundation Layer Usage
from flext_core import FlextResult, FlextContainer, FlextEntity
from flext_core.flext_types import TAnyDict, TLogMessage
from flext_core.constants import FlextLogLevel, Platform
from flext_core.version import get_version_info, is_feature_available
```

**Ecosystem Integration**: This layer ensures type safety and compatibility across all 33 projects, providing the contracts that prevent breaking changes.

### **Layer 2: Core Pattern Layer** - Railway-Oriented Programming

**Module Role**: Provide railway-oriented programming foundation and dependency injection.

**Key Components**:

- **result.py**: FlextResult[T] pattern enabling functional error handling
- **container.py**: FlextContainer for enterprise dependency injection
- **exceptions.py**: Comprehensive exception hierarchy with business context
- **utilities.py**: Pure utility functions for performance tracking and generation

```python
# Railway-Oriented Programming Pattern
def process_data(data: dict) -> FlextResult[ProcessedData]:
    return (
        validate_input(data)
        .map(transform_data)
        .flat_map(save_to_database)
        .map(format_response)
    )

# Enterprise Dependency Injection
container = get_flext_container()
result = container.register("user_service", UserService())
service = container.get("user_service").unwrap()
```

**Ecosystem Integration**: FlextResult[T] is used in 15,000+ function signatures across all ecosystem projects, ensuring consistent error handling.

### **Layer 3: Configuration Layer** - System Integration & Logging

**Module Role**: Handle system configuration, logging, and external integration contracts.

**Key Components**:

- **config.py**: FlextBaseSettings for environment-aware configuration management
- **loggings.py**: Structured logging with correlation ID support and enterprise observability
- **payload.py**: Message/Event/Payload patterns for data exchange
- **interfaces.py**: Protocol definitions for external system integration

```python
# Configuration Management Pattern
class AppSettings(FlextBaseSettings):
    database_url: str = "postgresql://localhost/app"
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"

# Enterprise Logging Pattern
logger = get_logger(__name__, "INFO")
with create_log_context(logger, request_id="123", user_id="456"):
    logger.info("Processing request", operation="user_creation", duration_ms=45)
```

**Ecosystem Integration**: Configuration patterns are used by all infrastructure libraries (Oracle, LDAP, gRPC) for consistent environment management.

### **Layer 4: Domain Layer** - Domain-Driven Design Patterns

**Module Role**: Provide rich domain modeling patterns following DDD principles.

**Key Components**:

- **entities.py**: FlextEntity for rich domain entities with business logic
- **value_objects.py**: FlextValueObject for immutable value types
- **aggregate_root.py**: FlextAggregateRoot for DDD aggregates with invariants
- **domain_services.py**: FlextDomainService for domain service patterns

```python
# Domain Entity with Business Logic
class User(FlextEntity):
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        self.add_domain_event({"type": "UserActivated", "user_id": self.id})
        return FlextResult.ok(None)

# Immutable Value Object
class Email(FlextValueObject):
    address: str

    def __post_init__(self):
        if "@" not in self.address:
            raise ValueError("Invalid email format")
```

**Ecosystem Integration**: Domain patterns are used across all Singer projects, client-a migration, and enterprise applications for consistent business modeling.

### **Layer 5: CQRS Layer** - Command Query Responsibility Segregation

**Module Role**: Implement CQRS patterns for enterprise scalability.

**Key Components**:

- **commands.py**: Command pattern implementation with business operations
- **handlers.py**: Handler pattern implementation for command/query processing
- **validation.py**: Input validation system with business rule enforcement

```python
# CQRS Command Pattern
class CreateUserCommand(FlextCommand):
    name: str
    email: str

class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    async def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        return (
            self.validate_command(command)
            .flat_map(lambda cmd: self.create_user_entity(cmd))
            .flat_map(lambda user: self.save_user(user))
        )
```

**Ecosystem Integration**: CQRS patterns enable scalable architectures in FlexCore (Go), FLEXT Service, and high-throughput data processing.

### **Layer 6: Extension Layer** - Reusable Behaviors & Cross-Cutting Concerns

**Module Role**: Provide reusable patterns and cross-cutting concerns.

**Key Components**:

- **mixins.py**: Reusable behavior mixins for common functionality
- **decorators.py**: Enterprise decorator patterns for cross-cutting concerns
- **fields.py**: Field metadata system for enhanced data modeling
- **guards.py**: Validation guards and builders for type safety
- **core.py**: FlextCore main class integrating all patterns

```python
# Mixin Pattern for Reusable Behavior
class User(FlextEntity, TimestampMixin, SoftDeleteMixin):
    name: str
    email: str
    # Automatically inherits: created_at, updated_at, deleted_at, is_deleted

# Cross-Cutting Concerns with Decorators
@with_correlation_id
@with_performance_tracking("user_service.create_user")
def create_user(self, data: dict) -> FlextResult[User]:
    return self.validate(data).flat_map(self._create_user_impl)
```

**Ecosystem Integration**: Extension patterns reduce code duplication across all 33 projects while maintaining architectural consistency.

## Design Principles

### Principles Implementation

**Single Responsibility Principle**:

- Each class has one reason to change
- Clear separation of concerns across layers
- Focused interfaces and implementations

**Open/Closed Principle**:

- Extensible through inheritance and composition
- Plugin architecture for domain services
- Strategy pattern for varying behaviors

**Liskov Substitution Principle**:

- Consistent interfaces across implementations
- Type-safe substitution with FlextResult
- Behavioral contracts maintained

**Interface Segregation Principle**:

- Small, focused interfaces
- Client-specific abstractions
- No forced dependencies on unused methods

**Dependency Inversion Principle**:

- Depend on abstractions, not concretions
- Infrastructure implements domain interfaces
- Inward-pointing dependencies only

### Clean Architecture Compliance

**Dependency Rule**: Dependencies point inward only. Outer layers depend on inner layers, never the reverse.

**Source Code Dependencies**: All source code dependencies point inward toward higher-level policies.

**Data Flow**: Data crosses boundaries in simple data structures or through well-defined interfaces.

**Framework Independence**: The architecture does not depend on frameworks or external tools.

## Error Handling Strategy

### FlextResult Pattern

All operations return `FlextResult[T]` for explicit error handling:

```python
def complex_operation() -> FlextResult[ProcessedData]:
    return (
        validate_input()
        .flat_map(process_data)
        .flat_map(save_results)
        .map(format_output)
    )

# Usage
result = complex_operation()
if result.success:
    handle_success(result.data)
else:
    handle_error(result.error)
```

**Benefits**:

- Explicit error handling in type signatures
- Functional composition with map/flat_map
- No hidden exceptions or side effects
- Testable error paths

### Error Propagation

Errors propagate through the application layers:

1. **Domain Layer**: Business rule violations
2. **Application Layer**: Workflow coordination errors
3. **Infrastructure Layer**: Technical failures
4. **Presentation Layer**: Input validation errors

## Testing Strategy

### Layer-Specific Testing

**Domain Layer Testing**:

```python
def test_user_activation():
    user = User.create("John Doe", "john@example.com").unwrap()

    result = user.activate()

    assert result.success
    assert user.is_active
    assert len(user.get_domain_events()) == 1
```

**Application Layer Testing**:

```python
def test_user_registration_workflow():
    service = UserApplicationService(mock_container)
    command = RegisterUserCommand(name="John", email="john@example.com")

    result = service.register_user(command)

    assert result.success
    assert isinstance(result.data, User)
```

**Infrastructure Layer Testing**:

```python
def test_container_service_registration():
    container = FlextContainer()
    service = UserService()

    result = container.register("user_service", service)

    assert result.success
    retrieved = container.get("user_service")
    assert retrieved.success
    assert retrieved.data is service
```

## Performance Considerations

### Memory Management

- Immutable value objects reduce memory fragmentation
- Lazy loading of services through FlextContainer
- Event cleanup after processing domain events

### Execution Performance

- Type-safe interfaces eliminate runtime type checking
- FlextResult eliminates exception handling overhead
- Dependency injection reduces object creation costs

### Scalability Patterns

- Stateless service design enables horizontal scaling
- Event-driven architecture supports loose coupling
- Clear layer boundaries enable independent scaling

## Integration Points

### External Systems

The architecture provides clear integration points:

**Inbound Adapters** (Primary/Driving):

- REST API controllers
- CLI command handlers
- Message queue consumers

**Outbound Adapters** (Secondary/Driven):

- Database repositories
- External service clients
- File system adapters

### Framework Integration

FLEXT Core integrates with popular frameworks:

- **FastAPI**: Type-safe API development
- **Django**: Enterprise web applications
- **SQLAlchemy**: Database integration
- **Pydantic**: Data validation and serialization

This architectural foundation provides the stability and flexibility needed for enterprise applications while maintaining clean separation of concerns and testability.
