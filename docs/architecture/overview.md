# FLEXT Core Architecture

Architectural foundation based on Clean Architecture and
Domain-Driven Design principles.

## Architectural Overview

FLEXT Core implements layered architecture promoting separation of
concerns, testability, and maintainability. Follows SOLID principles
and patterns.

```text
┌─────────────────────────────────────────────────────────────┐
│                    FLEXT CORE ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────┤
│  PRESENTATION LAYER (External APIs & Interfaces)           │
│  ├── Public API (FlextResult, FlextContainer, etc.)        │
│  └── Type Definitions (Modern FlextXxx types)              │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER (Business Logic Orchestration)          │
│  ├── Commands & Handlers (CQRS Pattern)                    │
│  ├── Application Services                                   │
│  └── Validation & Business Rules                           │
├─────────────────────────────────────────────────────────────┤
│  DOMAIN LAYER (Business Core)                              │
│  ├── Entities (FlextEntity)                                │
│  ├── Value Objects (FlextValueObject)                      │
│  ├── Aggregates (FlextAggregateRoot)                       │
│  └── Domain Services                                       │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER (Technical Implementations)          │
│  ├── Dependency Injection (FlextContainer)                 │
│  ├── Configuration Management                              │
│  ├── Logging & Monitoring                                  │
│  └── Cross-cutting Concerns                                │
└─────────────────────────────────────────────────────────────┘
```

## Architectural Layers

### 1. Presentation Layer - Public Interface

**Responsibility**: Expose clean, type-safe APIs for external consumption.

```python
# Main Public API
from flext_core import (
    FlextResult,      # Type-safe error handling
    FlextContainer,   # Dependency injection
    FlextEntity,      # Domain entities
    FlextValueObject, # Immutable values
    FlextCommand,     # Command pattern
    FlextHandler,     # Message processing
)
```

**Key Features**:

- Type-safe interfaces using Python 3.13 generics
- Consistent error handling with FlextResult[T]
- Clear separation between public and internal APIs
- Comprehensive type definitions for IDE support

### 2. Application Layer - Business Logic Orchestration

**Responsibility**: Coordinate business workflows and enforce business rules.

```python
# Application Services
class UserApplicationService:
    def __init__(self, container: FlextContainer):
        self.user_repo = container.get("user_repository").unwrap()
        self.email_service = container.get("email_service").unwrap()
    
    def register_user(self, command: RegisterUserCommand) -> FlextResult[User]:
        return (
            self._validate_registration(command)
            .flat_map(self._create_user)
            .flat_map(self._send_welcome_email)
        )
```

**Components**:

- **Commands**: Encapsulate business operations
- **Handlers**: Process commands and queries
- **Application Services**: Orchestrate domain operations
- **Validation**: Business rule enforcement

### 3. Domain Layer - Business Core

**Responsibility**: Model the business domain with rich behavior and invariants.

```python
# Domain Entities
class User(FlextEntity):
    name: str = Field(..., description="User full name")
    email: str = Field(..., description="User email address")
    
    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("User already active")
        
        self.is_active = True
        self.add_domain_event(UserActivatedEvent(user_id=self.id))
        return FlextResult.ok(None)

# Value Objects
class Email(FlextValueObject):
    value: str = Field(..., description="Email address")
    
    @field_validator("value")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v
```

**Domain Patterns**:

- **Entities**: Objects with identity and lifecycle
- **Value Objects**: Immutable objects with value-based equality
- **Aggregates**: Consistency boundaries with business invariants
- **Domain Events**: Capture business-significant occurrences

### 4. Infrastructure Layer - Technical Implementation

**Responsibility**: Provide technical capabilities and external integrations.

```python
# Dependency Injection
container = get_flext_container()

# Service Registration
registration_result = container.register("user_service", UserService())
if registration_result.is_failure:
    handle_registration_error(registration_result.error)

# Service Retrieval
user_service = container.get("user_service")
if user_service.is_success:
    process_with_service(user_service.data)
```

**Infrastructure Components**:

- **FlextContainer**: Type-safe dependency injection
- **Configuration**: Environment-aware settings management  
- **Logging**: Structured logging with context
- **Monitoring**: Metrics and health checks

## Design Principles

### SOLID Principles Implementation

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
if result.is_success:
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
    
    assert result.is_success
    assert user.is_active
    assert len(user.get_domain_events()) == 1
```

**Application Layer Testing**:

```python
def test_user_registration_workflow():
    service = UserApplicationService(mock_container)
    command = RegisterUserCommand(name="John", email="john@example.com")
    
    result = service.register_user(command)
    
    assert result.is_success
    assert isinstance(result.data, User)
```

**Infrastructure Layer Testing**:

```python
def test_container_service_registration():
    container = FlextContainer()
    service = UserService()
    
    result = container.register("user_service", service)
    
    assert result.is_success
    retrieved = container.get("user_service")
    assert retrieved.is_success
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
