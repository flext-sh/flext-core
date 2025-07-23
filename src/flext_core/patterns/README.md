# FLEXT Core Patterns

Enterprise patterns for building robust, maintainable applications.

## Overview

This module provides implementation of common enterprise patterns including Command/Query Responsibility Segregation (CQRS), message handling, and validation patterns.

## Components

### Commands (`commands.py`)
- `FlextCommand` - Base class for command objects
- `FlextCommandHandler` - Command processing interface
- Command validation and execution patterns

### Handlers (`handlers.py`)
- `FlextHandler` - Base handler with metadata support
- `FlextMessageHandler` - Generic message processing
- `FlextEventHandler` - Domain event processing  
- `FlextRequestHandler` - Request/response handling
- `FlextHandlerRegistry` - Central handler management

### Validation (`validation.py`)
- `FlextValidator` - Data validation interface
- `FlextBusinessRule` - Business rule specifications
- `FlextValidationPipeline` - Orchestrated validation

### Logging (`logging.py`)
- Structured logging patterns
- Context-aware log handling
- Performance monitoring integration

### Type Definitions (`typedefs.py`)
- Type aliases for pattern components
- NewType definitions for type safety
- Generic type variables

### Fields (`fields.py`)
- Custom Pydantic field types
- Validation helpers
- Type-safe field definitions

## Usage Examples

### Command Pattern

```python
from flext_core.patterns import FlextCommand, FlextCommandHandler

class CreateUserCommand(FlextCommand):
    name: str
    email: str

class CreateUserHandler(FlextCommandHandler[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Implementation
        return FlextResult.ok(user)
```

### Event Handling

```python
from flext_core.patterns import FlextEventHandler

class UserCreatedHandler(FlextEventHandler[UserCreatedEvent]):
    def handle_event(self, event: UserCreatedEvent) -> FlextResult[None]:
        # Handle the event
        return FlextResult.ok(None)
    
    def get_event_type(self) -> str:
        return "user_created"
```

### Validation

```python
from flext_core.patterns import FlextValidator, FlextBusinessRule

class EmailValidator(FlextValidator[str]):
    def validate(self, email: str) -> FlextResult[str]:
        if "@" not in email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(email)

class AgeRule(FlextBusinessRule[User]):
    def is_satisfied_by(self, user: User) -> bool:
        return user.age >= 18
```

## Design Principles

- **Type Safety**: All patterns use Python 3.13 generics for type safety
- **Error Handling**: Consistent FlextResult usage throughout
- **Composability**: Patterns can be combined and extended
- **Testability**: Clear interfaces enable easy unit testing
- **Performance**: Minimal overhead with efficient implementations

## Integration

These patterns integrate seamlessly with other FLEXT Core components:

- **FlextResult**: All operations return FlextResult for error handling
- **FlextContainer**: Handlers can be registered as services
- **Domain Layer**: Events and commands work with domain entities
- **Configuration**: Validation integrates with settings management

For detailed usage examples, see the [patterns documentation](../../docs/api/patterns.md).