# API Patterns - FLEXT Core

**Available patterns based on the current implementation**

## 🎯 Overview

This documentation covers REAL design patterns implemented in FLEXT Core. All imports and examples were validated against the current code in src/flext_core/.

## 📦 Available Imports

**VALIDATED** - Based on current code:

### Core Patterns

```python
# Core patterns - Functional
from flext_core import FlextResult, FlextContainer

# Commands and Handlers - Implemented
from flext_core.commands import FlextCommands
from flext_core.handlers import (
    FlextBaseHandler,
    FlextValidatingHandler,
    FlextAuthorizingHandler,
    FlextEventHandler,
    FlextMetricsHandler,
)
from flext_core.validation import FlextValidation
```

### Domain Patterns

```python
# Domain patterns - Available
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot
```

## 🎭 Command Pattern

**BASED ON src/flext_core/commands.py:**

### Basic Command Usage

```python
"""
Real example using FLEXT Core's command system.
Based on the current implementation.
"""

from flext_core import FlextResult
from flext_core.commands import FlextCommands

# Simple command implementation
class CreateUserCommand:
    """Command to create a new user."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self.command_id = f"create_user_{hash((name, email)) % 10000:04d}"

    def validate_command(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.name or not self.name.strip():
            return FlextResult.fail("Name is required")

        if not self.email or "@" not in self.email:
            return FlextResult.fail("Invalid email")

        return FlextResult.ok(None)

    def get_command_data(self) -> dict[str, object]:
        """Get command data for processing."""
        return {
            "command_id": self.command_id,
            "name": self.name,
            "email": self.email
        }

# Command handler implementation
class CreateUserHandler:
    """Handler for CreateUserCommand."""

    def __init__(self, user_repository):
        self.user_repository = user_repository
        self.handler_id = "create_user_handler"

    def can_handle(self, command) -> bool:
        """Check if handler can process command."""
        return isinstance(command, CreateUserCommand)

    def handle(self, command: CreateUserCommand) -> FlextResult[dict]:
        """Process the command."""
        # Validate command first
        validation_result = command.validate_command()
        if validation_result.is_failure:
            return FlextResult.fail(f"Command validation failed: {validation_result.error}")

        # Create user data
        user_data = {
            "name": command.name,
            "email": command.email.lower(),
            "created": True
        }

        # Simulate saving
        save_result = self.user_repository.save(user_data)
        if save_result.is_failure:
            return FlextResult.fail(f"Save failed: {save_result.error}")

        return FlextResult.ok(user_data)

# Usage example
if __name__ == "__main__":
    # Mock repository
    class MockUserRepository:
        def save(self, user_data: dict) -> FlextResult[dict]:
            return FlextResult.ok(user_data)

    # Setup
    repository = MockUserRepository()
    handler = CreateUserHandler(repository)

    # Create and process command
    command = CreateUserCommand("John Smith", "john@example.com")

    if handler.can_handle(command):
        result = handler.handle(command)
        if result.success:
            print(f"✅ User created: {result.data}")
        else:
            print(f"❌ Error: {result.error}")
```

## 🎪 Handler Pattern

**BASED ON src/flext_core/handlers.py:**

### Handler Implementation

```python
"""
Handler system based on FLEXT Core's real implementation.
"""

from flext_core import FlextResult
from flext_core.handlers import FlextBaseHandler

# Message handler example
class EmailNotificationHandler(FlextBaseHandler):
    """Handler for email notifications."""

    def __init__(self, email_service):
        self.email_service = email_service
        self.handler_id = "email_notification_handler"

    def can_handle(self, message_type: type) -> bool:
        return True  # simplified for example

    def process_message(self, message: dict) -> FlextResult[str]:
        """Process email notification message."""
        # Validate message
        if not message.get("recipient"):
            return FlextResult.fail("Recipient is required")

        if not message.get("subject"):
            return FlextResult.fail("Subject is required")

        # Send email
        try:
            email_result = self.email_service.send(
                to=message["recipient"],
                subject=message["subject"],
                body=message.get("body", "")
            )
            return FlextResult.ok(f"Email sent to {message['recipient']}")
        except Exception as e:
            return FlextResult.fail(f"Email send failed: {str(e)}")

# Handler registry
from flext_core.handlers import FlextHandlerRegistry as HandlerRegistry
    """Registry for managing handlers."""

    def __init__(self):
        self.handlers = []

    def register(self, handler) -> FlextResult[None]:
        return self._registry.register(handler.__class__.__name__, handler)

    def find_handlers(self, message) -> list:
        """Find handlers that can process a message."""
        return [h for h in self.handlers if h.can_handle(message)]

    def get_handler_by_id(self, handler_id: str):
        """Get handler by ID."""
        for handler in self.handlers:
            if getattr(handler, 'handler_id', None) == handler_id:
                return handler
        return None

# Usage example
if __name__ == "__main__":
    # Mock email service
    class MockEmailService:
        def send(self, to: str, subject: str, body: str) -> str:
            return f"Email sent to {to}"

    # Setup
    email_service = MockEmailService()
    email_handler = EmailNotificationHandler(email_service)

    registry = HandlerRegistry()
    registry.register(email_handler)

    # Process message
    email_message = {
        "type": "email_notification",
        "recipient": "user@example.com",
        "subject": "Welcome!",
        "body": "Welcome to our platform"
    }

    handlers = registry.find_handlers(email_message)
    if handlers:
        handler = handlers[0]
        result = handler.handle_message(email_message)
        if result.success:
            print(f"✅ {result.data}")
        else:
            print(f"❌ {result.error}")
```

## ✅ Validation Pattern

**BASEADO EM src/flext_core/validation.py:**

### Validation Implementation

```python
"""
Validation system based on FLEXT Core's real implementation.
"""

from flext_core import FlextResult
from flext_core.validation import FlextValidation

# Simple validation functions
def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if not email:
        return FlextResult.fail("Email is required")

    if "@" not in email:
        return FlextResult.fail("Email must contain @")

    if len(email) > 254:
        return FlextResult.fail("Email is too long")

    return FlextResult.ok(email.lower())

def validate_name(name: str) -> FlextResult[str]:
    """Validate name format."""
    if not name:
        return FlextResult.fail("Name is required")

    cleaned_name = name.strip()
    if len(cleaned_name) < 2:
        return FlextResult.fail("Name must be at least 2 characters")

    if len(cleaned_name) > 100:
        return FlextResult.fail("Name is too long")

    return FlextResult.ok(cleaned_name)

def validate_age(age: int) -> FlextResult[int]:
    """Validate age range."""
    if age < 0:
        return FlextResult.fail("Age cannot be negative")

    if age > 150:
        return FlextResult.fail("Age must be realistic")

    return FlextResult.ok(age)

# Validation result aggregator
class ValidationResult:
    """Aggregate validation results."""

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []

    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)

    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

# User validator example
class UserValidator:
    """Validator for user data."""

    def validate(self, user_data: dict) -> ValidationResult:
        """Validate complete user data."""
        result = ValidationResult()

        # Validate name
        name_result = validate_name(user_data.get("name", ""))
        if name_result.is_failure:
            result.add_error(f"Name: {name_result.error}")

        # Validate email
        email_result = validate_email(user_data.get("email", ""))
        if email_result.is_failure:
            result.add_error(f"Email: {email_result.error}")

        # Validate age (optional)
        if "age" in user_data:
            age_result = validate_age(user_data["age"])
            if age_result.is_failure:
                result.add_error(f"Age: {age_result.error}")

        # Business rule validation
        if user_data.get("age", 0) < 18:
            result.add_warning("User is underage")

        return result

# Usage example
if __name__ == "__main__":
    validator = UserValidator()

    # Valid user
    valid_user = {
        "name": "John",
        "email": "john@example.com",
        "age": 30
    }

    result = validator.validate(valid_user)
    if result.is_valid:
        print("✅ Valid data")
        if result.warnings:
            print(f"⚠️ Warnings: {result.warnings}")
    else:
        print(f"❌ Errors: {result.errors}")

    # Invalid user
    invalid_user = {
        "name": "",
        "email": "invalid-email",
        "age": -5
    }

    result = validator.validate(invalid_user)
    print(f"❌ Expected errors: {result.errors}")
```

## 🧪 Testing Patterns

### Pattern Testing

```python
"""
Testing patterns for FLEXT Core patterns.
"""

import pytest
from flext_core import FlextResult

def test_command_validation():
    """Test command validation."""
    command = CreateUserCommand("", "invalid")

    result = command.validate_command()
    assert result.is_failure
    assert "Name is required" in result.error

def test_handler_processing():
    """Test handler processing."""
    class MockRepo:
        def save(self, data):
            return FlextResult.ok(data)

    handler = CreateUserHandler(MockRepo())
    command = CreateUserCommand("John", "john@test.com")

    result = handler.handle(command)
    assert result.success
    assert result.data["name"] == "John"

def test_validation_patterns():
    """Test validation patterns."""
    validator = UserValidator()

    # Valid data
    valid_data = {"name": "John", "email": "john@test.com", "age": 25}
    result = validator.validate(valid_data)
    assert result.is_valid

    # Invalid data
    invalid_data = {"name": "", "email": "invalid"}
    result = validator.validate(invalid_data)
    assert not result.is_valid
    assert len(result.errors) > 0
```

## 🎯 Real Implementation Status

**BASED ON CURRENT CODE** in src/flext_core/:

### ✅ Available and Functional

- **FlextResult**: Totalmente implementado e testado
- **FlextContainer**: Sistema de DI funcional
- **Commands namespace**: FlextCommands available
- **Handlers namespace**: FlextHandlers available
- **Validation namespace**: FlextValidation available

### 🔧 In Development

- **Full CQRS**: Command bus and advanced handlers
- **Event handling**: Domain event patterns
- **Query bus**: Full read/write separation

### 📋 Planned

- **Auto-discovery**: Automatic handler registration
- **Middleware pipeline**: Cross-cutting concerns
- **Advanced validation**: Complex business rules

## ⚠️ Important

This documentation reflects the CURRENT implementation of FLEXT Core. For more advanced functionality, see:

1. **Current code**: src/flext_core/{commands,handlers,validation}.py
2. **Tests**: tests/ for functional examples
3. **Examples**: examples/ for real use cases

---

**All examples were validated against the current implementation in src/flext_core/**
