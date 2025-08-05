# API Patterns - FLEXT Core

**Padrões disponíveis baseados na implementação atual**

## 🎯 Visão Geral

Esta documentação cobre os padrões de design REAIS implementados no FLEXT Core. Todas as importações e exemplos foram validados contra o código atual em src/flext_core/.

## 📦 Importações Disponíveis

**VALIDADO** - Baseado no código atual:

### Core Patterns

```python
# Core patterns - FUNCTIONAL
from flext_core import FlextResult, FlextContainer

# Commands e Handlers - IMPLEMENTADOS
from flext_core import commands, handlers, validation

# Acesso via classes namespace
from flext_core.commands import FlextCommands
from flext_core.handlers import FlextHandlers
from flext_core.validation import FlextValidation
```

### Domain Patterns

```python
# Domain patterns - DISPONÍVEIS
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot
```

## 🎭 Command Pattern

**BASEADO EM src/flext_core/commands.py:**

### Basic Command Usage

```python
"""
Exemplo real usando o sistema de commands do FLEXT Core.
Baseado na implementação atual.
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
            return FlextResult.fail("Nome é obrigatório")

        if not self.email or "@" not in self.email:
            return FlextResult.fail("Email inválido")

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
    command = CreateUserCommand("João Silva", "joao@example.com")

    if handler.can_handle(command):
        result = handler.handle(command)
        if result.success:
            print(f"✅ User created: {result.data}")
        else:
            print(f"❌ Error: {result.error}")
```

## 🎪 Handler Pattern

**BASEADO EM src/flext_core/handlers.py:**

### Handler Implementation

```python
"""
Sistema de handlers baseado na implementação real do FLEXT Core.
"""

from flext_core import FlextResult
from flext_core.handlers import FlextHandlers

# Message handler example
class EmailNotificationHandler:
    """Handler for email notifications."""

    def __init__(self, email_service):
        self.email_service = email_service
        self.handler_id = "email_notification_handler"

    def can_handle(self, message) -> bool:
        """Check if this handler can process the message."""
        return isinstance(message, dict) and message.get("type") == "email_notification"

    def handle_message(self, message: dict) -> FlextResult[str]:
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
class HandlerRegistry:
    """Registry for managing handlers."""

    def __init__(self):
        self.handlers = []

    def register(self, handler) -> FlextResult[None]:
        """Register a handler."""
        if not hasattr(handler, 'can_handle'):
            return FlextResult.fail("Handler must have can_handle method")

        if not hasattr(handler, 'handler_id'):
            return FlextResult.fail("Handler must have handler_id")

        self.handlers.append(handler)
        return FlextResult.ok(None)

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
Sistema de validação baseado na implementação real do FLEXT Core.
"""

from flext_core import FlextResult
from flext_core.validation import FlextValidation

# Simple validation functions
def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if not email:
        return FlextResult.fail("Email é obrigatório")

    if "@" not in email:
        return FlextResult.fail("Email deve conter @")

    if len(email) > 254:
        return FlextResult.fail("Email muito longo")

    return FlextResult.ok(email.lower())

def validate_name(name: str) -> FlextResult[str]:
    """Validate name format."""
    if not name:
        return FlextResult.fail("Nome é obrigatório")

    cleaned_name = name.strip()
    if len(cleaned_name) < 2:
        return FlextResult.fail("Nome deve ter pelo menos 2 caracteres")

    if len(cleaned_name) > 100:
        return FlextResult.fail("Nome muito longo")

    return FlextResult.ok(cleaned_name)

def validate_age(age: int) -> FlextResult[int]:
    """Validate age range."""
    if age < 0:
        return FlextResult.fail("Idade não pode ser negativa")

    if age > 150:
        return FlextResult.fail("Idade deve ser realista")

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
            result.add_warning("Usuário menor de idade")

        return result

# Usage example
if __name__ == "__main__":
    validator = UserValidator()

    # Valid user
    valid_user = {
        "name": "João Silva",
        "email": "joao@example.com",
        "age": 30
    }

    result = validator.validate(valid_user)
    if result.is_valid:
        print("✅ Dados válidos")
        if result.warnings:
            print(f"⚠️ Warnings: {result.warnings}")
    else:
        print(f"❌ Erros: {result.errors}")

    # Invalid user
    invalid_user = {
        "name": "",
        "email": "invalid-email",
        "age": -5
    }

    result = validator.validate(invalid_user)
    print(f"❌ Erros esperados: {result.errors}")
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
    assert "Nome é obrigatório" in result.error

def test_handler_processing():
    """Test handler processing."""
    class MockRepo:
        def save(self, data):
            return FlextResult.ok(data)

    handler = CreateUserHandler(MockRepo())
    command = CreateUserCommand("João", "joao@test.com")

    result = handler.handle(command)
    assert result.success
    assert result.data["name"] == "João"

def test_validation_patterns():
    """Test validation patterns."""
    validator = UserValidator()

    # Valid data
    valid_data = {"name": "João", "email": "joao@test.com", "age": 25}
    result = validator.validate(valid_data)
    assert result.is_valid

    # Invalid data
    invalid_data = {"name": "", "email": "invalid"}
    result = validator.validate(invalid_data)
    assert not result.is_valid
    assert len(result.errors) > 0
```

## 🎯 Real Implementation Status

**BASEADO NO CÓDIGO ATUAL** em src/flext_core/:

### ✅ Disponível e Funcional

- **FlextResult**: Totalmente implementado e testado
- **FlextContainer**: Sistema de DI funcional
- **Commands namespace**: FlextCommands disponível
- **Handlers namespace**: FlextHandlers disponível
- **Validation namespace**: FlextValidation disponível

### 🔧 Em Desenvolvimento

- **CQRS completo**: Command bus e handlers avançados
- **Event handling**: Padrões de eventos de domínio
- **Query bus**: Separação completa de read/write

### 📋 Planejado

- **Auto-discovery**: Registro automático de handlers
- **Middleware pipeline**: Cross-cutting concerns
- **Advanced validation**: Regras de negócio complexas

## ⚠️ Importante

Esta documentação reflete a implementação ATUAL do FLEXT Core. Para funcionalidades mais avançadas, consulte:

1. **Código atual**: src/flext_core/{commands,handlers,validation}.py
2. **Testes**: tests/ para exemplos funcionais
3. **Examples**: examples/ para casos de uso reais

---

**Todos os exemplos foram validados contra a implementação atual em src/flext_core/**
