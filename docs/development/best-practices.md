# Boas Práticas - FLEXT Core

**Práticas recomendadas baseadas na implementação atual**

## 🎯 Princípios Fundamentais

### 1. Type Safety First

**SEMPRE use type hints - obrigatório no FLEXT Core.**

```python
# ✅ Correto - Type hints baseados na API atual
from flext_core import FlextResult

def process_user_data(user_id: str, data: dict[str, object]) -> FlextResult[dict]:
    """Process user data with type safety."""
    if not user_id:
        return FlextResult.fail("User ID é obrigatório")

    processed_data = {"user_id": user_id, "data": data}
    return FlextResult.ok(processed_data)

# ❌ Evite - Sem type hints
def process_user_data(user_id, data):
    pass
```

### 2. FlextResult Pattern

**Use FlextResult para error handling - padrão central do FLEXT Core.**

```python
from flext_core import FlextResult

# ✅ Correto - Railway-oriented programming
def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Divide numbers with explicit error handling."""
    if b == 0:
        return FlextResult.fail("Division by zero not allowed")

    return FlextResult.ok(a / b)

# Usage pattern
result = divide_numbers(10, 2)
if result.success:
    print(f"Result: {result.data}")
else:
    print(f"Error: {result.error}")

# ❌ Evite - Exceções para fluxo de controle business
def divide_numbers_bad(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")  # Não use para business logic
    return a / b
```

### 3. Dependency Injection

**Use FlextContainer para gerenciar dependências.**

```python
from flext_core import FlextContainer, FlextResult

# ✅ Correto - Setup proper do container
def setup_dependencies() -> FlextContainer:
    """Configure application dependencies."""
    container = FlextContainer()

    # Register services
    database_service = DatabaseService("sqlite:///app.db")
    result = container.register("database", database_service)

    if result.is_failure:
        raise RuntimeError(f"Failed to setup dependencies: {result.error}")

    return container

# ❌ Evite - Hard-coded dependencies
class UserService:
    def __init__(self):
        self.database = DatabaseService()  # Hard-coded
```

## 🔧 Padrões de Desenvolvimento

### 1. Validation Strategy

**Valide dados em múltiplas camadas.**

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Input validation layer."""
    if not email:
        return FlextResult.fail("Email é obrigatório")

    if "@" not in email:
        return FlextResult.fail("Email deve conter @")

    if len(email) > 254:
        return FlextResult.fail("Email muito longo")

    return FlextResult.ok(email.lower())

def validate_business_rules(email: str, domain: str) -> FlextResult[None]:
    """Business validation layer."""
    if not email.endswith(f"@{domain}"):
        return FlextResult.fail(f"Email deve ser do domínio {domain}")

    return FlextResult.ok(None)
```

### 2. Error Handling Chain

**Chain operations com FlextResult.**

```python
from flext_core import FlextResult

def create_user_account(email: str, name: str, company_domain: str) -> FlextResult[dict]:
    """Complete user creation with error handling chain."""

    # Validate input
    email_result = validate_email(email)
    if email_result.is_failure:
        return FlextResult.fail(f"Email validation: {email_result.error}")

    validated_email = email_result.data

    # Validate business rules
    business_result = validate_business_rules(validated_email, company_domain)
    if business_result.is_failure:
        return FlextResult.fail(f"Business rule: {business_result.error}")

    # Create account
    account_data = {
        "email": validated_email,
        "name": name,
        "domain": company_domain,
        "created": True
    }

    return FlextResult.ok(account_data)
```

### 3. Service Pattern

**Organize services com dependency injection.**

```python
from flext_core import FlextContainer, FlextResult

class UserService:
    """User service with injected dependencies."""

    def __init__(self, database_service):
        self.database = database_service

    def create_user(self, email: str, name: str) -> FlextResult[dict]:
        """Create user with proper error handling."""
        # Validate
        if not email or not name:
            return FlextResult.fail("Email and name are required")

        # Create user data
        user_data = {"email": email, "name": name}

        # Save using injected dependency
        save_result = self.database.save_user(user_data)
        if save_result.is_failure:
            return FlextResult.fail(f"Save failed: {save_result.error}")

        return FlextResult.ok(user_data)

# Setup with container
def setup_user_service(container: FlextContainer) -> FlextResult[UserService]:
    """Setup user service with dependencies."""
    db_result = container.get("database")
    if db_result.is_failure:
        return FlextResult.fail("Database service not found")

    user_service = UserService(db_result.data)

    reg_result = container.register("user_service", user_service)
    if reg_result.is_failure:
        return FlextResult.fail(f"Failed to register user service: {reg_result.error}")

    return FlextResult.ok(user_service)
```

## 🧪 Testing Patterns

### 1. FlextResult Testing

**Test both success and failure paths.**

```python
import pytest
from flext_core import FlextResult

def test_divide_numbers_success():
    """Test successful division."""
    result = divide_numbers(10, 2)

    assert result.success
    assert result.data == 5.0
    assert result.error is None

def test_divide_numbers_failure():
    """Test division by zero."""
    result = divide_numbers(10, 0)

    assert result.is_failure
    assert result.data is None
    assert "zero" in result.error.lower()

def test_divide_numbers_error_type():
    """Test error handling with FlextResult."""
    result = divide_numbers(10, 0)

    # FlextResult pattern
    if result.success:
        value = result.data
    else:
        error_message = result.error
        assert isinstance(error_message, str)
```

### 2. Container Testing

**Test dependency injection setup.**

```python
import pytest
from flext_core import FlextContainer

def test_container_registration():
    """Test service registration."""
    container = FlextContainer()
    service = DatabaseService("test.db")

    result = container.register("database", service)
    assert result.success

    get_result = container.get("database")
    assert get_result.success
    assert get_result.data is service

def test_container_missing_service():
    """Test missing service error."""
    container = FlextContainer()

    result = container.get("nonexistent")
    assert result.is_failure
    assert "not found" in result.error.lower()
```

## 🚀 Performance Practices

### 1. Efficient Error Handling

**Use FlextResult without exception overhead.**

```python
from flext_core import FlextResult

# ✅ Efficient - No exception overhead
def process_batch_data(items: list[dict]) -> FlextResult[list[dict]]:
    """Process items efficiently with FlextResult."""
    processed_items = []

    for item in items:
        # Validate each item
        if not item.get("id"):
            return FlextResult.fail(f"Item missing ID: {item}")

        # Process item
        processed_item = {"id": item["id"], "processed": True}
        processed_items.append(processed_item)

    return FlextResult.ok(processed_items)

# ❌ Avoid - Exception overhead in loops
def process_batch_data_bad(items: list[dict]) -> list[dict]:
    """Inefficient exception handling."""
    processed_items = []
    for item in items:
        if not item.get("id"):
            raise ValueError(f"Item missing ID")  # Exception overhead
        processed_items.append(item)
    return processed_items
```

### 2. Container Efficiency

**Register services once, reuse container.**

```python
from flext_core import FlextContainer

# ✅ Efficient - Single container setup
_application_container: FlextContainer | None = None

def get_container() -> FlextContainer:
    """Get singleton container instance."""
    global _application_container

    if _application_container is None:
        _application_container = FlextContainer()
        setup_services(_application_container)

    return _application_container

def setup_services(container: FlextContainer) -> None:
    """Setup all services once."""
    database = DatabaseService("app.db")
    container.register("database", database)

    user_service = UserService(database)
    container.register("user_service", user_service)

# ❌ Avoid - Creating container repeatedly
def get_user_service_bad():
    """Inefficient - new container every time."""
    container = FlextContainer()  # Created every call
    setup_services(container)
    return container.get("user_service").data
```

## 📚 Code Organization

### 1. Module Structure

**Organize code following Clean Architecture.**

```python
# ✅ Good structure following FLEXT patterns
# services/user_service.py
from flext_core import FlextResult

class UserService:
    """Business logic for users."""

    def create_user(self, data: dict) -> FlextResult[dict]:
        # Business logic here
        pass

# infrastructure/database.py
class DatabaseService:
    """Infrastructure concerns."""

    def save_user(self, user_data: dict) -> FlextResult[str]:
        # Database operations
        pass

# main.py - composition root
from flext_core import FlextContainer

def main():
    """Application entry point."""
    container = FlextContainer()

    # Setup dependencies
    database = DatabaseService("app.db")
    container.register("database", database)

    user_service = UserService(database)
    container.register("user_service", user_service)

    # Run application
    app_result = run_application(container)
    if app_result.is_failure:
        print(f"Application failed: {app_result.error}")
        exit(1)
```

## ⚠️ Common Pitfalls

### ❌ Avoid These Patterns

1. **Mixed error handling approaches:**

```python
# Don't mix FlextResult with exceptions in business logic
def bad_example(data: str) -> FlextResult[str]:
    if not data:
        raise ValueError("Bad data")  # ❌ Mixed approaches
    return FlextResult.ok(data.upper())
```

2. **Direct dependency creation:**

```python
# Don't create dependencies directly
class UserService:
    def __init__(self):
        self.database = DatabaseService()  # ❌ Hard-coded
```

3. **Ignoring FlextResult errors:**

```python
# Don't ignore FlextResult failure paths
result = some_operation()
data = result.data  # ❌ What if result.is_failure?
```

## ✅ Best Practices Summary

### DO

- Use FlextResult for all business operations
- Apply type hints throughout your code
- Use FlextContainer for dependency injection
- Test both success and failure paths
- Follow Clean Architecture separation
- Handle errors explicitly

### DON'T

- Mix exceptions with FlextResult in business logic
- Create hard-coded dependencies
- Ignore FlextResult failure paths
- Use exceptions for control flow
- Skip type annotations
- Create containers repeatedly

---

**Estas práticas são baseadas na implementação atual do FLEXT Core e foram validadas contra o código em src/flext_core/.**
