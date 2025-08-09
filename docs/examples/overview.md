# Practical Examples - FLEXT Core

Examples based on the real code in `src/flext_core`.

## 🎯 Overview

This section shows practical examples using REAL components from FLEXT Core. All examples were validated against the current code in `src/flext_core/__init__.py`.

## 📦 Available Imports

Based on `src/flext_core/__init__.py`:

```python
# Core patterns (FUNCTIONAL)
from flext_core import FlextResult, FlextContainer

# Domain patterns (Available)
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

# Configuration (FUNCTIONAL)
from flext_core import FlextSettings

# Other exports - check __init__.py for current status
```

## 🔄 Example 1: FlextResult Railway Pattern

Validated against the actual implementation:

```python
"""
Real example using FlextResult — the central pattern of FLEXT Core.
This example works with the current implementation.
"""

from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Validate email format."""
    if not email:
        return FlextResult.fail("Email is required")

    if "@" not in email:
        return FlextResult.fail("Email must contain @")

    return FlextResult.ok(email.lower())

def create_user_id(email: str) -> FlextResult[str]:
    """Create user ID from validated email."""
    user_id = f"user_{hash(email) % 10000:04d}"
    return FlextResult.ok(user_id)

def save_user_data(user_id: str, email: str) -> FlextResult[dict]:
    """Simulate saving user data."""
    user_data = {
        "id": user_id,
        "email": email,
        "created": True
    }
    return FlextResult.ok(user_data)

# Railway-oriented programming pattern
def create_user(email: str) -> FlextResult[dict]:
    """Complete user creation with railway pattern."""
    return (
        validate_email(email)
        .flat_map(lambda validated_email: create_user_id(validated_email)
                  .map(lambda user_id: (user_id, validated_email)))
        .flat_map(lambda data: save_user_data(data[0], data[1]))
    )

# Usage examples
if __name__ == "__main__":
    # Success case
    result = create_user("user@example.com")
    if result.success:
        print(f"✅ User created: {result.data}")
    else:
        print(f"❌ Error: {result.error}")

    # Error case
    error_result = create_user("invalid-email")
    print(f"❌ Expected error: {error_result.error}")
```

## 🏗️ Example 2: FlextContainer Dependency Injection

Validated against the actual implementation:

```python
"""
Real example using FlextContainer — FLEXT Core's DI system.
This example works with the current implementation.
"""

from flext_core import FlextContainer, FlextResult

# Simple services for DI example
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, data: dict) -> FlextResult[str]:
        """Simulate database save."""
        return FlextResult.ok(f"Saved {data} to {self.connection_string}")

class UserService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create user using injected database service."""
        user_data = {"name": name, "email": email}

        save_result = self.db_service.save(user_data)
        if save_result.is_failure:
            return FlextResult.fail(f"Save failed: {save_result.error}")

        return FlextResult.ok(user_data)

# Container setup and usage
def setup_container() -> FlextContainer:
    """Setup dependency injection container."""
    container = FlextContainer()

    # Register database service
    db_service = DatabaseService("sqlite:///app.db")
    reg_result = container.register("database", db_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register database: {reg_result.error}")

    # Register user service with dependency
    user_service = UserService(db_service)
    reg_result = container.register("user_service", user_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register user service: {reg_result.error}")

    return container

# Usage example
if __name__ == "__main__":
    # Setup container
    container = setup_container()

    # Get service from container
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.data

        # Use service
        create_result = user_service.create_user("John", "john@test.com")
        if create_result.success:
            print(f"✅ User created: {create_result.data}")
        else:
            print(f"❌ Create failed: {create_result.error}")
    else:
        print(f"❌ Service not found: {service_result.error}")
```

## 🏛️ Example 3: FlextEntity Domain Pattern

Validated — Domain entities using the current API:

```python
"""
Example using FlextEntity — FLEXT Core's domain pattern.
Correct — Using the current models.py API.
"""

from flext_core.models import FlextEntity
from flext_core import FlextResult
from typing import Optional
from datetime import datetime
from pydantic import Field

class User(FlextEntity):
    """Simple user entity example."""

    # Entity attributes (not in __init__)
    id: str
    name: str
    email: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None

    def validate_business_rules(self) -> FlextResult[None]:
        """Required abstract method implementation."""
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Valid email is required")
        if not self.name or len(self.name.strip()) == 0:
            return FlextResult.fail("Name cannot be empty")
        return FlextResult.ok(None)

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult.fail("User is already active")

        self.is_active = True
        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user account."""
        if not self.is_active:
            return FlextResult.fail("User is already inactive")

        self.is_active = False
        return FlextResult.ok(None)

    def login(self) -> FlextResult[None]:
        """Record user login."""
        if not self.is_active:
            return FlextResult.fail("Cannot login - user is inactive")

        self.last_login = datetime.now()
        return FlextResult.ok(None)

# Usage example
if __name__ == "__main__":
    # Create user entity with proper field-based initialization
    user = User(
        id="user_123",
        name="Maria Silva",
        email="maria@test.com"
    )
    print(f"✅ User created: {user.name} (ID: {user.id})")

    # Business operations
    login_result = user.login()
    if login_result.success:
        print(f"✅ Login successful at {user.last_login}")

    # Test business rules
    deactivate_result = user.deactivate()
    if deactivate_result.success:
        print("✅ User deactivated")

    # This should fail
    login_after_deactivate = user.login()
    print(f"❌ Expected failure: {login_after_deactivate.error}")
```

## ⚙️ Example 4: FlextSettings Configuration

Validated — Working configuration system:

```python
"""
Example using FlextSettings — FLEXT Core configuration system.
Based on the current implementation.
"""

from flext_core import FlextSettings
from typing import Optional

class AppSettings(FlextSettings):
    """Application configuration using FLEXT Core settings."""

    # Basic settings with defaults
    app_name: str = "FLEXT Demo App"
    debug: bool = False
    port: int = 8000

    # Database settings
    database_url: str = "sqlite:///app.db"
    max_connections: int = 10

    # Optional settings
    redis_url: Optional[str] = None

    class Config:
        env_prefix = "APP_"

# Usage example
if __name__ == "__main__":
    # Load configuration (from env vars or defaults)
    settings = AppSettings()

    print(f"✅ App: {settings.app_name}")
    print(f"✅ Debug: {settings.debug}")
    print(f"✅ Port: {settings.port}")
    print(f"✅ Database: {settings.database_url}")
    print(f"✅ Redis: {settings.redis_url or 'Not configured'}")

    # Environment-aware settings
    if settings.debug:
        print("🔧 Running in debug mode")
    else:
        print("🚀 Running in production mode")
```

## 🧪 How to Run the Examples

### 1. Check Dependencies

```bash
# Verify FLEXT Core is installed
python -c "from flext_core import FlextResult, FlextContainer; print('✅ Imports working')"
```

### 2. Run Examples

```bash
# Save any example as a .py file and run
python railway_example.py
python container_example.py
python entity_example.py
python config_example.py
```

### 3. Test Modifications

```bash
# Modify examples for your needs
# All examples use only the documented public API
```

## 🎯 Next Steps

1. **[Quickstart](../getting-started/quickstart.md)** — Get started with FLEXT Core
2. **[Core API](../api/core.md)** — Complete API reference
3. **[Architecture](../architecture/overview.md)** — Understand the patterns

## ⚠️ Important Note

These examples are based on the CURRENT implementation in `src/flext_core/`. For more elaborate examples, check the tests (`tests/`) and the project's `examples/` directory.

Component status (based on current code):

- ✅ **FlextResult**: Fully functional
- ✅ **FlextContainer**: Implemented and tested
- 🔧 **FlextEntity**: API available, functionality may be evolving
- 🔧 **FlextSettings**: Pydantic-based, functional
- 📋 **Advanced patterns**: Check current code for status

---

All examples here were validated against the code in `src/flext_core/__init__.py`.
