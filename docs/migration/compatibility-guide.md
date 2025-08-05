# Migration and Compatibility Guide

**Migration guide based on actual FLEXT Core implementation**

## 🔄 Migration Overview

This guide helps migrate existing Python applications to use FLEXT Core patterns. The migration focuses on proven patterns that exist in the current implementation.

## 📋 Migration Checklist

### Pre-Migration Assessment

- [ ] Identify current error handling patterns (exceptions vs results)
- [ ] List dependency injection usage
- [ ] Review configuration management
- [ ] Document current testing patterns

### Migration Steps

1. **Install FLEXT Core** and verify basic functionality
2. **Migrate error handling** to FlextResult pattern
3. **Replace dependency injection** with FlextContainer
4. **Update configuration** with FlextBaseSettings
5. **Adopt domain patterns** where applicable

## 🔧 Error Handling Migration

### From Exceptions to FlextResult

**Before (Exception-based):**

```python
# Old exception-based approach
class UserService:
    def get_user(self, user_id: str) -> User:
        if not user_id:
            raise ValueError("User ID is required")

        user = self.repository.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        return user

    def create_user(self, name: str, email: str) -> User:
        if not name or not email:
            raise ValueError("Name and email are required")

        if "@" not in email:
            raise ValueError("Invalid email format")

        user = User(name=name, email=email)
        self.repository.save(user)
        return user

# Usage with exception handling
try:
    user = user_service.get_user("123")
    print(f"Found user: {user.name}")
except ValueError as e:
    print(f"Error: {e}")
```

**After (FlextResult-based):**

```python
from flext_core import FlextResult

class UserService:
    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user with type-safe error handling."""
        if not user_id:
            return FlextResult.fail("User ID is required")

        user = self.repository.find_by_id(user_id)
        if not user:
            return FlextResult.fail(f"User {user_id} not found")

        return FlextResult.ok(user)

    def create_user(self, name: str, email: str) -> FlextResult[User]:
        """Create user with comprehensive error handling."""
        if not name or not email:
            return FlextResult.fail("Name and email are required")

        if "@" not in email:
            return FlextResult.fail("Invalid email format")

        user = User(name=name, email=email)
        save_result = self.repository.save(user)

        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save user: {save_result.error}")

        return FlextResult.ok(user)

# Usage with functional composition
result = user_service.get_user("123")
if result.success:
    print(f"Found user: {result.data.name}")
else:
    print(f"Error: {result.error}")

# Chain operations safely
creation_result = (
    user_service.create_user("John", "john@example.com")
    .map(lambda user: f"Created user: {user.name}")
)

if creation_result.success:
    print(creation_result.data)
```

### Migration Benefits

1. **Type Safety**: Errors are explicit in function signatures
2. **Composability**: Chain operations with map/flat_map
3. **No Silent Failures**: All errors must be handled explicitly
4. **Better Testing**: Easy to test both success and failure paths

## 🏗️ Dependency Injection Migration

### From Manual DI to FlextContainer

**Before (Manual dependency management):**

```python
# Manual dependency management
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

class UserService:
    def __init__(self, database_service: DatabaseService):
        self.database = database_service

# Manual wiring
db_service = DatabaseService("sqlite:///app.db")
user_service = UserService(db_service)
```

**After (FlextContainer):**

```python
from flext_core import FlextContainer

# Service classes remain the same
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

class UserService:
    def __init__(self, database_service: DatabaseService):
        self.database = database_service

# Container-based registration
def setup_services() -> FlextResult[FlextContainer]:
    """Setup all application services."""
    container = FlextContainer()

    # Register database service
    db_service = DatabaseService("sqlite:///app.db")
    db_result = container.register("database", db_service)
    if db_result.is_failure:
        return FlextResult.fail(f"Failed to register database: {db_result.error}")

    # Register user service
    user_service = UserService(db_service)
    user_result = container.register("user_service", user_service)
    if user_result.is_failure:
        return FlextResult.fail(f"Failed to register user service: {user_result.error}")

    return FlextResult.ok(container)

# Usage with error handling
def get_user_service() -> FlextResult[UserService]:
    container_result = setup_services()
    if container_result.is_failure:
        return FlextResult.fail(f"Container setup failed: {container_result.error}")

    container = container_result.data
    return container.get("user_service")

# Functional usage
service_result = get_user_service()
if service_result.success:
    user_service = service_result.data
    user_result = user_service.get_user("123")
    # Process user_result...
```

## ⚙️ Configuration Migration

### From Manual Environment Variables to FlextBaseSettings

**Before (Manual configuration):**

```python
import os

class Config:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///app.db")
        self.api_key = os.getenv("API_KEY")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.port = int(os.getenv("PORT", "8000"))

        # Manual validation
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")

# Global configuration
config = Config()
```

**After (FlextBaseSettings):**

```python
from flext_core import FlextBaseSettings
from typing import Optional

class AppSettings(FlextBaseSettings):
    """Type-safe application configuration."""

    # Database configuration
    database_url: str = "sqlite:///app.db"

    # API configuration
    api_key: str  # Required field

    # Server configuration
    port: int = 8000
    debug: bool = False

    # Optional settings
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"

# Usage with validation
def load_config() -> FlextResult[AppSettings]:
    """Load configuration with validation."""
    try:
        settings = AppSettings()
        return FlextResult.ok(settings)
    except Exception as e:
        return FlextResult.fail(f"Configuration error: {e}")

# Load and use configuration
config_result = load_config()
if config_result.success:
    settings = config_result.data
    print(f"Server will run on port {settings.port}")
    print(f"Debug mode: {settings.debug}")
else:
    print(f"Configuration failed: {config_result.error}")
```

## 🏛️ Domain Model Migration

### From Simple Data Classes to FlextEntity

**Before (Simple data classes):**

```python
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: str
    is_active: bool = True

# Business logic in service layer
class UserService:
    def activate_user(self, user: User) -> None:
        user.is_active = True
        self.repository.save(user)

    def change_email(self, user: User, new_email: str) -> None:
        if "@" not in new_email:
            raise ValueError("Invalid email format")
        user.email = new_email
        self.repository.save(user)
```

**After (FlextEntity with business logic):**

```python
from flext_core import FlextEntity, FlextResult

class User(FlextEntity):
    """User domain entity with business logic."""

    def __init__(self, user_id: str, name: str, email: str):
        super().__init__(user_id)
        self.name = name
        self.email = email
        self.is_active = True

    def activate(self) -> FlextResult[None]:
        """Activate the user."""
        if self.is_active:
            return FlextResult.fail("User is already active")

        self.is_active = True
        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate the user."""
        if not self.is_active:
            return FlextResult.fail("User is already inactive")

        self.is_active = False
        return FlextResult.ok(None)

    def change_email(self, new_email: str) -> FlextResult[None]:
        """Change user email with validation."""
        if "@" not in new_email:
            return FlextResult.fail("Invalid email format")

        if self.email == new_email:
            return FlextResult.fail("New email is the same as current email")

        self.email = new_email
        return FlextResult.ok(None)

    @classmethod
    def create(cls, name: str, email: str) -> FlextResult["User"]:
        """Factory method for creating users."""
        if not name or not email:
            return FlextResult.fail("Name and email are required")

        if "@" not in email:
            return FlextResult.fail("Invalid email format")

        user_id = f"user_{hash((name, email)) % 10000:04d}"
        user = cls(user_id, name, email)

        return FlextResult.ok(user)

# Usage with rich domain model
def create_and_activate_user(name: str, email: str) -> FlextResult[User]:
    """Create and activate a user with full error handling."""
    create_result = User.create(name, email)
    if create_result.is_failure:
        return create_result

    user = create_result.data
    activate_result = user.activate()
    if activate_result.is_failure:
        return FlextResult.fail(f"Activation failed: {activate_result.error}")

    return FlextResult.ok(user)

# Functional composition
result = create_and_activate_user("John Doe", "john@example.com")
if result.success:
    user = result.data
    print(f"Created and activated user: {user.name}")
else:
    print(f"Failed: {result.error}")
```

## 🧪 Testing During Migration

### Parallel Testing Strategy

```python
import pytest
from flext_core import FlextResult

class TestUserServiceMigration:
    """Test migration from exceptions to FlextResult."""

    def test_old_exception_behavior(self):
        """Test that old behavior still works for comparison."""
        # Keep old implementation for comparison during migration
        old_service = OldUserService()

        with pytest.raises(ValueError, match="User ID is required"):
            old_service.get_user("")

    def test_new_flext_result_behavior(self):
        """Test new FlextResult-based implementation."""
        new_service = NewUserService()

        result = new_service.get_user("")
        assert result.is_failure
        assert "User ID is required" in result.error

    def test_equivalent_success_behavior(self):
        """Ensure both implementations have equivalent success behavior."""
        old_service = OldUserService()
        new_service = NewUserService()

        # Mock successful case
        old_user = old_service.get_user("valid_id")
        new_result = new_service.get_user("valid_id")

        assert new_result.success
        assert new_result.data.id == old_user.id
        assert new_result.data.name == old_user.name

def test_container_migration():
    """Test container setup and service retrieval."""
    container = FlextContainer()

    # Register test service
    test_service = "test_value"
    reg_result = container.register("test", test_service)
    assert reg_result.success

    # Retrieve service
    get_result = container.get("test")
    assert get_result.success
    assert get_result.data == test_service

def test_configuration_migration():
    """Test configuration loading."""
    import os

    # Set test environment variable
    os.environ["TEST_API_KEY"] = "test_key_123"

    try:
        class TestSettings(FlextBaseSettings):
            api_key: str
            debug: bool = False

            class Config:
                env_prefix = "TEST_"

        settings = TestSettings()
        assert settings.api_key == "test_key_123"
        assert settings.debug is False

    finally:
        os.environ.pop("TEST_API_KEY", None)
```

## 🚀 Migration Best Practices

### 1. Gradual Migration Strategy

- **Start small**: Migrate one component at a time
- **Test thoroughly**: Maintain test coverage throughout migration
- **Keep both versions**: Run old and new implementations in parallel initially
- **Monitor carefully**: Watch for performance or behavior changes

### 2. Common Migration Patterns

**Error Handling Chain:**

```python
def complex_operation(user_id: str) -> FlextResult[ProcessedUser]:
    """Chain multiple operations safely."""
    return (
        self.get_user(user_id)
        .flat_map(self.validate_user)
        .flat_map(self.process_user)
    )
```

**Service Setup:**

```python
def setup_application() -> FlextResult[None]:
    """Setup entire application with error handling."""
    container_result = setup_services()
    if container_result.is_failure:
        return FlextResult.fail(f"Service setup failed: {container_result.error}")

    config_result = load_config()
    if config_result.is_failure:
        return FlextResult.fail(f"Config load failed: {config_result.error}")

    return FlextResult.ok(None)
```

## 🔍 Common Migration Issues

### Issue: Converting Exception Chains

**Problem**: Multiple exception types in try/catch blocks.

**Solution**: Use FlextResult chains with specific error messages:

```python
# Before: Multiple exception types
try:
    user = get_user(user_id)
    validate_user(user)
    process_user(user)
except UserNotFoundError:
    handle_not_found()
except ValidationError:
    handle_validation()
except ProcessingError:
    handle_processing()

# After: FlextResult chain
result = (
    get_user(user_id)
    .flat_map(validate_user)
    .flat_map(process_user)
)

if result.is_failure:
    # Handle all errors uniformly or parse error message
    handle_error(result.error)
```

### Issue: Configuration Validation

**Problem**: Runtime configuration errors with Pydantic.

**Solution**: Validate early with clear error messages:

```python
def load_configuration() -> FlextResult[AppSettings]:
    """Load configuration with detailed error handling."""
    try:
        settings = AppSettings()
        return FlextResult.ok(settings)
    except Exception as e:
        return FlextResult.fail(f"Configuration validation failed: {e}")

# Use at application startup
config_result = load_configuration()
if config_result.is_failure:
    print(f"Cannot start application: {config_result.error}")
    exit(1)

config = config_result.data
```

---

**This migration guide is based on the actual FLEXT Core implementation in src/flext_core/**
