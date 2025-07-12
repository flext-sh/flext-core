# FLEXT-CORE PATTERNS AND STANDARDS

**Authority**: VERIFIED - All patterns tested and operational  
**Coverage**: 100% test coverage achieved  
**Last Updated**: 2025-07-11  
**Target Audience**: FLEXT Framework Developers

---

## 🏗️ ARCHITECTURAL FOUNDATIONS

### Clean Architecture Implementation

flext-core implements **Domain-Driven Design (DDD)** with clean architecture principles:

```
src/flext_core/
├── domain/           # Business logic, entities, value objects
├── application/      # Use cases, command/query handlers
├── infrastructure/   # External concerns, repositories
└── config/          # Configuration and dependency injection
```

**Key Principle**: Dependencies point inward. Domain has no external dependencies.

### Dependency Rules

1. **Domain Layer**: Pure business logic, no external dependencies
2. **Application Layer**: Orchestrates domain objects, defines use cases
3. **Infrastructure Layer**: Implements interfaces defined in domain
4. **Configuration Layer**: Handles cross-cutting concerns

---

## 🔧 CORE APIs AND PATTERNS

### ServiceResult Pattern (MANDATORY)

**Use ServiceResult for ALL service operations**:

```python
from flext_core import ServiceResult

# Success case
result = ServiceResult.ok(data)
result = ServiceResult.success(data)  # Alias

# Failure case
result = ServiceResult.fail("Error message")
result = ServiceResult.failure("Error message")  # Alias

# Pending operations
result = ServiceResult.pending()

# Usage patterns
if result.is_successful:
    data = result.data
else:
    error = result.error

# Functional patterns
value = result.unwrap()  # Raises if failed
value = result.unwrap_or("default")  # Safe unwrap

# Chaining operations
result2 = result.map(transform_func)
result3 = result.and_then(chain_func)
```

**Properties Available**:

- `is_successful` / `is_success` - Success check
- `data` / `value` - Result data
- `error` / `error_message` - Error details
- `status` - ResultStatus enum

### Configuration System (REQUIRED)

**BaseConfig for domain configs**:

```python
from flext_core import BaseConfig

class MyDomainConfig(BaseConfig):
    service_name: str = "my-service"
    timeout: int = 30
    enabled: bool = True
```

**BaseSettings for application settings**:

```python
from flext_core import BaseSettings

class MyAppSettings(BaseSettings):
    # Inherits project_name, project_version, environment, debug
    api_host: str = "localhost"
    api_port: int = 8000

    # Environment integration
    @classmethod
    def from_env(cls, env_file: str | None = None):
        return super().from_env(env_file)
```

### Domain Modeling Standards

**Use provided base classes**:

```python
from flext_core import DomainEntity, DomainValueObject, DomainAggregateRoot

class User(DomainEntity):
    """User entity with automatic ID and timestamps."""
    name: str
    email: str
    # id, created_at, updated_at provided automatically

class Email(DomainValueObject):
    """Value object for email addresses."""
    address: str

    def model_post_init(self, __context):
        if "@" not in self.address:
            raise ValueError("Invalid email")

class Organization(DomainAggregateRoot):
    """Aggregate root for complex domain operations."""
    name: str
    users: list[User] = []
```

### Repository Pattern

**Use typed repositories**:

```python
from flext_core import InMemoryRepository
from flext_core.domain.core import Repository

class UserRepository(Repository[User, UUID]):
    """Repository interface for User entities."""

    async def find_by_email(self, email: str) -> User | None:
        """Find user by email address."""
        ...

class InMemoryUserRepository(InMemoryRepository[User, UUID]):
    """In-memory implementation for testing."""

    async def find_by_email(self, email: str) -> User | None:
        for user in self._entities.values():
            if user.email == email:
                return user
        return None
```

---

## 📝 TYPING STANDARDS

### Modern Python 3.13 Types

**Use new type syntax**:

```python
# Type aliases
type UserId = UUID
type UserEmail = str

# Generic classes
class Repository[T, K]:
    def get(self, key: K) -> T | None: ...

# Generic functions
def process[T](data: T) -> ServiceResult[T]: ...
```

**Import from flext_core**:

```python
from flext_core import (
    # ID types
    EntityId, UserId, PipelineId, PluginId,
    # String types
    ProjectName, Version,
    # Enums
    EntityStatus, ResultStatus, Environment,
    # Protocols
    EntityProtocol, ConfigProtocol, SettingsProtocol
)
```

### Pydantic Integration

**Use flext_core Pydantic bases**:

```python
from flext_core import (
    DomainBaseModel,     # For domain models
    APIBaseModel,        # For API models
    APIRequest,          # For API requests
    APIResponse,         # For API responses
    Field               # Re-exported from Pydantic
)

class CreateUserRequest(APIRequest):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')

class UserResponse(APIResponse):
    user: User
    created_at: datetime
```

---

## 🧪 TESTING STANDARDS

### Test Organization

**Follow the established structure**:

```
tests/
├── unit/            # Unit tests for individual components
├── integration/     # Integration tests between layers
├── application/     # Application service tests
├── domain/         # Domain logic tests
├── infrastructure/ # Infrastructure tests
└── conftest.py     # Shared test configuration
```

### Test Patterns

**Use flext_core test utilities**:

```python
import pytest
from flext_core import ServiceResult, DomainEntity

class TestUser(DomainEntity):
    name: str
    email: str

def test_service_result_success():
    """Test successful service result."""
    result = ServiceResult.ok("test-data")

    assert result.is_successful
    assert result.data == "test-data"
    assert result.error is None

def test_service_result_failure():
    """Test failed service result."""
    result = ServiceResult.fail("test error")

    assert not result.is_successful
    assert result.data is None
    assert result.error == "test error"

def test_entity_creation():
    """Test domain entity creation."""
    user = TestUser(name="John", email="john@test.com")

    assert user.name == "John"
    assert user.email == "john@test.com"
    assert user.id is not None
    assert user.created_at is not None
```

### Coverage Requirements

- **Minimum**: 95% line coverage
- **Target**: 100% line coverage (achieved in flext-core)
- **Branch coverage**: Required for critical paths
- **Tests must be deterministic and fast**

---

## 🔌 DEPENDENCY INJECTION

### Container Usage

**Register and resolve services**:

```python
from flext_core import DIContainer, injectable, singleton

# Manual registration
container = DIContainer()
container.register(UserRepository, InMemoryUserRepository())

# Decorator-based registration
@injectable(UserRepository)
class DatabaseUserRepository(UserRepository):
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

@singleton
class EmailService:
    def __init__(self, smtp_config: SMTPConfig):
        self.smtp_config = smtp_config

# Usage in application
container = get_container()
user_repo = container.resolve(UserRepository)
email_service = container.resolve(EmailService)  # Singleton
```

### Settings Integration

**Configure dependencies with settings**:

```python
from flext_core import BaseSettings, get_settings

class MySettings(BaseSettings):
    database_url: str = "sqlite:///app.db"
    redis_url: str = "redis://localhost:6379"

def configure_app():
    settings = get_settings(MySettings)
    container = get_container()

    # Register settings-based services
    settings.configure_dependencies(container)
```

---

## 🚦 ERROR HANDLING

### Exception Hierarchy

**Use flext_core exceptions**:

```python
from flext_core.domain.core import (
    DomainError,        # Base domain exception
    ValidationError,    # Validation failures
    NotFoundError,      # Entity not found
    RepositoryError     # Repository failures
)

class UserNotFoundError(NotFoundError):
    """Raised when user is not found."""
    def __init__(self, user_id: UUID):
        super().__init__(f"User {user_id} not found")

class InvalidEmailError(ValidationError):
    """Raised when email format is invalid."""
    def __init__(self, email: str):
        super().__init__(f"Invalid email format: {email}")
```

### ServiceResult Error Handling

**Convert exceptions to ServiceResult**:

```python
def create_user(name: str, email: str) -> ServiceResult[User]:
    """Create a new user."""
    try:
        if not email or "@" not in email:
            return ServiceResult.fail("Invalid email format")

        user = User(name=name, email=email)
        # ... save user logic

        return ServiceResult.ok(user)

    except ValidationError as e:
        return ServiceResult.fail(f"Validation error: {e}")
    except Exception as e:
        return ServiceResult.fail(f"Unexpected error: {e}")
```

---

## 📊 LOGGING AND OBSERVABILITY

### Structured Logging

**Use consistent logging patterns**:

```python
import structlog

logger = structlog.get_logger(__name__)

def process_user(user_id: UUID) -> ServiceResult[User]:
    """Process user with structured logging."""
    logger.info("Processing user", user_id=str(user_id))

    try:
        # ... processing logic
        logger.info("User processed successfully", user_id=str(user_id))
        return ServiceResult.ok(user)

    except Exception as e:
        logger.error("User processing failed",
                    user_id=str(user_id),
                    error=str(e))
        return ServiceResult.fail(str(e))
```

### Health Checks

**Implement health checks for all services**:

```python
from flext_core import ComponentHealth, HealthStatus

class UserService:
    def health_check(self) -> ComponentHealth:
        """Check service health."""
        try:
            # Perform health check logic
            return ComponentHealth(
                name="user-service",
                status=HealthStatus.HEALTHY,
                details={"database": "connected"}
            )
        except Exception as e:
            return ComponentHealth(
                name="user-service",
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)}
            )
```

---

## 📦 PROJECT STRUCTURE STANDARDS

### Required Files

Every FLEXT project MUST have:

```
project-root/
├── pyproject.toml          # Python 3.13, dependencies from flext-core
├── src/                    # Source code
│   └── your_project/
│       ├── __init__.py     # Export public API
│       ├── domain/         # Domain logic
│       ├── application/    # Use cases
│       ├── infrastructure/ # External integrations
│       └── config/         # Configuration
├── tests/                  # Test suite
├── README.md              # Project documentation
└── CLAUDE.md              # Project-specific patterns
```

### Import Standards

**Import from flext-core first**:

```python
# Standard library
import os
from typing import Any
from uuid import UUID

# Third-party
import pytest
from pydantic import Field

# FLEXT framework - import core first
from flext_core import (
    ServiceResult,
    BaseSettings,
    DomainEntity,
    PipelineService
)

# Local imports last
from your_project.domain import User
from your_project.config import Settings
```

### Version Management

**Use semantic versioning**:

```toml
[project]
name = "flext-your-project"
version = "0.7.0"  # Match flext-core version
dependencies = [
    "flext-core>=0.7.0,<0.8.0"  # Pin to compatible version
]
```

---

## 🔄 CONTINUOUS INTEGRATION

### Quality Gates

**All projects must pass**:

```bash
# Type checking
mypy src tests

# Linting
ruff check src tests

# Testing with coverage
pytest --cov=src --cov-fail-under=95

# Security
bandit -r src
```

### Pre-commit Hooks

**Required quality checks**:

```yaml
# .pre-commit-config.yaml
repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.8.0
      hooks:
          - id: ruff
          - id: ruff-format
    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.13.0
      hooks:
          - id: mypy
```

---

## 📋 CHECKLIST FOR NEW PROJECTS

### Foundation Setup

- [ ] Create project structure following flext-core patterns
- [ ] Add flext-core as primary dependency
- [ ] Implement BaseSettings for configuration
- [ ] Set up clean architecture layers
- [ ] Add comprehensive test suite

### API Standards

- [ ] Use ServiceResult for all service operations
- [ ] Implement proper error handling with flext-core exceptions
- [ ] Use typed repository patterns
- [ ] Add health check endpoints
- [ ] Document all public APIs

### Quality Standards

- [ ] Achieve 95%+ test coverage
- [ ] Pass all type checking (mypy)
- [ ] Pass all linting (ruff)
- [ ] Add pre-commit hooks
- [ ] Document project-specific patterns in CLAUDE.md

### Integration Standards

- [ ] Register with FLEXT dependency injection container
- [ ] Implement structured logging
- [ ] Add observability hooks
- [ ] Test integration with other FLEXT modules
- [ ] Verify no circular dependencies

---

## 🎯 MIGRATION FROM LEGACY CODE

### Step-by-Step Migration

1. **Add flext-core dependency**
2. **Replace custom result types with ServiceResult**
3. **Migrate configuration to BaseSettings/BaseConfig**
4. **Update exception handling to use flext-core exceptions**
5. **Implement clean architecture structure**
6. **Add comprehensive tests**
7. **Update imports to use flext-core exports**

### Common Patterns

```python
# BEFORE (legacy)
class LegacyResult:
    def __init__(self, success, data=None, error=None):
        self.success = success
        self.data = data
        self.error = error

# AFTER (flext-core)
from flext_core import ServiceResult

# result = LegacyResult(True, data, None)
result = ServiceResult.ok(data)
```

---

## ⚡ PERFORMANCE CONSIDERATIONS

### Best Practices

1. **Use async/await for I/O operations**
2. **Implement connection pooling for databases**
3. **Cache frequently accessed data**
4. **Use batch operations for bulk data**
5. **Monitor performance with structured logging**

### Resource Management

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_transaction():
    """Proper resource management."""
    transaction = await db.begin()
    try:
        yield transaction
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise
    finally:
        await transaction.close()
```

---

**AUTHORITY**: All patterns in this document are VERIFIED and operational in flext-core with 100% test coverage. Other FLEXT projects MUST follow these patterns for consistency and interoperability.

**ENFORCEMENT**: Any deviation from these patterns must be documented with justification in the project's CLAUDE.md file.
