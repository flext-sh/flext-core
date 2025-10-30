# Architecture Patterns

Common architectural patterns used in FLEXT-Core and best practices for applying them.

## Railway-Oriented Programming

**Pattern:** Use `FlextResult[T]` for composable error handling.

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult[str].fail("Invalid email")
    return FlextResult[str].ok(email)

def check_available(email: str) -> FlextResult[str]:
    if email in reserved_emails:
        return FlextResult[str].fail("Email taken")
    return FlextResult[str].ok(email)

# Railway composition
result = (
    validate_email("user@example.com")
    .flat_map(check_available)
    .map(lambda e: f"Ready: {e}")
)
```

**Benefits:**

- ✅ Composable error handling
- ✅ No exception throwing
- ✅ Type-safe error propagation
- ✅ Predictable error flows

## Dependency Injection

**Pattern:** Use `FlextContainer` for service registration and resolution.

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Register services
logger = FlextLogger(__name__)
container.register("logger", logger, singleton=True)

# Resolve services
logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Message")
```

**Benefits:**

- ✅ Loose coupling between components
- ✅ Easy testing with mocked dependencies
- ✅ Singleton management
- ✅ Centralized service lifecycle

## Domain-Driven Design

**Pattern:** Model business logic with Entities, Value Objects, and Services.

```python
from flext_core import FlextModels, FlextService, FlextResult

# Value Object - immutable, compared by value
class Money(FlextModels.Value):
    amount: float
    currency: str

# Entity - has identity
class Order(FlextModels.Entity):
    customer_id: str
    items: list
    total: Money

# Service - encapsulates business logic
class OrderService(FlextService):
    def place_order(self, customer_id: str, items: list) -> FlextResult[Order]:
        # Business logic here
        pass
```

**Benefits:**

- ✅ Clear business logic organization
- ✅ Reusable domain concepts
- ✅ Business rule validation
- ✅ Domain event support

## Command Query Responsibility Segregation (CQRS)

**Pattern:** Separate read (queries) and write (commands) operations.

```python
from flext_core import FlextBus, FlextResult

# Command - modifies state
class CreateUserCommand:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

# Query - reads state
class GetUserQuery:
    def __init__(self, user_id: str):
        self.user_id = user_id

# Handler for command
@bus.command_handler
class CreateUserHandler:
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Create and return user
        pass

# Handler for query
@bus.query_handler
class GetUserHandler:
    def handle(self, query: GetUserQuery) -> FlextResult[User]:
        # Retrieve and return user (no modification)
        pass
```

**Benefits:**

- ✅ Clear intent (read vs write)
- ✅ Optimized query models
- ✅ Scalable read/write separation
- ✅ Easier testing

## Event-Driven Architecture

**Pattern:** Use domain events for decoupled communication.

```python
from flext_core import FlextModels, FlextService

class UserCreatedEvent:
    def __init__(self, user_id: str, email: str):
        self.user_id = user_id
        self.email = email

class UserService(FlextService):
    def create_user(self, name: str, email: str) -> FlextResult[User]:
        user = User(id="new", name=name, email=email)

        # Emit domain event
        self.add_domain_event(UserCreatedEvent(user.id, email))

        return FlextResult[User].ok(user)

# Subscribers listen to events
class EmailNotificationSubscriber:
    def on_user_created(self, event: UserCreatedEvent):
        # Send welcome email
        send_email(event.email, "Welcome!")
```

**Benefits:**

- ✅ Decoupled components
- ✅ Async processing support
- ✅ Audit trail of important events
- ✅ Integration with external systems

## Hexagonal Architecture (Ports & Adapters)

**Pattern:** Isolate domain from external systems with ports and adapters.

```python
# Port - defines interface
class UserRepository:
    """Port: abstraction for user persistence."""
    def save(self, user: User) -> FlextResult[User]:
        raise NotImplementedError

    def get_by_id(self, user_id: str) -> FlextResult[User]:
        raise NotImplementedError

# Adapter - concrete implementation
class PostgresUserRepository(UserRepository):
    """Adapter: PostgreSQL implementation."""
    def save(self, user: User) -> FlextResult[User]:
        # PostgreSQL-specific implementation
        pass

    def get_by_id(self, user_id: str) -> FlextResult[User]:
        # PostgreSQL query
        pass

# Domain doesn't know about implementation
class UserService(FlextService):
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def create_user(self, user: User) -> FlextResult[User]:
        return self.repository.save(user)
```

**Benefits:**

- ✅ Domain independent of infrastructure
- ✅ Easy to swap implementations
- ✅ Testable with mock repositories
- ✅ Clear boundaries

## Layer Caching Pattern

**Pattern:** Implement multi-tier caching strategy.

```python
from flext_core import FlextResult
import functools

class UserService:
    def __init__(self, repository: UserRepository, cache):
        self.repository = repository
        self.cache = cache

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user with caching."""
        # Check cache first
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            return FlextResult[User].ok(cached)

        # Query repository
        result = self.repository.get_by_id(user_id)

        # Cache on success
        if result.is_success:
            user = result.unwrap()
            self.cache.set(f"user:{user_id}", user, ttl=3600)

        return result
```

**Benefits:**

- ✅ Improved performance
- ✅ Reduced database queries
- ✅ Consistent data access
- ✅ Configurable TTL

## Validation Pipeline Pattern

**Pattern:** Chain validations using FlextResult.

```python
from flext_core import FlextResult

def validate_password(password: str) -> FlextResult[str]:
    if len(password) < 8:
        return FlextResult[str].fail("Too short")
    if not any(c.isupper() for c in password):
        return FlextResult[str].fail("Need uppercase")
    if not any(c.isdigit() for c in password):
        return FlextResult[str].fail("Need digit")
    return FlextResult[str].ok(password)

def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult[str].fail("Invalid email")
    return FlextResult[str].ok(email)

def validate_username(username: str) -> FlextResult[str]:
    if len(username) < 3:
        return FlextResult[str].fail("Too short")
    if not username.isalnum():
        return FlextResult[str].fail("Only alphanumeric")
    return FlextResult[str].ok(username)

# Pipeline
def register_user(username: str, email: str, password: str) -> FlextResult[dict]:
    return (
        validate_username(username)
        .flat_map(lambda u: validate_email(email).map(lambda e: (u, e)))
        .flat_map(lambda ue: validate_password(password).map(lambda p: (*ue, p)))
        .map(lambda data: {"username": data[0], "email": data[1], "password": data[2]})
    )

# Test
result = register_user("alice", "alice@example.com", "SecurePass123")
if result.is_success:
    print(f"✅ {result.unwrap()}")
else:
    print(f"❌ {result.error}")
```

**Benefits:**

- ✅ Clear validation flow
- ✅ Fails fast on first error
- ✅ Composable validators
- ✅ Easy to test individual validators

## Middleware Pipeline Pattern

**Pattern:** Chain middleware for cross-cutting concerns.

```python
from flext_core import FlextBus

class LoggingMiddleware:
    def process(self, message, next_handler):
        logger.info(f"Processing {type(message).__name__}")
        result = next_handler(message)
        logger.info(f"Completed {type(message).__name__}")
        return result

class ValidationMiddleware:
    def process(self, message, next_handler):
        if not self.is_valid(message):
            return FlextResult.fail("Invalid message")
        return next_handler(message)

class AuthenticationMiddleware:
    def process(self, message, next_handler):
        if not self.is_authenticated():
            return FlextResult.fail("Not authenticated")
        return next_handler(message)

# Configure bus with middleware
bus = FlextBus()
bus.add_middleware(LoggingMiddleware())
bus.add_middleware(ValidationMiddleware())
bus.add_middleware(AuthenticationMiddleware())
```

**Benefits:**

- ✅ Separation of concerns
- ✅ Reusable middleware
- ✅ Clear execution order
- ✅ Testable in isolation

## Service Locator Pattern (Anti-Pattern Warning)

**⚠️ Use sparingly:** Dependency Injection preferred.

```python
# ❌ AVOID - Service Locator
class UserService:
    def get_logger(self):
        return FlextContainer.get_global().get("logger").unwrap()

# ✅ PREFER - Dependency Injection
class UserService:
    def __init__(self, logger):
        self.logger = logger
```

**When acceptable:**

- Bootstrap code (application startup)
- Legacy code migration
- When true DI would be too complex

## Adapter Pattern

**Pattern:** Convert interface to expected interface.

```python
# External library with incompatible interface
class ExternalLogger:
    def log_message(self, msg: str, level: str):
        print(f"[{level}] {msg}")

# Adapter to make it compatible
class LoggerAdapter(FlextLogger):
    def __init__(self, external_logger):
        self.external = external_logger

    def info(self, message: str, extra: dict = None):
        self.external.log_message(message, "INFO")

    def error(self, message: str, extra: dict = None):
        self.external.log_message(message, "ERROR")

# Use
external = ExternalLogger()
logger = LoggerAdapter(external)
```

**Benefits:**

- ✅ Integrate incompatible libraries
- ✅ Hide external API differences
- ✅ Maintain consistent interface

## Factory Pattern

**Pattern:** Create objects without specifying exact classes.

```python
from abc import ABC, abstractmethod

class UserFactory:
    @staticmethod
    def create_user(user_type: str, **kwargs) -> FlextResult[User]:
        if user_type == "REDACTED_LDAP_BIND_PASSWORD":
            return FlextResult[User].ok(AdminUser(**kwargs))
        elif user_type == "regular":
            return FlextResult[User].ok(RegularUser(**kwargs))
        else:
            return FlextResult[User].fail(f"Unknown user type: {user_type}")

# Usage
result = UserFactory.create_user("REDACTED_LDAP_BIND_PASSWORD", name="Alice")
if result.is_success:
    REDACTED_LDAP_BIND_PASSWORD = result.unwrap()
```

**Benefits:**

- ✅ Encapsulates object creation
- ✅ Easy to add new types
- ✅ Centralized creation logic

## Summary

Key patterns in FLEXT-Core:

- ✅ **Railway-Oriented**: Error composition with FlextResult
- ✅ **Dependency Injection**: Service management with FlextContainer
- ✅ **Domain-Driven Design**: Business logic organization
- ✅ **CQRS**: Separated read/write paths
- ✅ **Event-Driven**: Decoupled communication
- ✅ **Hexagonal Architecture**: Domain isolation
- ✅ **Pipeline**: Composable validation/middleware
- ✅ **Factory/Adapter**: Object creation and adaptation

Use these patterns to build maintainable, scalable FLEXT-Core applications.
