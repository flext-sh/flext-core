# Quick Start — FLEXT-Core 0.12.0-dev

<!-- TOC START -->
- [Installation](#installation)
- [Core Concepts (5 min)](#core-concepts-5-min)
  - [1. Railway Results (`r[T]`)](#1-railway-results-rt)
  - [2. Dependency Injection](#2-dependency-injection)
  - [3. CQRS Dispatcher](#3-cqrs-dispatcher)
  - [4. Services & Models](#4-services--models)
- [Practical Examples](#practical-examples)
- [Common Patterns](#common-patterns)
- [Next Steps](#next-steps)
<!-- TOC END -->

**Updated**: 2026-04-14 | **Python**: 3.13+ | **Status**: Current

## Installation

```bash
poetry add flext-core
# or
pip install flext-core
```

## Core Concepts (5 min)

### 1. Railway Results (`r[T]`)

Everything that can fail returns `r[T]` — a result carrier that eliminates exceptions in business logic.

```python
from flext_core import r

# Success path
success_result = r[int].ok(42)
assert success_result.success == True
assert success_result.unwrap() == 42

# Failure path
fail_result = r[int].fail("Something went wrong")
assert fail_result.success == False
assert fail_result.error == "Something went wrong"

# Chain operations
result = (
    r[int]
    .ok(10)
    .map(lambda x: x * 2)  # Transform: 10 → 20
    .flat_map(lambda x: r[int].ok(x + 5))  # Chain: 20 → 25
    .unwrap_or(0)  # Extract or default
)
# result = 25
```

**Key Methods**:
- `ok(value)`, `fail(error)` — Constructor
- `success`, `error` — Inspect result
- `map`, `flat_map`, `lash` — Transform/chain
- `unwrap()`, `unwrap_or(default)`, `unwrap_type()` — Extract value

### 2. Dependency Injection

Register services once, resolve them anywhere with `FlextContainer`.

```python
from flext_core import FlextContainer, r

# Get global singleton
container = FlextContainer.get_global()

# Register factory
container.factory("db", lambda: MyDatabase("localhost"))

# Resolve with error handling
db_result = container.resolve("db")
if db_result.success:
    db = db_result.unwrap()
    # Use db...
else:
    print(f"Error: {db_result.error}")

# Scoped resolution (short-lived instances)
with container.scope() as scoped:
    service = scoped.resolve("service").unwrap()
    # service exists only in this context
```

### 3. CQRS Dispatcher

Route typed messages (commands/queries) to handlers automatically.

```python
from pydantic import BaseModel
from flext_core import FlextDispatcher, r, m


# 1. Define command
class CreateUserCommand(m.Entity):
    username: str
    email: str


# 2. Define queries
class GetUserQuery(m.Entity):
    user_id: int


# 3. Handle them
def create_user_handler(cmd: CreateUserCommand) -> r[str]:
    return r[str].ok(f"Created: {cmd.username}")


def get_user_handler(q: GetUserQuery) -> r[dict]:
    return r[dict].ok({"id": q.user_id, "name": "Alice"})


# 4. Register & dispatch
dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUserCommand, create_user_handler)
dispatcher.register_handler(GetUserQuery, get_user_handler)

# Dispatch commands
result = dispatcher.dispatch(
    CreateUserCommand(username="alice", email="alice@example.com")
)
print(result.unwrap())  # "Created: alice"

# Dispatch queries
result = dispatcher.dispatch(GetUserQuery(user_id=1))
print(result.unwrap())  # {"id": 1, "name": "Alice"}
```

### 4. Services & Models

Create domain services with DI and entity models with validation.

```python
from pydantic import Field
from flext_core import s, r, m, c


# Define domain entity
class User(m.Entity):
    name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")


# Create service with DI
class UserService(s):
    """User operations with dependency injection."""

    def create_user(self, name: str, email: str) -> r[User]:
        # Resolve database from container
        db = self.container.resolve("database")
        if not db.success:
            return r[User].fail(db.error)

        # Validate & create
        try:
            user = User(name=name, email=email)
            db.unwrap().insert(user.model_dump())
            return r[User].ok(user)
        except Exception as e:
            return r[User].fail_exc(e, error_code="USER_CREATION_FAILED")


# Use service
service = UserService()
result = service.create_user("Alice", "alice@example.com")

if result.success:
    user = result.unwrap()
    print(f"Created user: {user.name}")
else:
    print(f"Error: {result.error}")
```

## Practical Examples

### Example 1: Data Validation Pipeline

```python
from flext_core import r, u


def validate_email(email: str) -> r[str]:
    if "@" not in email:
        return r[str].fail("Invalid email format")
    return r[str].ok(email)


def validate_age(age: int) -> r[int]:
    if age < 18:
        return r[int].fail("Must be 18 or older")
    return r[int].ok(age)


# Chain validations
def validate_user_input(email: str, age: int) -> r[dict]:
    return validate_email(email).flat_map(
        lambda e: validate_age(age).map(lambda a: {"email": e, "age": a})
    )


result = validate_user_input("alice@example.com", 25)
print(result.unwrap())  # {"email": "alice@example.com", "age": 25}

result = validate_user_input("invalid", 15)
print(result.error)  # "Must be 18 or older"
```

### Example 2: Error Recovery

```python
from flext_core import r


def fetch_user(user_id: int) -> r[dict]:
    # Simulated failure
    return r[dict].fail("User not found")


# Recover with default
user = fetch_user(999).unwrap_or({"id": 0, "name": "Anonymous"})
# user = {"id": 0, "name": "Anonymous"}


# Recover with recovery function
def create_default_user(user_id: int) -> r[dict]:
    return r[dict].ok({"id": user_id, "name": "Generated User"})


result = (
    fetch_user(999).lash(lambda _: create_default_user(999))  # Fallback on failure
)
print(result.unwrap())  # {"id": 999, "name": "Generated User"}
```

### Example 3: Structured Logging

```python
from flext_core import u

# Get logger
logger = u.fetch_logger("my_app")

logger.info("User created", extra={"user_id": 123, "email": "alice@example.com"})
logger.warning("High latency detected", extra={"ms": 5000})
logger.error("Database connection failed", extra={"host": "localhost", "port": 5432})
```

### Example 4: Settings & Configuration

```python
from pydantic import Field
from flext_core import FlextSettings


class AppSettings(FlextSettings):
    """Application configuration."""

    database_url: str = "sqlite://app.db"
    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_prefix = "FLEXT_APP_"  # Read: FLEXT_APP_DATABASE_URL, etc.


# Load settings (from env or defaults)
settings = FlextSettings.get_global()
print(settings.database_url)  # From env or default
print(settings.debug)  # From env or False
```

## Common Patterns

### Pattern: Batch Operations with Accumulation

```python
from flext_core import r


def process_items(items: list[str]) -> r[list[dict]]:
    results = []
    for item in items:
        # For each item, validate and process
        result = validate_and_process(item)  # Returns r[dict]
        if result.success:
            results.append(result.unwrap())
        else:
            # Fail fast on first error
            return r[list[dict]].fail(result.error)

    return r[list[dict]].ok(results)


# Or accumulate errors
def process_items_lenient(items: list[str]) -> r[dict]:
    successes = []
    failures = []

    for item in items:
        result = validate_and_process(item)
        if result.success:
            successes.append(result.unwrap())
        else:
            failures.append({"item": item, "error": result.error})

    return r[dict].ok({
        "successful": successes,
        "failed": failures,
        "total": len(items),
    })
```

### Pattern: Conditional Logic

```python
from flext_core import r


def process_order(order_id: int, is_premium: bool) -> r[str]:
    # Get order
    order = fetch_order(order_id)
    if not order.success:
        return r[str].fail(order.error)

    # Apply logic based on type
    if is_premium:
        result = apply_premium_discount(order.unwrap())
    else:
        result = apply_standard_discount(order.unwrap())

    return result.map(lambda o: f"Order {o.id} processed")
```

## Next Steps

1. **Explore Modules**:
   - `m.*` — Domain models & entities
   - `c.*` — Global constants & error codes
   - `p.*` — Protocol contracts
   - `u.*` — Utility functions (guards, parsers, etc.)

2. **Read Full Guides**: See [`docs/architecture/`](architecture/) for deeper dives

3. **Review Examples**: Check [`examples/`](../examples/) directory

4. **Run Tests**: `make test` to see usage patterns in test suite

---

**Questions?** File an issue or check [CONTRIBUTING.md](../CONTRIBUTING.md).
