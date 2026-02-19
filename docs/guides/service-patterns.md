# Service Patterns Guide


<!-- TOC START -->
- [Canonical Rules](#canonical-rules)
- [Overview](#overview)
- [Execution Patterns](#execution-patterns)
  - [V1: Explicit Execution (✅ Production Ready)](#v1-explicit-execution-production-ready)
  - [V2 Property: `.result` (🟡 Under Validation)](#v2-property-result-under-validation)
  - [V2 Auto: `auto_execute` (🟡 Under Validation)](#v2-auto-autoexecute-under-validation)
- [Infrastructure Properties](#infrastructure-properties)
  - [Example with Infrastructure](#example-with-infrastructure)
- [Composition Patterns](#composition-patterns)
  - [Railway Composition](#railway-composition)
  - [Service Factories](#service-factories)
- [Integration with Handlers](#integration-with-handlers)
- [Testing Services](#testing-services)
  - [Unit Testing](#unit-testing)
  - [Integration Testing](#integration-testing)
- [Migration Guide](#migration-guide)
  - [From V1 to V2 Property](#from-v1-to-v2-property)
  - [Gradual Adoption](#gradual-adoption)
- [Best Practices](#best-practices)
  - [DO ✅](#do-)
  - [DON'T ❌](#dont-)
- [Next Steps](#next-steps)
- [See Also](#see-also)
- [References](#references)
<!-- TOC END -->

**Status**: Production Ready | **Version**: 0.10.0 | **Pattern**: Services

**Version:** 1.0 (2025-12-03)  
**Python:** 3.13+  
**Pydantic:** 2.x  
**Status:** V1 stable; V2 patterns under validation

This guide describes FlextService usage patterns and the evolution from
explicit execution (V1) to zero-ceremony patterns (V2).

## Canonical Rules

- Follow root governance in `CLAUDE.md`.
- Keep service examples returning `FlextResult[T]` and matching layer boundaries.
- Keep runtime/DI guidance aligned with `dependency-injection-advanced.md`.

---

## Overview

`FlextService[T]` is the foundation for domain services in FLEXT-Core. A service
is essentially a **Pydantic model with an `execute()` method** that returns
`FlextResult[T]`.

```python
from flext_core import FlextService, FlextResult

class CreateUserService(FlextService[User]):
    name: str
    email: str

    def execute(self) -> FlextResult[User]:
        user = User(name=self.name, email=self.email)
        return FlextResult[User].ok(user)
```

---

## Execution Patterns

### V1: Explicit Execution (✅ Production Ready)

The baseline pattern uses explicit method calls:

```python
# Instantiate service with parameters
service = CreateUserService(name="Alice", email="alice@example.com")

# Execute and handle result
result = service.execute()
if result.is_success:
    user = result.value
    print(f"Created user: {user.name}")
else:
    print(f"Error: {result.error}")
```

**Characteristics:**

- ✅ Railway pattern explicit – full control over errors
- ✅ Type-safe with `FlextResult[T]`
- ✅ 100% backward compatible
- ⚠️ Verbose (`.execute().value` on every use)

**When to use:**

- Existing codebases (32+ projects using this pattern)
- When explicit error handling is critical
- Railway composition with `.flat_map()`

### V2 Property: `.result` (🟡 Under Validation)

> **Status:** Code implemented in `service.py:122-140`, tests exist in
> `tests/unit/test_service_v2_patterns.py`. Run full test suite for validation.

The property pattern provides a shorthand:

```python
# V2 Property: Access result directly
try:
    user = CreateUserService(name="Alice", email="alice@example.com").result
    print(f"Created user: {user.name}")
except FlextExceptions.BaseError as e:
    print(f"Error: {e}")
```

**Characteristics:**

- ✅ 68% reduction in code (7 chars vs 19)
- ✅ Lazy evaluation (executes on first access)
- ⚠️ Error handling via exceptions

### V2 Auto: `auto_execute` (🟡 Under Validation)

> **Status:** Code implemented in `service.py:58-113`, tests exist in
> `tests/test_service_auto_execute.py`. Run full test suite for validation.

The auto-execution pattern returns the value directly on instantiation:

```python
class AutoUserService(FlextService[User]):
    auto_execute: ClassVar[bool] = True  # Opt-in
    name: str
    email: str

    def execute(self) -> FlextResult[User]:
        return FlextResult[User].ok(User(name=self.name, email=self.email))

# Instantiation returns User directly (not service instance)
user = AutoUserService(name="Alice", email="alice@example.com")
print(f"Created user: {user.name}")
```

**Characteristics:**

- ✅ 95% reduction in code (4 chars vs 19)
- ✅ Zero ceremony – just instantiate
- ⚠️ Opt-in via `auto_execute = True`
- ⚠️ Error handling via exceptions

---

## Infrastructure Properties

FlextService inherits from `FlextMixins`, providing automatic access to
infrastructure:

| Property         | Type             | Description                    |
| ---------------- | ---------------- | ------------------------------ |
| `self.config`    | `FlextSettings`  | Configuration singleton        |
| `self.logger`    | `FlextLogger`    | Logger with context            |
| `self.container` | `FlextContainer` | Dependency injection container |
| `self.context`   | `FlextContext`   | Execution context (task-local) |

All properties are **lazy-loaded** – no overhead if unused.

### Example with Infrastructure

```python
class ProcessOrderService(FlextService[Order]):
    order_id: str

    def execute(self) -> FlextResult[Order]:
        # Logging (via FlextMixins)
        self.logger.info(f"Processing order {self.order_id}")

        # Configuration (via FlextMixins)
        max_retries = self.config.max_retry_attempts

        # Dependency resolution (via FlextMixins)
        repo_result = self.container.get("order_repository")
        if repo_result.is_failure:
            return FlextResult[Order].fail("Repository unavailable")

        repo = repo_result.value
        return repo.find_by_id(self.order_id)
```

---

## Composition Patterns

### Railway Composition

Chain services using `flat_map`:

```python
def process_user(name: str, email: str) -> FlextResult[User]:
    return (
        ValidateEmailService(email=email).execute()
        .flat_map(lambda _: ValidateNameService(name=name).execute())
        .flat_map(lambda _: CreateUserService(name=name, email=email).execute())
    )
```

**Note**: Use `.flat_map()` for chaining operations. This is the standard FLEXT pattern and works seamlessly with all FlextResult operations.

### Service Factories

Create services dynamically:

```python
def create_notification_service(
    channel: str,
    message: str,
) -> FlextService[bool]:
    match channel:
        case "email":
            return EmailNotificationService(message=message)
        case "sms":
            return SmsNotificationService(message=message)
        case _:
            return NoOpNotificationService(message=message)

# Usage
service = create_notification_service("email", "Hello!")
result = service.execute()
```

---

## Integration with Handlers

Services are called by CQRS handlers for domain operations:

```python
from flext_core.handlers import FlextHandlers
from flext_core.result import r

class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
    def handle(self, command: CreateUserCommand) -> r[User]:
        # Handler orchestrates, service executes
        return CreateUserService(
            name=command.name,
            email=command.email,
        ).execute()
```

See [CQRS Architecture](../architecture/cqrs.md) for handler details.

---

## Testing Services

### Unit Testing

```python
def test_create_user_service_success():
    service = CreateUserService(name="Alice", email="alice@example.com")
    result = service.execute()

    assert result.is_success
    user = result.value
    assert user.name == "Alice"
    assert user.email == "alice@example.com"

def test_create_user_service_invalid_email():
    service = CreateUserService(name="Alice", email="invalid")
    result = service.execute()

    assert result.is_failure
    assert "email" in result.error.lower()
```

### Integration Testing

```python
def test_service_with_container(container: FlextContainer):
    container.register("email_validator", MockEmailValidator())

    service = CreateUserService(name="Alice", email="alice@example.com")
    result = service.execute()

    assert result.is_success
```

---

## Migration Guide

### From V1 to V2 Property

```python
# Before (V1)
result = MyService(param="value").execute()
if result.is_success:
    value = result.value
else:
    handle_error(result.error)

# After (V2 Property)
try:
    value = MyService(param="value").result
except FlextExceptions.BaseError as e:
    handle_error(str(e))
```

### Gradual Adoption

1. **New code:** Consider V2 patterns if tests pass
2. **Existing code:** Keep V1 (no changes required)
3. **Critical paths:** Prefer V1 for explicit error handling

---

## Best Practices

### DO ✅

- Keep services focused (single responsibility)
- Return `FlextResult` from `execute()` (railway pattern)
- Use lazy infrastructure properties (`self.config`, `self.logger`)
- Validate inputs early with `FlextResult.fail()`

### DON'T ❌

- Raise exceptions in `execute()` (use `FlextResult.fail()`)
- Access infrastructure in `__init__` (properties are lazy)
- Mix V1 and V2 patterns in the same module

---

## Next Steps

1. **Domain-Driven Design**: Explore [DDD Patterns](./domain-driven-design.md) for entity and aggregate patterns
2. **Dependency Injection**: See [Advanced DI](./dependency-injection-advanced.md) for service composition
3. **Railway Patterns**: Review [Railway-Oriented Programming](./railway-oriented-programming.md) for result composition
4. **Error Handling**: Check [Error Handling Guide](./error-handling.md) for comprehensive error patterns
5. **API Reference**: Review [FlextService API](../api-reference/domain.md#flextservice) for complete API

## See Also

- [Domain-Driven Design](./domain-driven-design.md) - DDD patterns with FlextModels
- [Dependency Injection Advanced](./dependency-injection-advanced.md) - Service composition with DI
- [Railway-Oriented Programming](./railway-oriented-programming.md) - Result composition patterns
- [Error Handling Guide](./error-handling.md) - Comprehensive error handling
- [API Reference: FlextService](../api-reference/domain.md#flextservice) - Complete service API
- **FLEXT CLAUDE.md**: Architecture principles and development workflow

## References

- `flext_core/service.py` – Service base class
- `flext_core/mixins.py` – Infrastructure properties
- `flext_core/result.py` – FlextResult monad
- [CQRS Architecture](../architecture/cqrs.md)

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_service.py` for comprehensive service pattern examples and test cases.
