# FLEXT-Core API Stability Guarantees

**Version**: 1.0.0 | **Date**: 2025-10-01 | **Status**: STABLE

## Stability Promise

FLEXT-Core provides **absolute API stability guarantees** for the 1.x series. All code written against 1.0.0 will continue to work with any 1.x.y release without modifications.

## Guaranteed Stable APIs

### Level 1: Core Foundation (100% Stable)

These APIs form the foundation and are **guaranteed stable forever** in the 1.x series:

#### FlextResult - Railway-Oriented Programming

```python
from flext_core import FlextResult

# ✅ GUARANTEED STABLE - All methods and properties
result = FlextResult[T].ok(value)          # Create success result
result = FlextResult[T].fail(error)        # Create failure result

# Properties (all guaranteed)
result.is_success: bool                     # Check if successful
result.is_failure: bool                     # Check if failed
result.value: T | None                      # Get value (success) or None
result.data: T | None                       # Alias for .value (permanent)
result.error: str | None                    # Get error message (failure)

# Methods (all guaranteed)
result.unwrap() -> T                        # Get value or raise
result.unwrap_or(default: T) -> T          # Get value or default
result.map(func: Callable[[T], U]) -> FlextResult[U]  # Transform value
result.bind(func: Callable[[T], FlextResult[U]]) -> FlextResult[U]  # Chain operations
```

**Dual Access Promise**: Both `.value` and `.data` are **permanently supported** for accessing success values. This ensures backward compatibility with any existing code.

#### FlextContainer - Dependency Injection

```python
from flext_core import FlextContainer

# ✅ GUARANTEED STABLE - All methods
container = FlextContainer.get_global()    # Get singleton instance
container.register(interface, impl)        # Register implementation
container.register_factory(interface, factory)  # Register factory
container.register_singleton(interface, instance)  # Register singleton
container.resolve(interface) -> T          # Resolve dependency
container.resolve_all(interface) -> list[T]  # Resolve all implementations
container.clear()                          # Clear registrations
```

#### FlextModels - Domain-Driven Design

```python
from flext_core import FlextModels

# ✅ GUARANTEED STABLE - All base classes
class MyEntity(FlextModels.Entity):
    """Entity with identity."""
    id: str  # Inherited identity field

class MyValue(FlextModels.Value):
    """Value object (immutable)."""
    pass

class MyAggregate(FlextModels.AggregateRoot):
    """Aggregate root with domain events."""
    pass

# All base functionality guaranteed:
# - Field validation (Pydantic 2.x)
# - Equality comparison
# - Immutability (for Value objects)
# - Event handling (for Aggregates)
```

#### FlextService - Service Pattern

```python
from flext_core import FlextService

# ✅ GUARANTEED STABLE
class MyService(FlextService[ConfigType]):
    """Base service class."""

    def __init__(self) -> None:
        super().__init__()  # Initialize service

    async def execute(self, input: TInput) -> FlextResult[TOutput]:
        """Execute service operation."""
        pass

# Guaranteed methods:
# - __init__(): Initialization
# - Service lifecycle hooks
# - Integration with FlextContainer
```

#### FlextLogger - Structured Logging

```python
from flext_core import FlextLogger

# ✅ GUARANTEED STABLE
logger = FlextLogger(__name__)

# All log levels guaranteed:
logger.debug("message", extra={"key": "value"})
logger.info("message", extra={"key": "value"})
logger.warning("message", extra={"key": "value"})
logger.error("message", extra={"key": "value"})
logger.critical("message", extra={"key": "value"})
logger.exception("message", exc_info=True)

# Structured logging with extra context guaranteed
```

### Level 2: Advanced Patterns (99% Stable)

These APIs are stable but may receive minor enhancements in compatible ways:

#### FlextContext - Context Management

```python
from flext_core import FlextContext

# ✅ STABLE - Core functionality guaranteed
context = FlextContext()
context.set("key", value)
context.get("key") -> Any
context.clear()

# May receive: New helper methods (backward compatible)
```

#### FlextCqrs - Command Query Responsibility Segregation

```python
from flext_core import FlextCqrs

# ✅ STABLE - Core CQRS pattern
cqrs = FlextCqrs()
result = await cqrs.execute_command(command)
result = await cqrs.execute_query(query)

# May receive: New handler types (backward compatible)
```

#### FlextBus - Event Bus

```python
from flext_core import FlextBus

# ✅ STABLE - Core event bus
bus = FlextBus()
await bus.publish(event)
bus.subscribe(event_type, handler)

# May receive: New event types (backward compatible)
```

#### FlextRegistry - Component Registration

```python
from flext_core import FlextRegistry

# ✅ STABLE - Core registry
registry = FlextRegistry()
registry.register(name, component)
component = registry.get(name)

# May receive: New query methods (backward compatible)
```

### Level 3: Utilities & Helpers (95% Stable)

Utility functions and helpers are stable but may receive enhancements:

#### FlextUtilities

```python
from flext_core import FlextUtilities

# ✅ STABLE - Core utilities
FlextUtilities.safe_parse(data, model)
FlextUtilities.deep_merge(dict1, dict2)

# May receive: New utility functions (backward compatible)
```

#### FlextConfig

```python
from flext_core import FlextConfig

# ✅ STABLE - Configuration management
config = FlextConfig.get_global_instance()
config.get("key", default)
config.set("key", value)

# May receive: New configuration sources (backward compatible)
```

## What Changes Are Allowed

### ✅ Allowed Changes (No Version Bump Required)

1. **Documentation improvements**
2. **Internal refactoring** (no public API changes)
3. **Performance optimizations**
4. **Security patches** (within same API)

### ✅ Allowed in PATCH Releases (1.0.0 → 1.0.1)

1. **Bug fixes** that don't change expected behavior
2. **Security fixes**
3. **Documentation corrections**
4. **Internal implementation improvements**

### ✅ Allowed in MINOR Releases (1.0.0 → 1.1.0)

1. **New classes, methods, functions** (additions only)
2. **New optional parameters** with defaults
3. **Deprecation warnings** (features still work)
4. **Performance improvements**
5. **New patterns or utilities**

### ❌ Forbidden in 1.x Series (Requires 2.0.0)

1. **Removing public APIs**
2. **Changing method signatures** without backward compatibility
3. **Changing expected behavior** of existing operations
4. **Removing deprecated features** (must wait for 2.0.0)
5. **Breaking Pydantic model changes**

## Deprecation Process

### Timeline

1. **Version 1.x**: Feature works, deprecation warning added
2. **Version 1.x+1**: Feature still works with warning
3. **Version 2.0.0**: Feature removed (major version)

### Example

```python
# Version 1.0.0 - Feature exists
def old_method():
    return "result"

# Version 1.1.0 - Feature deprecated
import warnings

def old_method():
    warnings.warn(
        "old_method() is deprecated since 1.1.0 and will be removed in 2.0.0. "
        "Use new_method() instead. "
        "Migration: Replace old_method() with new_method().",
        DeprecationWarning,
        stacklevel=2
    )
    return new_method()  # Still works!

def new_method():
    return "result"

# Version 1.2.0 - Still works with warning
# Version 2.0.0 - Feature removed (breaking change)
```

## HTTP Primitives (NEW in 0.9.9)

### FlextConstants.Http - HTTP Constants

```python
from flext_core import FlextConstants

# ✅ GUARANTEED STABLE - HTTP constants
FlextConstants.Http.HTTP_OK                  # 200
FlextConstants.Http.HTTP_CREATED             # 201
FlextConstants.Http.HTTP_BAD_REQUEST         # 400
FlextConstants.Http.HTTP_NOT_FOUND           # 404
FlextConstants.Http.HTTP_INTERNAL_SERVER_ERROR  # 500

# Status ranges
FlextConstants.Http.HTTP_SUCCESS_MIN         # 200
FlextConstants.Http.HTTP_SUCCESS_MAX         # 299
FlextConstants.Http.HTTP_CLIENT_ERROR_MIN    # 400
FlextConstants.Http.HTTP_CLIENT_ERROR_MAX    # 499

# Methods
FlextConstants.Http.Method.GET
FlextConstants.Http.Method.POST
FlextConstants.Http.Method.PUT
FlextConstants.Http.Method.DELETE
FlextConstants.Http.Method.PATCH

# Content types
FlextConstants.Http.ContentType.JSON
FlextConstants.Http.ContentType.XML
FlextConstants.Http.ContentType.FORM

# Ports
FlextConstants.Http.HTTP_PORT                # 80
FlextConstants.Http.HTTPS_PORT               # 443
```

### FlextModels.HttpRequest/HttpResponse - HTTP Models

```python
from flext_core import FlextModels

# ✅ GUARANTEED STABLE - HTTP base models
class HttpRequest(FlextModels.HttpRequest):
    """Extend HTTP request base."""
    # Inherited fields:
    # - url: str
    # - method: str
    # - headers: dict[str, str]
    # - body: str | dict | None
    # - timeout: float

    # Inherited computed fields:
    # - is_secure: bool
    # - has_body: bool

class HttpResponse(FlextModels.HttpResponse):
    """Extend HTTP response base."""
    # Inherited fields:
    # - status_code: int
    # - headers: dict[str, str]
    # - body: str | dict | None
    # - elapsed_time: float

    # Inherited computed fields:
    # - is_success: bool
    # - is_client_error: bool
    # - is_server_error: bool
```

## Testing Compatibility

### How We Ensure Stability

1. **Comprehensive test suite** (79%+ coverage)
2. **Integration tests** with all dependent projects
3. **Type checking** (MyPy strict + PyRight)
4. **Automated compatibility checks** in CI/CD
5. **Deprecation warnings** caught in tests

### Ecosystem Testing

Before any release, we validate with:
- **flext-api**: HTTP client integration
- **flext-cli**: CLI framework integration
- **flext-auth**: Authentication integration
- **flext-web**: Web framework integration
- **flext-ldap**: LDAP integration

## Contact & Support

If you discover an API stability issue:

1. **Report it**: https://github.com/flext-sh/flext-core/issues
2. **Label**: Use "stability-guarantee" label
3. **Priority**: Stability issues are treated as P0 bugs
4. **Fix timeline**: Hotfix within 48 hours

## Version-Specific Guarantees

### 1.0.x Series (October 2025 - TBD)

- Python 3.13 support guaranteed
- Pydantic 2.x API stable
- All Level 1 APIs guaranteed forever
- Level 2 APIs guaranteed with minor enhancements
- Level 3 APIs stable with possible additions

### 1.x.y Series (Future)

- May add Python 3.14+ support (minor version)
- All 1.0.x APIs remain stable
- New features added in backward-compatible way
- Deprecations announced minimum 2 versions early

### 2.0.0 (Future - 2026+)

- May include breaking changes
- Complete migration guide provided
- Automated migration tools available
- Minimum 6 months notice before release

## Summary

**FLEXT-Core 1.x Stability Promise**:

✅ **Zero breaking changes** in 1.x series
✅ **Backward compatibility** guaranteed
✅ **Deprecation cycle**: Minimum 2 minor versions
✅ **Migration support**: Complete documentation
✅ **Rapid fixes**: Stability issues = P0 bugs

**Trust the foundation. Build with confidence.**
