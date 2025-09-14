# FLEXT Core - Actual Capabilities Reference

**Purpose**: This document provides an honest, verified assessment of what FLEXT Core actually provides based on the source code reality.

**Usage**: All other documentation should reference this file to avoid duplication and false claims.

---

## Verified Project Status (2025-01-13)

- **Test Coverage**: 83% (verified via pytest)
- **Ecosystem Projects**: 29 active projects (verified via directory listing)
- **Python Version**: 3.13+ required
- **Architecture**: Simplified, practical patterns (not "enterprise 7-layer" as claimed elsewhere)

---

## Actual Module Capabilities

### ✅ Core Foundation (VERIFIED)

**File**: `src/flext_core/result.py`
- `FlextResult[T]` - Railway-oriented programming
- Methods: `.ok()`, `.fail()`, `.map()`, `.flat_map()`, `.filter()`, `.unwrap()`
- **Note**: NO `.map_error()` method exists (contrary to some documentation)

**File**: `src/flext_core/container.py`  
- `FlextContainer.get_global()` - Dependency injection singleton
- Basic service registration and retrieval
- **Note**: Import via `from flext_core import FlextContainer`, NOT `get_flext_container`

### ✅ Domain Models (VERIFIED)

**File**: `src/flext_core/models.py`
- `FlextModels.Entity` - Basic entity with ID, version, domain events
- `FlextModels.Value` - Value object base class
- `FlextModels.AggregateRoot` - Simple aggregate root (just entity + version field)
- **Reality**: Basic DDD support, not "enterprise-grade comprehensive" as claimed

### ✅ Processing (VERIFIED)

**File**: `src/flext_core/processing.py` (aliased as FlextProcessing)
- `Handler` - Simple handler protocol
- `HandlerRegistry` - Basic handler registration
- `Pipeline` - Simple pipeline pattern
- **Reality**: Basic handler support, not "8 design patterns" or "7-layer architecture"

### ✅ Other Verified Modules

- `config.py` - Configuration management with Pydantic
- `loggings.py` - Structured logging with structlog
- `validations.py` - Basic validation patterns
- `exceptions.py` - Exception hierarchy
- `utilities.py` - Helper functions
- `constants.py` - Enums and constants

### ❌ Missing Modules (Referenced in docs but don't exist)

- `handlers.py` - Doesn't exist (use `processing.py`)
- `interfaces.py` - Doesn't exist (use `protocols.py`)
- `validation.py` - Doesn't exist (use `validations.py`)
- `observability.py` - Doesn't exist (use `loggings.py`)
- `payload.py` - Doesn't exist (use `models.py`)

---

## Architecture Reality

### Actual File Structure
```
src/flext_core/
├── Foundation
│   ├── result.py          # Railway pattern
│   ├── container.py       # DI container
│   ├── exceptions.py      # Exception hierarchy
│   └── constants.py       # Constants
├── Domain
│   ├── models.py          # DDD basic patterns
│   └── domain_services.py # Domain services
├── Application
│   ├── commands.py        # CQRS commands
│   ├── processing.py      # Handler processing
│   └── validations.py     # Validation
└── Infrastructure
    ├── config.py          # Configuration
    ├── loggings.py        # Logging
    ├── context.py         # Context
    └── adapters.py        # Type adapters
```

### What Works vs What's Claimed

**✅ Works**: Basic railway pattern, DI container, simple domain models, basic handlers
**❌ Over-claimed**: "Enterprise 7-layer architecture", "8 design patterns", "comprehensive systems"

---

## API Examples (VERIFIED)

### Railway Pattern (Works)
```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult[str].fail("Empty data")
    return FlextResult[str].ok(data.upper())

# Chain operations (verified methods only)
result = (
    process_data("hello")
    .map(lambda s: s + " WORLD")
    .filter(lambda s: len(s) > 5, "Too short")
)
```

### Dependency Injection (Works)
```python
from flext_core import FlextContainer

container = FlextContainer.get_global()
container.register("service", MyService())
service_result = container.get("service")
```

### Domain Models (Basic functionality)
```python
from flext_core import FlextModels

class User(FlextModels.Entity):
    name: str
    
    def activate(self) -> None:
        self.add_domain_event(FlextModels.Event(type="UserActivated"))
```

---

## Documentation Guidelines

**For all other documentation files**:

1. **Reference this file** instead of making independent claims
2. **Verify examples** against actual code before publishing
3. **Use measured language** - avoid "enterprise", "comprehensive", "7-layer" claims
4. **Test code examples** to ensure they actually work

**Example reference usage**:
```markdown
<!-- In any docs file -->
> **Note**: For verified capabilities, see [ACTUAL_CAPABILITIES.md](../ACTUAL_CAPABILITIES.md)
```

---

**Last Updated**: 2025-01-13
**Next Review**: When adding new features (verify capabilities before documenting)
