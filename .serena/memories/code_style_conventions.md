# FLEXT-CORE Code Style & Conventions

## Architecture Principles

### 1. Unified Class Pattern (MANDATORY)

```python
class UnifiedProjectService(FlextService):
    """Single responsibility class with nested helpers."""

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    class _ValidationHelper:
        """Nested helper class - no loose functions."""
        @staticmethod
        def validate_business_rules(data: dict) -> FlextResult[dict]:
            pass
```

### 2. FlextResult Pattern (MANDATORY)

- ALL operations return `FlextResult[T]` for type-safe error handling
- Use `.unwrap()` for safe extraction, `.is_failure` for checks
- NO try/except fallbacks - explicit error checking only

### 3. Import Strategy (ROOT-LEVEL ONLY)

```python
# ✅ CORRECT
from flext_core import FlextResult, FlextLogger, FlextContainer

# ❌ FORBIDDEN
from flext_core.result import FlextResult  # Internal imports prohibited
```

## Quality Standards

### Code Quality (ZERO TOLERANCE)

- **MyPy Strict Mode**: ZERO errors in `src/` directory
- **PyRight Validation**: ZERO errors
- **Line Length**: 79 characters maximum (PEP8 strict)
- **Type Hints**: Required for ALL functions, class attributes, public APIs
- **Naming Convention**: `FlextXxx` prefix for ALL public exports
- **Docstrings**: Required for ALL public APIs (Google style)

### Absolutely Forbidden

- ❌ Multiple classes per module (single unified class only)
- ❌ Helper functions outside classes (use nested classes)
- ❌ `try/except` fallback mechanisms (use explicit FlextResult)
- ❌ `# type: ignore` without specific error codes
- ❌ `object` types instead of proper annotations
- ❌ Wrappers, legacy access patterns, fallbacks
- ❌ Direct imports from internal modules

## File Organization

```
src/flext_core/
├── Foundation Layer (No Dependencies)
│   ├── result.py           # FlextResult[T] railway pattern
│   ├── container.py        # Dependency injection
│   ├── exceptions.py       # Exception hierarchy
│   └── constants.py        # Constants and enums
├── Domain Layer (Depends on Foundation)
│   ├── models.py           # FlextModels (Entity/Value/AggregateRoot)
│   └── service.py  # Domain service patterns
└── Application Layer (Depends on Domain)
    ├── commands.py         # CQRS patterns
    └── handlers.py         # Handler registry
```
