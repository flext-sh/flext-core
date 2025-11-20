# EPIC-03: Exceptions via Protocol + Quad API

**Phase**: 3
**Duration**: Week 2 (3-5 days)
**Risk**: 🟡 Medium
**LOC Impact**: +20 to -50 (overloads add, logic consolidation reduces)
**Dependencies**: EPIC-02 (ExceptionContext model ready)
**Status**: 🟡 WAITING ON EPIC-02

---

## 🎯 OBJECTIVE

1. Break circular dependency `exceptions ↔ models` using protocols
2. Implement quad API for all 13 exceptions (model/dict/kwargs/hybrid)
3. Consolidate context handling in `BaseError`
4. Maintain 100% backward compatibility

**Why This Matters**: Exceptions are used everywhere. Get quad API right here, and it proves the pattern works.

---

## 📋 TASKS CHECKLIST

### Task 3.1: Update BaseError with Protocol-Based Context

- [ ] Import `ExceptionContextProtocol` (NOT `ExceptionContext` model)
- [ ] Implement quad API in `BaseError.__init__()`
- [ ] Use `coerce_model()` from automation module
- [ ] Add overloads for all 4 API styles
- [ ] Maintain backward compatibility with existing signatures

**Implementation** (`src/flext_core/exceptions.py`):

```python
"""
FlextExceptions - Exception hierarchy with quad API support.

All exceptions support 4 ways of providing context:
1. Model: context=ExceptionContext(...)
2. Dict: context={"correlation_id": "123"}
3. Kwargs: correlation_id="123", operation="test"
4. Hybrid: context={"correlation_id": "123"}, metadata={"key": "value"}
"""

from __future__ import annotations

from typing import Any, overload
from flext_core.protocols import ExceptionContextProtocol, HasModelDump
from flext_core.automation import coerce_model


class BaseError(Exception):
    """
    Base exception with rich context support (quad API).

    Accepts context in 4 ways:
    - Model: BaseError(context=ExceptionContext(...))
    - Dict: BaseError(context={"correlation_id": "123"})
    - Kwargs: BaseError(correlation_id="123", operation="test")
    - Hybrid: BaseError(context={...}, extra_field="value")

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        context: Rich context information
        cause: Original exception (if wrapping)
    """

    # Default values (subclasses override)
    default_error_code: str = "BASE_ERROR"
    default_message: str = "An error occurred"

    @overload
    def __init__(
        self,
        message: str | None = None,
        *,
        context: ExceptionContextProtocol,
        cause: Exception | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        message: str | None = None,
        *,
        context: dict[str, Any],
        cause: Exception | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        message: str | None = None,
        *,
        cause: Exception | None = None,
        correlation_id: str = "",
        operation: str = "",
        metadata: dict[str, Any] | None = None,
        auto_log: bool = True,
        severity: str = "medium",
        **extra: Any,
    ) -> None: ...

    def __init__(
        self,
        message: str | None = None,
        *,
        context: ExceptionContextProtocol | dict[str, Any] | None = None,
        cause: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize exception with quad API support.

        Args:
            message: Error message (uses default if None)
            context: Context as model or dict
            cause: Original exception (if wrapping)
            **kwargs: Context fields (if context not provided)

        Examples:
            >>> # Style 1: Model
            >>> from flext_core.models import ExceptionContext
            >>> raise BaseError(context=ExceptionContext(correlation_id="123"))
            >>>
            >>> # Style 2: Dict
            >>> raise BaseError(context={"correlation_id": "123"})
            >>>
            >>> # Style 3: Kwargs
            >>> raise BaseError(correlation_id="123", operation="test")
            >>>
            >>> # Style 4: Hybrid
            >>> raise BaseError(
            ...     context={"correlation_id": "123"},
            ...     metadata={"extra": "data"},
            ... )
        """
        # Import model here to avoid circular dependency
        from flext_core.models import ExceptionContext

        # Message handling
        self.message = message or self.default_message
        self.error_code = self.default_error_code
        self.cause = cause

        # Context handling via quad API
        if context is not None or kwargs:
            # Coerce to model (supports all 4 patterns)
            self.context = coerce_model(ExceptionContext, context, **kwargs)
        else:
            # No context provided, use defaults
            self.context = ExceptionContext()

        # Standard exception message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format exception message with context."""
        parts = [f"[{self.error_code}] {self.message}"]

        if self.context.correlation_id:
            parts.append(f"(correlation_id={self.context.correlation_id})")

        if self.context.operation:
            parts.append(f"(operation={self.context.operation})")

        if self.cause:
            parts.append(f"caused by: {type(self.cause).__name__}: {self.cause}")

        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary.

        Returns:
            Dict with error_code, message, context, and cause

        Examples:
            >>> try:
            ...     raise BaseError(correlation_id="123")
            ... except BaseError as e:
            ...     error_dict = e.to_dict()
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context.model_dump(),
            "cause": str(self.cause) if self.cause else None,
        }
```

---

### Task 3.2: Update All 13 Exception Subclasses

- [ ] For each exception, update to use BaseError quad API
- [ ] Override only `default_error_code` and `default_message`
- [ ] Remove any custom `__init__` that duplicates BaseError logic
- [ ] Add tests for quad API usage

**Exceptions to Update**:
1. `ValidationError` → `VALIDATION_ERROR`
2. `ConfigurationError` → `CONFIGURATION_ERROR`
3. `HandlerNotFoundError` → `HANDLER_NOT_FOUND`
4. `HandlerExecutionError` → `HANDLER_EXECUTION_ERROR`
5. `TimeoutError` (if custom) → `TIMEOUT_ERROR`
6. `RetryExhaustedError` → `RETRY_EXHAUSTED`
7. `CircuitBreakerOpenError` → `CIRCUIT_BREAKER_OPEN`
8. `RateLimitExceededError` → `RATE_LIMIT_EXCEEDED`
9. `CacheError` → `CACHE_ERROR`
10. `SerializationError` → `SERIALIZATION_ERROR`
11. `DeserializationError` → `DESERIALIZATION_ERROR`
12. `AuthenticationError` → `AUTHENTICATION_ERROR`
13. `AuthorizationError` → `AUTHORIZATION_ERROR`

**Template**:

```python
class ValidationError(BaseError):
    """
    Validation failed.

    Supports quad API (model/dict/kwargs/hybrid).

    Examples:
        >>> raise ValidationError(
        ...     "Invalid field: email",
        ...     field="email",
        ...     value="invalid",
        ... )
        >>> raise ValidationError(
        ...     context={"correlation_id": "123", "operation": "validate"},
        ... )
    """

    default_error_code = "VALIDATION_ERROR"
    default_message = "Validation failed"


class RetryExhaustedError(BaseError):
    """
    All retry attempts exhausted.

    Supports quad API.

    Examples:
        >>> raise RetryExhaustedError(
        ...     f"Failed after {attempts} attempts",
        ...     attempts=attempts,
        ...     last_error=str(exc),
        ... )
    """

    default_error_code = "RETRY_EXHAUSTED"
    default_message = "All retry attempts exhausted"
```

---

### Task 3.3: Add Protocol Usage Documentation

- [ ] Document when to use `ExceptionContextProtocol` vs `ExceptionContext`
- [ ] Add examples to docstrings
- [ ] Update architecture docs

**Guideline**:
- **Use Protocol**: When importing in modules that shouldn't depend on models
  (e.g., `decorators.py`, `dispatcher.py`)
- **Use Model**: When you need full Pydantic features (validation, serialization)
- **Example**:

```python
# In decorators.py (no model dependency)
from flext_core.protocols import ExceptionContextProtocol

def my_decorator(ctx: ExceptionContextProtocol) -> None:
    data = ctx.model_dump()  # Protocol method

# In application code (can use model)
from flext_core.models import ExceptionContext

ctx = ExceptionContext(correlation_id="123")
```

---

### Task 3.4: Comprehensive Testing

- [ ] Test quad API for ALL 13 exceptions
- [ ] Test each of the 4 styles:
  - Model instance
  - Dict
  - Kwargs
  - Hybrid (dict + kwargs)
- [ ] Test backward compatibility (existing code patterns)
- [ ] Test `to_dict()` method
- [ ] Test message formatting

**Test Template** (`tests/unit/test_exceptions_quad_api.py`):

```python
import pytest
from flext_core.models import ExceptionContext
from flext_core.exceptions import ValidationError, RetryExhaustedError


class TestQuadAPIValidationError:
    """Test quad API for ValidationError."""

    def test_with_model(self):
        ctx = ExceptionContext(correlation_id="test-123", operation="validate")
        exc = ValidationError("Test error", context=ctx)
        assert exc.context.correlation_id == "test-123"
        assert exc.error_code == "VALIDATION_ERROR"

    def test_with_dict(self):
        exc = ValidationError(
            "Test error",
            context={"correlation_id": "test-456", "operation": "validate"},
        )
        assert exc.context.correlation_id == "test-456"

    def test_with_kwargs(self):
        exc = ValidationError(
            "Test error",
            correlation_id="test-789",
            operation="validate",
        )
        assert exc.context.correlation_id == "test-789"

    def test_hybrid_dict_plus_kwargs(self):
        exc = ValidationError(
            "Test error",
            context={"correlation_id": "test-000"},
            metadata={"field": "email"},
        )
        assert exc.context.correlation_id == "test-000"
        assert exc.context.metadata["field"] == "email"

    def test_to_dict(self):
        exc = ValidationError("Test", correlation_id="123")
        data = exc.to_dict()
        assert data["error_code"] == "VALIDATION_ERROR"
        assert data["message"] == "Test"
        assert data["context"]["correlation_id"] == "123"

    def test_backward_compat_no_context(self):
        """Existing code without context still works."""
        exc = ValidationError("Simple error")
        assert exc.message == "Simple error"
        assert exc.context.correlation_id == ""  # Default


# Repeat for all 13 exceptions...
```

---

### Task 3.5: Update Callers

- [ ] Search for exception raises: `raise ValidationError(...)`
- [ ] Update to use models/dicts/kwargs where appropriate
- [ ] Add context where it was missing
- [ ] Run tests after each batch

**Migration Examples**:

```python
# ❌ Before
raise ValidationError("Invalid input")

# ✅ After (add context)
raise ValidationError(
    "Invalid input",
    correlation_id=request.correlation_id,
    operation="user.validate",
)

# Or with dict
raise ValidationError(
    "Invalid input",
    context={
        "correlation_id": request.correlation_id,
        "operation": "user.validate",
    },
)
```

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] `BaseError` implements quad API
- [ ] All 13 exceptions inherit quad API
- [ ] Protocol-based imports (no circular deps)
- [ ] Comprehensive tests (4 styles × 13 exceptions)
- [ ] All existing tests pass
- [ ] Type checking passes
- [ ] Lint passes
- [ ] No circular dependencies
- [ ] Coverage maintained (≥79%)
- [ ] Migration guide created
- [ ] PR: `feat(core): add quad API to exceptions`

### Validation

```bash
cd /home/marlonsc/flext/flext-core

make lint
make type-check
make test
python scripts/detect_cycles.py  # Must pass

# Check tests coverage
pytest tests/unit/test_exceptions_quad_api.py -v --cov=src/flext_core/exceptions
```

---

## 📊 SUCCESS METRICS

### Quantitative
- 13 exceptions support quad API
- 52+ test cases (4 styles × 13 exceptions)
- 0 circular dependencies
- Coverage ≥79%
- LOC: +20 to -50 (net near zero)

### Qualitative
- Exception usage more consistent
- Better error context everywhere
- Easier debugging with rich context

---

## 🔗 DEPENDENCIES

### Requires
- **EPIC-02**: ExceptionContext model
- **EPIC-01**: coerce_model() helper

### Blocks
- **EPIC-04**: Decorators will use new exceptions
- **EPIC-05**: Dispatcher will use new exceptions

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes in exception signatures | High | Maintain backward compat, add deprecation warnings |
| Protocol doesn't cover all use cases | Medium | Iterate on protocol design, add methods as needed |
| Overload complexity confuses users | Medium | Comprehensive docs and examples |

---

**Prev**: [EPIC-02: Models & Facade](./EPIC-02-models-facade.md)
**Next**: [EPIC-04: Decorators Unified](./EPIC-04-decorators.md)

**Status**: 🟡 WAITING ON EPIC-02
