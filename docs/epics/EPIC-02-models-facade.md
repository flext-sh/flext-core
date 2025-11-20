# EPIC-02: Models de Automation + Fachada Pura

**Phase**: 2
**Duration**: Week 1-2 (3-5 days)
**Risk**: 🟢 Low
**LOC Impact**: -30 to -50
**Dependencies**: EPIC-01 (automation.py ready)
**Status**: 🟡 WAITING ON EPIC-01

---

## 🎯 OBJECTIVE

1. Create automation models (`RetryOptions`, `CacheOptions`, etc.) in `_models/automation.py`
2. Refactor `FlextModels` into **pure facade** (no logic, only assignments/wrappers)
3. Eliminate all business `dict[str, object]` identified in Phase 0
4. Expose everything through `FlextModels.Automation` namespace

**Why This Matters**: Models are the contract. Get them right, and quad API becomes trivial.

---

## 📋 TASKS CHECKLIST

### Task 2.1: Create `_models/automation.py`

- [ ] Create module: `src/flext_core/_models/automation.py`
- [ ] Implement core automation models:
  - `ExceptionContext`
  - `RetryOptions`
  - `CacheOptions`
  - `TimeoutOptions`
  - `ValidationOptions`
  - `RateLimitOptions`
  - `CircuitBreakerOptions`
  - `BulkheadOptions`
  - `FallbackOptions`
- [ ] Use `Field()` with constraints (ge, le, gt, lt) instead of validators where possible
- [ ] Add docstrings with examples
- [ ] Add unit tests for validation

**Implementation Template**:

```python
"""
Automation models for flext-core.

Models supporting the quad API pattern for configuration objects.
All models here follow these rules:
- Use Field() constraints instead of custom validators when possible
- Include comprehensive docstrings with examples
- Provide sensible defaults
- Are frozen=False for flexibility (users can modify after creation)
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class ExceptionContext(BaseModel):
    """
    Context information for exceptions.

    Replaces dict[str, Any] for exception metadata.

    Attributes:
        correlation_id: Unique request identifier
        operation: Operation being performed
        metadata: Additional context data
        auto_log: Whether to auto-log this exception
        severity: Exception severity level

    Examples:
        >>> ctx = ExceptionContext(
        ...     correlation_id="req-123",
        ...     operation="user.create",
        ...     metadata={"user_id": 456},
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        extra="allow",  # Allow additional fields for flexibility
    )

    correlation_id: str = Field(
        default="",
        description="Unique request/operation identifier",
    )
    operation: str = Field(
        default="",
        description="Operation being performed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data",
    )
    auto_log: bool = Field(
        default=True,
        description="Whether to automatically log this exception",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Exception severity level",
    )


class RetryOptions(BaseModel):
    """
    Retry behavior configuration.

    Attributes:
        max_attempts: Maximum retry attempts (1-10)
        backoff_factor: Exponential backoff multiplier (≥1.0)
        initial_delay: Initial delay in seconds (≥0)
        max_delay: Maximum delay in seconds (>= initial_delay)
        jitter: Whether to add random jitter to delays
        retry_on: Exception types to retry on
        exclude_exceptions: Exception types to never retry

    Examples:
        >>> opts = RetryOptions(max_attempts=5, backoff_factor=2.0)
        >>> opts = RetryOptions(
        ...     max_attempts=3,
        ...     backoff_factor=1.5,
        ...     jitter=True,
        ...     retry_on=[TimeoutError, ConnectionError],
        ... )
    """

    model_config = ConfigDict(frozen=False)

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts",
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier",
    )
    initial_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial delay in seconds before first retry",
    )
    max_delay: float = Field(
        default=60.0,
        ge=0.0,
        description="Maximum delay between retries in seconds",
    )
    jitter: bool = Field(
        default=False,
        description="Add random jitter to retry delays",
    )
    retry_on: list[type[Exception]] = Field(
        default_factory=list,
        description="Exception types that trigger retry",
    )
    exclude_exceptions: list[type[Exception]] = Field(
        default_factory=list,
        description="Exception types that prevent retry",
    )


class CacheOptions(BaseModel):
    """
    Caching behavior configuration.

    Attributes:
        ttl: Time-to-live in seconds (>0 or None for infinite)
        max_size: Maximum cache entries (≥1)
        strategy: Cache eviction strategy
        key_func: Custom key generation function name
        namespace: Cache namespace for isolation

    Examples:
        >>> opts = CacheOptions(ttl=300, max_size=1000)
        >>> opts = CacheOptions(ttl=None, strategy="lfu")  # LFU, infinite TTL
    """

    model_config = ConfigDict(frozen=False)

    ttl: float | None = Field(
        default=300.0,
        gt=0.0,
        description="Time-to-live in seconds (None = infinite)",
    )
    max_size: int = Field(
        default=100,
        ge=1,
        description="Maximum number of cached entries",
    )
    strategy: Literal["lru", "lfu", "fifo"] = Field(
        default="lru",
        description="Cache eviction strategy",
    )
    key_func: str | None = Field(
        default=None,
        description="Custom key generation function name",
    )
    namespace: str = Field(
        default="default",
        description="Cache namespace for isolation",
    )


class TimeoutOptions(BaseModel):
    """
    Timeout behavior configuration.

    Attributes:
        timeout: Timeout in seconds (>0)
        raise_on_timeout: Whether to raise exception on timeout
        default_value: Value to return on timeout (if not raising)

    Examples:
        >>> opts = TimeoutOptions(timeout=30.0)
        >>> opts = TimeoutOptions(timeout=5.0, raise_on_timeout=False, default_value=None)
    """

    model_config = ConfigDict(frozen=False)

    timeout: float = Field(
        gt=0.0,
        description="Timeout duration in seconds",
    )
    raise_on_timeout: bool = Field(
        default=True,
        description="Whether to raise exception on timeout",
    )
    default_value: Any = Field(
        default=None,
        description="Default value to return on timeout (if not raising)",
    )


class ValidationOptions(BaseModel):
    """
    Validation behavior configuration.

    Attributes:
        strict: Use strict validation mode
        allow_extra: Allow extra fields in input
        validate_assignment: Validate on field assignment
        coerce_types: Attempt type coercion

    Examples:
        >>> opts = ValidationOptions(strict=True, allow_extra=False)
    """

    model_config = ConfigDict(frozen=False)

    strict: bool = Field(
        default=False,
        description="Use strict validation mode",
    )
    allow_extra: bool = Field(
        default=False,
        description="Allow extra fields in input data",
    )
    validate_assignment: bool = Field(
        default=True,
        description="Validate values on field assignment",
    )
    coerce_types: bool = Field(
        default=True,
        description="Attempt automatic type coercion",
    )


class RateLimitOptions(BaseModel):
    """
    Rate limiting configuration.

    Attributes:
        calls: Number of calls allowed
        period: Time period in seconds (>0)
        strategy: Rate limit strategy
        raise_on_limit: Whether to raise exception when limit exceeded

    Examples:
        >>> opts = RateLimitOptions(calls=100, period=60.0)  # 100 calls/minute
        >>> opts = RateLimitOptions(calls=10, period=1.0, strategy="sliding")
    """

    model_config = ConfigDict(frozen=False)

    calls: int = Field(
        ge=1,
        description="Number of calls allowed in the period",
    )
    period: float = Field(
        gt=0.0,
        description="Time period in seconds",
    )
    strategy: Literal["fixed", "sliding"] = Field(
        default="fixed",
        description="Rate limiting strategy",
    )
    raise_on_limit: bool = Field(
        default=True,
        description="Whether to raise exception when limit exceeded",
    )


class CircuitBreakerOptions(BaseModel):
    """
    Circuit breaker configuration.

    Attributes:
        failure_threshold: Failures before opening circuit (≥1)
        success_threshold: Successes to close circuit (≥1)
        timeout: Circuit open timeout in seconds (>0)
        half_open_max_calls: Max calls in half-open state (≥1)

    Examples:
        >>> opts = CircuitBreakerOptions(
        ...     failure_threshold=5,
        ...     success_threshold=2,
        ...     timeout=60.0,
        ... )
    """

    model_config = ConfigDict(frozen=False)

    failure_threshold: int = Field(
        ge=1,
        description="Number of failures before opening circuit",
    )
    success_threshold: int = Field(
        ge=1,
        description="Number of successes to close circuit",
    )
    timeout: float = Field(
        gt=0.0,
        description="Time in seconds circuit stays open",
    )
    half_open_max_calls: int = Field(
        default=1,
        ge=1,
        description="Maximum calls allowed in half-open state",
    )


class BulkheadOptions(BaseModel):
    """
    Bulkhead isolation configuration.

    Attributes:
        max_concurrent: Maximum concurrent executions (≥1)
        max_queue: Maximum queued requests (≥0)
        timeout: Queue timeout in seconds (>0)

    Examples:
        >>> opts = BulkheadOptions(max_concurrent=10, max_queue=50)
    """

    model_config = ConfigDict(frozen=False)

    max_concurrent: int = Field(
        ge=1,
        description="Maximum concurrent executions",
    )
    max_queue: int = Field(
        default=0,
        ge=0,
        description="Maximum queued requests (0 = reject immediately)",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Queue timeout in seconds",
    )


class FallbackOptions(BaseModel):
    """
    Fallback behavior configuration.

    Attributes:
        fallback_value: Static value to return on failure
        fallback_func: Function name to call on failure
        exceptions: Exception types that trigger fallback

    Examples:
        >>> opts = FallbackOptions(fallback_value=[])
        >>> opts = FallbackOptions(
        ...     fallback_func="get_cached_data",
        ...     exceptions=[TimeoutError, ConnectionError],
        ... )
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    fallback_value: Any = Field(
        default=None,
        description="Static value to return on failure",
    )
    fallback_func: str | None = Field(
        default=None,
        description="Function name to call on failure",
    )
    exceptions: list[type[Exception]] = Field(
        default_factory=lambda: [Exception],
        description="Exception types that trigger fallback",
    )
```

---

### Task 2.2: Refactor `FlextModels` as Pure Facade

- [ ] Move all model definitions from `models.py` to `_models/*.py`
- [ ] Convert `FlextModels` to thin wrapper/assignment pattern
- [ ] Create `FlextModels.Automation` namespace
- [ ] Verify no logic remains in `models.py`

**Implementation** (`src/flext_core/models.py`):

```python
"""
FlextModels - Pure facade for all Pydantic models.

This module provides a single access point for all models without
defining any models directly. All models are defined in _models/*.py
and simply exposed here.

Rules:
- NO generic definitions here (use _models/)
- NO business logic here (pure facade)
- Only wrappers (for non-generics) and assignments (for generics)
"""

from __future__ import annotations

# Import model modules
from flext_core._models import (
    base as FlextModelsBase,
    automation as FlextModelsAutomation,
    # ... other model modules
)


# Non-generic models: thin wrappers
class ExceptionContext(FlextModelsAutomation.ExceptionContext):
    """Exception context model (wrapper for backward compatibility)."""


# Generic models: simple assignment
Payload = FlextModelsBase.Payload
Result = FlextModelsBase.Result


# Namespace for automation models
class Automation:
    """Automation-related models."""

    ExceptionContext = FlextModelsAutomation.ExceptionContext
    RetryOptions = FlextModelsAutomation.RetryOptions
    CacheOptions = FlextModelsAutomation.CacheOptions
    TimeoutOptions = FlextModelsAutomation.TimeoutOptions
    ValidationOptions = FlextModelsAutomation.ValidationOptions
    RateLimitOptions = FlextModelsAutomation.RateLimitOptions
    CircuitBreakerOptions = FlextModelsAutomation.CircuitBreakerOptions
    BulkheadOptions = FlextModelsAutomation.BulkheadOptions
    FallbackOptions = FlextModelsAutomation.FallbackOptions


# Re-export for convenience
__all__ = [
    "ExceptionContext",
    "Payload",
    "Result",
    "Automation",
]
```

---

### Task 2.3: Update Protocols for Models

- [ ] Add protocols for new models in `protocols.py`
- [ ] Create `ExceptionContextProtocol`
- [ ] Create `RetryOptionsProtocol` (if needed for breaking cycles)
- [ ] Update existing protocols to match new models

**Implementation** (add to `src/flext_core/protocols.py`):

```python
@runtime_checkable
class ExceptionContextProtocol(Protocol):
    """Protocol for exception context objects."""

    correlation_id: str
    operation: str
    metadata: dict[str, Any]
    auto_log: bool
    severity: str

    def model_dump(self) -> dict[str, Any]: ...


@runtime_checkable
class RetryOptionsProtocol(Protocol):
    """Protocol for retry options (if needed to break cycles)."""

    max_attempts: int
    backoff_factor: float
    initial_delay: float
    max_delay: float
    jitter: bool

    def model_dump(self) -> dict[str, Any]: ...
```

---

### Task 2.4: Migrate Business Dicts

- [ ] Review dict classification from Phase 0
- [ ] For each business dict, determine which model to use
- [ ] Replace dict[str, Any] with appropriate model
- [ ] Update calling code to use models
- [ ] Run tests after each migration batch

**Example Migration**:

```python
# ❌ Before
def process_handler(
    handler_name: str,
    context: dict[str, Any],
) -> FlextResult[Any]:
    correlation_id = context.get("correlation_id", "")
    metadata = context.get("metadata", {})
    # ...

# ✅ After
def process_handler(
    handler_name: str,
    context: ExceptionContext,
) -> FlextResult[Any]:
    correlation_id = context.correlation_id
    metadata = context.metadata
    # ...

# Callers can use quad API:
process_handler("my_handler", ExceptionContext(correlation_id="123"))
process_handler("my_handler", {"correlation_id": "123"})
process_handler("my_handler", correlation_id="123")  # if using autoschema
```

---

### Task 2.5: Update Tests

- [ ] Add tests for all automation models
- [ ] Test Field() constraints (ge, le, gt, lt)
- [ ] Test model_dump() / model_dump_json()
- [ ] Test quad API usage with models
- [ ] Update existing tests to use models instead of dicts

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] `_models/automation.py` created with 9 models
- [ ] `FlextModels` refactored to pure facade
- [ ] All business dicts migrated to models
- [ ] Protocols updated
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Lint passes
- [ ] No circular dependencies
- [ ] Coverage maintained (≥79%)
- [ ] PR created: `feat(core): add automation models and refactor facade`

### Validation

```bash
cd /home/marlonsc/flext/flext-core

make lint
make type-check
make test
python scripts/detect_cycles.py
python scripts/analyze_dicts.py  # Should show 0 business dicts

# Check LOC
cloc src/ > docs/metrics/phase2_loc.txt
diff docs/metrics/phase1_loc.txt docs/metrics/phase2_loc.txt
```

---

## 📊 SUCCESS METRICS

### Quantitative
- 9 automation models created
- 100% of business dicts converted
- LOC reduced by 30-50 (net)
- 0 business dicts remaining
- Coverage ≥79%

### Qualitative
- Models are intuitive
- Quad API works seamlessly
- No confusion about which model to use

---

## 🔗 DEPENDENCIES

### Requires
- **EPIC-01**: `automation.py` and helpers

### Blocks
- **EPIC-03**: Exceptions need ExceptionContext model
- **EPIC-04**: Decorators need Options models

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model design doesn't fit all cases | High | Review with team, iterate on design |
| Breaking changes for existing code | Medium | Provide migration guide, deprecation warnings |
| Performance overhead | Low | Models are lightweight, profile if concerned |

---

**Prev**: [EPIC-01: Automation Core](./EPIC-01-automation-core.md)
**Next**: [EPIC-03: Exceptions Quad API](./EPIC-03-exceptions-quad.md)

**Status**: 🟡 WAITING ON EPIC-01
