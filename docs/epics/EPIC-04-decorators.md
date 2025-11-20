# EPIC-04: Decorators Unificados

**Phase**: 4
**Duration**: Week 3 (5 days)
**Risk**: 🟡 Medium
**LOC Impact**: -100 to -150
**Dependencies**: EPIC-01, EPIC-02, EPIC-03
**Status**: 🟡 WAITING ON EPIC-03

---

## 🎯 OBJECTIVE

1. Standardize ALL decorators to use quad API pattern
2. Consolidate validation/coercion logic (one place: `coerce_model()`)
3. Use automation models (`RetryOptions`, `CacheOptions`, etc.)
4. Eliminate duplication across 7-8 decorators

**Why This Matters**: Decorators are the public API. Consistency here = better DX.

---

## 📋 TASKS CHECKLIST

### Task 4.1: Create Decorator Template

- [ ] Document the standard decorator pattern
- [ ] Create template with overloads
- [ ] Add to `docs/guides/decorator-pattern.md`

**Template**:

```python
"""
Standard Decorator Pattern with Quad API.

Use this template for all flext-core decorators.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, overload
from functools import wraps
from flext_core.automation import coerce_model
from flext_core.models import Automation  # For options models
from flext_core.exceptions import <RelevantError>

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@overload
def my_decorator(options: Automation.MyOptions) -> Callable[[F], F]: ...


@overload
def my_decorator(options: dict[str, Any]) -> Callable[[F], F]: ...


@overload
def my_decorator(**kwargs: Any) -> Callable[[F], F]: ...


def my_decorator(
    options: Automation.MyOptions | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """
    Decorator with quad API support.

    Accepts:
    - Model: @my_decorator(MyOptions(...))
    - Dict: @my_decorator({"param": value})
    - Kwargs: @my_decorator(param=value)
    - Hybrid: @my_decorator({"param1": v1}, param2=v2)

    Args:
        options: Configuration as model or dict
        **kwargs: Configuration as kwargs

    Returns:
        Decorated function

    Examples:
        >>> @my_decorator(Automation.MyOptions(param=10))
        ... def func(): ...
        >>>
        >>> @my_decorator({"param": 10})
        ... def func(): ...
        >>>
        >>> @my_decorator(param=10)
        ... def func(): ...
    """
    # Coerce to model (handles all 4 patterns)
    opts = coerce_model(Automation.MyOptions, options, **kwargs)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Implementation using opts
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                # Use rich exceptions
                raise <RelevantError>(
                    f"Decorator failed: {exc}",
                    operation=func.__name__,
                    cause=exc,
                )

        return wrapper  # type: ignore

    return decorator
```

---

### Task 4.2: Refactor `@retry` Decorator

- [ ] Use `RetryOptions` model
- [ ] Implement quad API
- [ ] Use `coerce_model()` for conversion
- [ ] Update exception to use quad API
- [ ] Add comprehensive tests

**Implementation** (`src/flext_core/decorators.py`):

```python
@overload
def retry(options: Automation.RetryOptions) -> Callable[[F], F]: ...


@overload
def retry(options: dict[str, Any]) -> Callable[[F], F]: ...


@overload
def retry(**kwargs: Any) -> Callable[[F], F]: ...


def retry(
    options: Automation.RetryOptions | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[F], F]:
    """
    Retry decorator with exponential backoff (quad API).

    Args:
        options: RetryOptions model or dict
        **kwargs: Retry configuration as kwargs

    Returns:
        Decorated function with retry behavior

    Examples:
        >>> @retry(Automation.RetryOptions(max_attempts=5))
        ... def flaky_function(): ...
        >>>
        >>> @retry({"max_attempts": 5, "backoff_factor": 2.0})
        ... def flaky_function(): ...
        >>>
        >>> @retry(max_attempts=5, backoff_factor=2.0)
        ... def flaky_function(): ...
    """
    opts = coerce_model(Automation.RetryOptions, options, **kwargs)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            delay = opts.initial_delay

            for attempt in range(1, opts.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    # Check if should retry
                    if opts.exclude_exceptions:
                        if any(isinstance(exc, exc_type) for exc_type in opts.exclude_exceptions):
                            raise

                    if opts.retry_on:
                        if not any(isinstance(exc, exc_type) for exc_type in opts.retry_on):
                            raise

                    # Last attempt - don't sleep
                    if attempt == opts.max_attempts:
                        break

                    # Sleep with exponential backoff
                    sleep_time = min(delay, opts.max_delay)
                    if opts.jitter:
                        import random
                        sleep_time *= random.uniform(0.5, 1.5)

                    time.sleep(sleep_time)
                    delay *= opts.backoff_factor

            # All retries exhausted
            raise RetryExhaustedError(
                f"Failed after {opts.max_attempts} attempts",
                operation=func.__name__,
                metadata={
                    "max_attempts": opts.max_attempts,
                    "last_error": str(last_exception),
                },
                cause=last_exception,
            )

        return wrapper  # type: ignore

    return decorator
```

---

### Task 4.3: Refactor Remaining Decorators

Apply same pattern to:

- [ ] `@cache` → `CacheOptions`
- [ ] `@timeout` → `TimeoutOptions`
- [ ] `@rate_limit` → `RateLimitOptions`
- [ ] `@circuit_breaker` → `CircuitBreakerOptions`
- [ ] `@bulkhead` → `BulkheadOptions`
- [ ] `@fallback` → `FallbackOptions`
- [ ] `@validate` → `ValidationOptions`

**For Each Decorator**:
1. Add overloads (3 overload signatures)
2. Use `coerce_model()` for coercion
3. Update exceptions to use quad API
4. Add tests for all 4 styles
5. Update docstrings with examples

---

### Task 4.4: Consolidate Decorator Utilities

- [ ] Create `_decorators_utils.py` for shared logic
- [ ] Extract common patterns:
  - Retry logic with backoff calculation
  - Cache key generation
  - Timeout enforcement
- [ ] Reduce duplication between decorators

**Shared Utilities**:

```python
# _decorators_utils.py

def calculate_backoff_delay(
    attempt: int,
    initial_delay: float,
    backoff_factor: float,
    max_delay: float,
    jitter: bool = False,
) -> float:
    """Calculate exponential backoff delay."""
    delay = initial_delay * (backoff_factor ** (attempt - 1))
    delay = min(delay, max_delay)

    if jitter:
        import random
        delay *= random.uniform(0.5, 1.5)

    return delay


def generate_cache_key(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    key_func: Callable[..., str] | None = None,
) -> str:
    """Generate cache key from function and arguments."""
    if key_func:
        return key_func(*args, **kwargs)

    # Default: hash of (func_name, args, kwargs)
    import hashlib
    import json

    key_data = {
        "func": func.__name__,
        "args": str(args),
        "kwargs": str(sorted(kwargs.items())),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]
```

---

### Task 4.5: Comprehensive Testing

- [ ] Create `tests/unit/test_decorators_quad_api.py`
- [ ] Test quad API for ALL decorators
- [ ] Test integration (combining decorators)
- [ ] Test edge cases (timeouts, retries exhausted, etc.)

**Test Template**:

```python
class TestRetryDecoratorQuadAPI:
    """Test retry decorator with quad API."""

    def test_with_model(self):
        opts = Automation.RetryOptions(max_attempts=3)
        call_count = 0

        @retry(opts)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Flaky")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count == 2

    def test_with_dict(self):
        @retry({"max_attempts": 3, "initial_delay": 0.01})
        def flaky_func():
            return "success"

        assert flaky_func() == "success"

    def test_with_kwargs(self):
        @retry(max_attempts=3, initial_delay=0.01)
        def flaky_func():
            return "success"

        assert flaky_func() == "success"

    def test_retry_exhausted_exception(self):
        @retry(max_attempts=2, initial_delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            always_fails()

        exc = exc_info.value
        assert exc.context.operation == "always_fails"
        assert exc.context.metadata["max_attempts"] == 2


# Repeat for all decorators...
```

---

### Task 4.6: Update Documentation

- [ ] Update decorator usage guide
- [ ] Add quad API examples for each decorator
- [ ] Document combining decorators
- [ ] Add migration guide from old signatures

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] 7-8 decorators refactored to quad API
- [ ] All decorators use `coerce_model()`
- [ ] Shared utilities extracted
- [ ] Comprehensive tests (4 styles × 8 decorators = 32+ tests)
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Lint passes
- [ ] No circular dependencies
- [ ] Coverage maintained (≥79%)
- [ ] Documentation updated
- [ ] PR: `feat(core): unify decorators with quad API`

### Validation

```bash
cd /home/marlonsc/flext/flext-core

make lint
make type-check
make test
python scripts/detect_cycles.py

# Check LOC reduction
cloc src/ > docs/metrics/phase4_loc.txt
diff docs/metrics/phase3_loc.txt docs/metrics/phase4_loc.txt
# Expected: -100 to -150 LOC
```

---

## 📊 SUCCESS METRICS

### Quantitative
- 7-8 decorators support quad API
- 32+ test cases (4 styles × 8 decorators)
- LOC reduced by 100-150
- 0 circular dependencies
- Coverage ≥79%

### Qualitative
- Consistent decorator API
- Easy to add new decorators
- Clear pattern for contributors

---

## 🔗 DEPENDENCIES

### Requires
- **EPIC-01**: `coerce_model()`
- **EPIC-02**: Options models
- **EPIC-03**: Exception quad API

### Blocks
- **EPIC-05**: Dispatcher uses decorators internally
- **EPIC-07**: Ecosystem validation needs stable decorators

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes in decorator API | High | Maintain backward compat, deprecate old signatures |
| Performance regression | Medium | Profile decorators, optimize hot paths |
| Complexity overwhelms users | Low | Clear docs, simple examples first |

---

**Prev**: [EPIC-03: Exceptions Quad API](./EPIC-03-exceptions-quad.md)
**Next**: [EPIC-05: Dispatcher Reduction](./EPIC-05-dispatcher-reduction.md)

**Status**: 🟡 WAITING ON EPIC-03
