# EPIC-05: Dispatcher & Services - Redução Pesada

**Phase**: 5
**Duration**: Week 4-5 (8-10 days)
**Risk**: 🔴 High (core component)
**LOC Impact**: -200 to -300
**Dependencies**: ALL previous epics
**Status**: 🟡 WAITING ON EPIC-04

---

## 🎯 OBJECTIVE

**HEAVY CODE REDUCTION** in dispatcher and services via:
1. Convert inner classes to Pydantic models
2. Consolidate validation/extraction helpers
3. Replace multi-parameter methods with config objects
4. Use `FlextMixins.ensure_result()` everywhere

**Why This Matters**: Dispatcher is ~1200 LOC. This is where we get -200 to -300 LOC reduction.

---

## ⚠️ RISK NOTICE

**This is the HIGHEST RISK phase**:
- Dispatcher is central to everything
- Mistakes here break all flows
- Must be incremental and heavily tested

**Mitigation Strategy**:
- Refactor ONE helper at a time
- Run full test suite after EACH change
- Keep feature flags for rollback
- Validate in 4+ satellite projects

---

## 📋 TASKS CHECKLIST

### Task 5.1: Identify Refactoring Targets

- [ ] Analyze dispatcher.py for:
  - Inner classes with manual `__init__`
  - Duplicated isinstance checks
  - Methods with 6+ parameters
  - Repeated validation patterns
- [ ] Create `docs/metrics/dispatcher_refactoring_plan.md`
- [ ] Prioritize by impact/risk ratio

**Analysis Template**:

```markdown
# Dispatcher Refactoring Analysis

## Inner Classes (Convert to Pydantic)

| Class | LOC | Parameters | Complexity | Priority |
|-------|-----|------------|------------|----------|
| CircuitBreakerManager | 150 | 8 | High | 1 |
| TimeoutEnforcer | 80 | 5 | Medium | 2 |
| RateLimiterManager | 120 | 6 | High | 1 |
| RetryPolicy | 100 | 7 | Medium | 2 |

## Validation Helpers (Consolidate)

| Pattern | Occurrences | Target Helper | Priority |
|---------|-------------|---------------|----------|
| `_validate_handler_mode()` | 3 places | Single method | 1 |
| `_extract_handler_name()` | 5 places | Generic `_extract_field()` | 1 |
| `isinstance(*, BaseModel)` | 12 places | `FlextMixins.to_dict()` | 1 |

## Config Objects (Replace Multi-Param Methods)

| Method | Parameters | Target Config | LOC Savings |
|--------|------------|---------------|-------------|
| `_execute_dispatch_attempt()` | 7 | `DispatchAttemptConfig` | ~30 |
| `_validate_handler_registration()` | 6 | `HandlerValidationConfig` | ~20 |
```

---

### Task 5.2: Convert Inner Classes to Pydantic

#### Priority 1: CircuitBreakerManager

- [ ] Convert to `BaseModel` with `PrivateAttr` for state
- [ ] Replace manual validation with Field() constraints
- [ ] Add tests for model validation
- [ ] Migrate usage in dispatcher

**Before**:

```python
class CircuitBreakerManager:
    def __init__(
        self,
        failure_threshold: int,
        success_threshold: int,
        timeout: float,
        half_open_max_calls: int = 1,
    ):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if timeout <= 0:
            raise ValueError("timeout must be > 0")

        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self._failure_count = 0
        self._success_count = 0
        self._state = "closed"
        self._opened_at: float | None = None
```

**After**:

```python
from pydantic import BaseModel, Field, PrivateAttr

class CircuitBreakerManager(BaseModel):
    """Circuit breaker state manager."""

    model_config = ConfigDict(frozen=False)

    failure_threshold: int = Field(ge=1)
    success_threshold: int = Field(ge=1)
    timeout: float = Field(gt=0.0)
    half_open_max_calls: int = Field(default=1, ge=1)

    # State (private attrs)
    _failure_count: int = PrivateAttr(default=0)
    _success_count: int = PrivateAttr(default=0)
    _state: str = PrivateAttr(default="closed")
    _opened_at: float | None = PrivateAttr(default=None)

    # Methods stay the same
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    def is_open(self) -> bool: ...
```

**Benefits**:
- Auto-validation (no manual if checks)
- `model_dump()` for debugging
- Type safety with mypy
- ~20-30 LOC saved per class

**Repeat for**: TimeoutEnforcer, RateLimiterManager, RetryPolicy

---

### Task 5.3: Consolidate Validation Helpers

#### Create Generic `_extract_field()` Helper

- [ ] Replace specific extractors with generic version
- [ ] Use patterns for common extractions
- [ ] Maintain backward compatibility

**Before** (duplicated pattern):

```python
def _extract_handler_name(self, obj: Any) -> str:
    # Pattern 1: attribute
    if hasattr(obj, "handler_name"):
        return str(obj.handler_name)
    # Pattern 2: callable name
    if callable(obj):
        return obj.__name__
    # Pattern 3: dict key
    if isinstance(obj, dict):
        return obj.get("handler_name", "")
    return ""

def _extract_timeout(self, obj: Any) -> float:
    # Same pattern, different field
    if hasattr(obj, "timeout"):
        return float(obj.timeout)
    if isinstance(obj, dict):
        return float(obj.get("timeout", 30.0))
    return 30.0

# ... repeated for 5+ fields
```

**After** (single generic helper):

```python
def _extract_field[T](
    self,
    obj: Any,
    field_name: str,
    default: T,
    converter: Callable[[Any], T] | None = None,
) -> T:
    """
    Generic field extraction from object/dict/callable.

    Args:
        obj: Source object
        field_name: Field to extract
        default: Default value if not found
        converter: Optional converter function

    Returns:
        Extracted value or default

    Examples:
        >>> name = self._extract_field(obj, "handler_name", "", str)
        >>> timeout = self._extract_field(obj, "timeout", 30.0, float)
    """
    value: Any = None

    # Pattern 1: Attribute
    if hasattr(obj, field_name):
        value = getattr(obj, field_name)

    # Pattern 2: Dict key
    elif isinstance(obj, dict):
        value = obj.get(field_name)

    # Pattern 3: Callable name (special case)
    elif field_name == "handler_name" and callable(obj):
        value = obj.__name__

    # Not found
    if value is None:
        return default

    # Convert if converter provided
    if converter:
        try:
            return converter(value)
        except (ValueError, TypeError):
            return default

    return value


# Usage
handler_name = self._extract_field(obj, "handler_name", "", str)
timeout = self._extract_field(obj, "timeout", 30.0, float)
operation_id = self._extract_field(obj, "operation_id", "", str)
```

**LOC Savings**: ~50-80 (5 specific methods → 1 generic)

---

#### Consolidate `_validate_handler_mode()`

- [ ] Single reusable validation method
- [ ] Used in both registration points

**Before**:

```python
# In register_handler
def register_handler(...):
    if mode not in ("sync", "async"):
        raise ValueError(f"Invalid mode: {mode}")

# In register_handler_with_request
def register_handler_with_request(...):
    if mode not in ("sync", "async"):
        raise ValueError(f"Invalid mode: {mode}")
```

**After**:

```python
VALID_HANDLER_MODES = {"sync", "async"}

def _validate_handler_mode(self, mode: str) -> None:
    """
    Validate handler execution mode.

    Args:
        mode: Handler mode to validate

    Raises:
        ValidationError: If mode is invalid
    """
    if mode not in VALID_HANDLER_MODES:
        raise ValidationError(
            f"Invalid handler mode: {mode}",
            operation="validate_handler_mode",
            metadata={"mode": mode, "valid_modes": list(VALID_HANDLER_MODES)},
        )

# Usage in both places
def register_handler(...):
    self._validate_handler_mode(mode)

def register_handler_with_request(...):
    self._validate_handler_mode(mode)
```

---

### Task 5.4: Create Config Objects

#### DispatchAttemptConfig

- [ ] Create model for `_execute_dispatch_attempt()`
- [ ] Replace 7-parameter signature
- [ ] Update callers

**Before**:

```python
def _execute_dispatch_attempt(
    self,
    handler_name: str,
    request: Any,
    timeout: float,
    mode: str,
    correlation_id: str,
    metadata: dict[str, Any],
    operation_id: str,
) -> FlextResult[Any]:
    # 100+ LOC of implementation
```

**After**:

```python
class DispatchAttemptConfig(BaseModel):
    """Configuration for a single dispatch attempt."""

    model_config = ConfigDict(frozen=False)

    handler_name: str = Field(min_length=1)
    request: Any
    timeout: float = Field(default=30.0, gt=0.0)
    mode: Literal["sync", "async"] = Field(default="async")
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)
    operation_id: str = Field(default="")


def _execute_dispatch_attempt(
    self,
    config: DispatchAttemptConfig,
) -> FlextResult[Any]:
    # Same implementation, but access via config.handler_name, config.request, etc.
```

**Callers**:

```python
# Before
result = self._execute_dispatch_attempt(
    handler_name,
    request,
    timeout,
    mode,
    correlation_id,
    metadata,
    operation_id,
)

# After
config = DispatchAttemptConfig(
    handler_name=handler_name,
    request=request,
    timeout=timeout,
    # ... or use quad API
)
result = self._execute_dispatch_attempt(config)
```

**Repeat for**: HandlerValidationConfig, HandlerRegistrationConfig

---

### Task 5.5: Use FlextMixins Everywhere

- [ ] Replace all `if not isinstance(result, FlextResult)` with `ensure_result()`
- [ ] Replace all model↔dict conversions with `to_dict()` / `from_dict()`
- [ ] Count occurrences before/after

**Search & Replace**:

```bash
# Find patterns to replace
grep -rn "if not isinstance.*FlextResult" src/flext_core/dispatcher.py
grep -rn "if isinstance.*BaseModel.*model_dump" src/flext_core/dispatcher.py

# Manual replacement (IDE refactoring recommended)
```

**Example**:

```python
# Before
if not isinstance(result, FlextResult):
    result = self.ok(result)

# After
result = FlextMixins.ResultHandling.ensure_result(result)
```

---

### Task 5.6: Incremental Testing & Validation

**CRITICAL: After EACH subtask above**:

- [ ] Run `make test`
- [ ] Run `make lint`
- [ ] Run `make type-check`
- [ ] Run dispatcher-specific tests
- [ ] Verify no regressions

**Test Strategy**:

```bash
# After each change
cd /home/marlonsc/flext/flext-core

# Run specific dispatcher tests
pytest tests/unit/test_dispatcher*.py -v

# Run integration tests
pytest tests/integration/ -k dispatcher -v

# Full suite
make test
```

---

### Task 5.7: Services Refactoring (Similar Pattern)

Apply same patterns to `services.py`:

- [ ] Convert inner classes to Pydantic
- [ ] Use config objects for multi-param methods
- [ ] Use FlextMixins helpers

**Lower Priority** (services.py less critical than dispatcher.py)

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] ALL inner classes converted to Pydantic
- [ ] ALL validation helpers consolidated
- [ ] 3+ config objects created and used
- [ ] `ensure_result()` used everywhere
- [ ] LOC reduced by 200-300
- [ ] ALL tests pass (unit + integration)
- [ ] Type checking passes
- [ ] Lint passes
- [ ] No circular dependencies
- [ ] Validated in 4+ satellite projects
- [ ] Coverage maintained (≥79%)
- [ ] PR: `refactor(core): reduce dispatcher complexity`

### Validation Checklist

```bash
cd /home/marlonsc/flext/flext-core

# Quality gates
make lint
make type-check
make test
python scripts/detect_cycles.py

# LOC reduction
cloc src/flext_core/dispatcher.py src/flext_core/services.py > docs/metrics/phase5_loc.txt
diff docs/metrics/phase4_loc.txt docs/metrics/phase5_loc.txt
# Expected: -200 to -300 LOC

# Dispatcher-specific validation
pytest tests/unit/test_dispatcher*.py -v --cov=src/flext_core/dispatcher
pytest tests/integration/ -k dispatcher -v

# Satellite project validation (CRITICAL)
cd ../flext-cli && make test
cd ../flext-ldap && make test
cd ../flext-ldif && make test
cd ../algar-oud-mig && make test
```

---

## 📊 SUCCESS METRICS

### Quantitative
- Dispatcher LOC: ~1200 → ~900 (-300)
- Services LOC: ~1500 → ~1400 (-100)
- Inner classes: All converted (4-5 classes)
- Config objects: 3 created
- Validation helpers: Consolidated (5 → 1)
- Coverage: ≥79% maintained

### Qualitative
- Dispatcher easier to understand
- Adding new features simpler
- Debugging improved (model_dump() everywhere)
- No regressions in satellite projects

---

## 🔗 DEPENDENCIES

### Requires
- **ALL PREVIOUS EPICS** (0-4)

### Blocks
- **EPIC-06**: Dict elimination depends on stable dispatcher
- **EPIC-07**: Final validation needs stable dispatcher

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking dispatcher breaks everything | **CRITICAL** | Incremental changes, test after each step |
| Satellite projects break | High | Validate in 4+ projects before merge |
| Performance regression | Medium | Profile before/after, optimize if needed |
| Subtle bugs in edge cases | High | Comprehensive integration tests, feature flags |

### Emergency Rollback Plan

```bash
# If disaster strikes:
git revert <phase5-commit-sha>
make test  # Verify rollback works
# Communicate to team, analyze failure, retry with smaller increments
```

---

## 📝 NOTES

- **GO SLOW**: This is the riskiest phase
- **TEST EVERYTHING**: After every change
- **COMMUNICATE**: Keep team informed of progress
- **ROLLBACK READY**: Be prepared to revert if issues found
- **FEATURE FLAGS**: Consider wrapping risky changes in flags

---

**Prev**: [EPIC-04: Decorators Unified](./EPIC-04-decorators.md)
**Next**: [EPIC-06: Dict Elimination](./EPIC-06-dict-elimination.md)

**Status**: 🟡 WAITING ON EPIC-04
**Risk Level**: 🔴 HIGH - PROCEED WITH CAUTION
