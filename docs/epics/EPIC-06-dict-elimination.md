# EPIC-06: Generalização e Remoção de Dict de Negócio

**Phase**: 6
**Duration**: Week 6 (5 days)
**Risk**: 🟡 Medium
**LOC Impact**: -100 to -150
**Dependencies**: EPIC-02 (models), EPIC-05 (stable dispatcher)
**Status**: 🟡 WAITING ON EPIC-05

---

## 🎯 OBJECTIVE

1. **Eliminate ALL business dicts** found in Phase 0 baseline
2. Apply `@autoschema` decorator where beneficial
3. Convert remaining dict-based APIs to use models
4. Achieve: `python scripts/analyze_dicts.py` → **0 business dicts**

**Why This Matters**: Business logic in dicts = no validation, no type safety, no IDE support.

---

## 📋 TASKS CHECKLIST

### Task 6.1: Review Dict Classification

- [ ] Load `docs/metrics/dict_classification.md` from Phase 0
- [ ] Verify all business dicts identified
- [ ] Check if new dicts appeared (run script again)
- [ ] Create migration priority list

**Commands**:

```bash
cd /home/marlonsc/flext/flext-core

# Re-run analysis (check for new dicts since Phase 0)
python scripts/analyze_dicts.py > docs/metrics/dicts_phase6_before.txt

# Compare with baseline
diff docs/metrics/dicts_baseline.txt docs/metrics/dicts_phase6_before.txt
```

---

### Task 6.2: Migrate High-Priority Business Dicts

From Phase 0 classification, prioritize:

**Priority 1 (Public APIs)**:
- [ ] Handler registration configs
- [ ] Dispatch context objects
- [ ] Service initialization configs

**Priority 2 (Internal APIs)**:
- [ ] Internal context passing
- [ ] Internal config objects

**Priority 3 (Low-usage)**:
- [ ] Rarely used configuration
- [ ] Legacy compatibility layers

---

### Task 6.3: Apply Model Conversions

For each business dict:

#### Pattern 1: Config Dict → Model

**Before**:

```python
def initialize_service(
    name: str,
    config: dict[str, Any],
) -> FlextService:
    timeout = config.get("timeout", 30.0)
    max_retries = config.get("max_retries", 3)
    enable_cache = config.get("enable_cache", False)
    # ... validation scattered

    return FlextService(...)
```

**After**:

```python
class ServiceInitConfig(BaseModel):
    """Service initialization configuration."""

    model_config = ConfigDict(frozen=False)

    timeout: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    enable_cache: bool = Field(default=False)


def initialize_service(
    name: str,
    config: ServiceInitConfig | dict[str, Any],
) -> FlextService:
    # Coerce to model (quad API support)
    from flext_core.automation import coerce_model

    cfg = coerce_model(ServiceInitConfig, config)

    return FlextService(
        name=name,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        enable_cache=cfg.enable_cache,
    )
```

---

#### Pattern 2: Context Dict → Model

**Before**:

```python
def process_request(
    handler: Callable[..., Any],
    context: dict[str, Any],
) -> FlextResult[Any]:
    correlation_id = context.get("correlation_id", "")
    user_id = context.get("user_id")
    metadata = context.get("metadata", {})
    # ...
```

**After**:

```python
class RequestContext(BaseModel):
    """Request processing context."""

    correlation_id: str = Field(default="")
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def process_request(
    handler: Callable[..., Any],
    context: RequestContext | dict[str, Any],
) -> FlextResult[Any]:
    from flext_core.automation import coerce_model

    ctx = coerce_model(RequestContext, context)

    # Direct attribute access
    correlation_id = ctx.correlation_id
    user_id = ctx.user_id
    metadata = ctx.metadata
    # ...
```

---

#### Pattern 3: Nested Dicts → Nested Models

**Before**:

```python
config = {
    "retry": {
        "max_attempts": 3,
        "backoff": 2.0,
    },
    "cache": {
        "ttl": 300,
        "max_size": 100,
    },
    "timeout": 30.0,
}
```

**After**:

```python
from flext_core.models import Automation

class ServiceConfig(BaseModel):
    """Complete service configuration."""

    retry: Automation.RetryOptions = Field(
        default_factory=lambda: Automation.RetryOptions(),
    )
    cache: Automation.CacheOptions = Field(
        default_factory=lambda: Automation.CacheOptions(),
    )
    timeout: float = Field(default=30.0, gt=0.0)


# Usage
config = ServiceConfig(
    retry=Automation.RetryOptions(max_attempts=3, backoff_factor=2.0),
    cache=Automation.CacheOptions(ttl=300, max_size=100),
    timeout=30.0,
)

# Or from dict (quad API)
config = ServiceConfig(**dict_config)
```

---

### Task 6.4: Apply @autoschema Decorator

For services/methods with large kwargs signatures:

**Before**:

```python
def complex_operation(
    data: str,
    timeout: float = 30.0,
    max_retries: int = 3,
    enable_cache: bool = False,
    cache_ttl: float = 300.0,
    retry_backoff: float = 2.0,
    # ... 10+ parameters
) -> FlextResult[str]:
    # 100+ LOC
```

**After**:

```python
class ComplexOperationOptions(BaseModel):
    timeout: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    enable_cache: bool = False
    cache_ttl: float = Field(default=300.0, gt=0.0)
    retry_backoff: float = Field(default=2.0, ge=1.0)


from flext_core.automation import autoschema


@autoschema()
def complex_operation(
    data: str,
    options: ComplexOperationOptions,
) -> FlextResult[str]:
    # Access via options.timeout, options.max_retries, etc.
    # ...


# Callers can use any style:
complex_operation("test", ComplexOperationOptions(...))
complex_operation("test", {"timeout": 60})
complex_operation("test", timeout=60, max_retries=5)
```

---

### Task 6.5: Update Dynamic Dicts Documentation

For dicts that are **legitimately dynamic** (logging, tracing, etc.):

- [ ] Add explicit type annotation: `dict[str, Any]  # DYNAMIC: logging context`
- [ ] Document why it's dynamic in docstring
- [ ] Ensure it's truly unstructured data

**Example**:

```python
def log_with_context(
    message: str,
    extra: dict[str, Any],  # DYNAMIC: free-form logging metadata
) -> None:
    """
    Log message with additional context.

    Args:
        message: Log message
        extra: Free-form metadata for logging (DYNAMIC - no fixed structure)

    Note:
        The `extra` dict is intentionally unstructured as it represents
        arbitrary contextual data that varies per log call.
    """
    logger.info(message, extra=extra)
```

---

### Task 6.6: Run Final Dict Analysis

- [ ] Run `python scripts/analyze_dicts.py`
- [ ] Verify: **0 business dicts** remaining
- [ ] Document any remaining dynamic dicts with rationale
- [ ] Save to `docs/metrics/dicts_phase6_after.txt`

**Expected Output**:

```
✅ Found 0 business dicts
⚠️  Found 12 dynamic dicts (documented as DYNAMIC)

Dynamic dicts (allowed):
- src/flext_core/logging.py:78: extra: dict[str, Any]  # DYNAMIC: logging
- src/flext_core/tracing.py:45: span_data: dict[str, Any]  # DYNAMIC: tracing
...

All business dicts have been migrated to models!
```

---

### Task 6.7: Update Tests

- [ ] Add tests for new models
- [ ] Test quad API usage
- [ ] Test backward compatibility (if any dicts still accepted)
- [ ] Update existing tests that used dicts

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] ALL business dicts converted to models
- [ ] `@autoschema` applied to 3+ methods
- [ ] Dynamic dicts documented with rationale
- [ ] `python scripts/analyze_dicts.py` → 0 business dicts
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Lint passes
- [ ] No circular dependencies
- [ ] Coverage maintained (≥79%)
- [ ] PR: `refactor(core): eliminate business dicts`

### Validation

```bash
cd /home/marlonsc/flext/flext-core

# Quality gates
make lint
make type-check
make test
python scripts/detect_cycles.py

# Dict analysis (MUST BE 0)
python scripts/analyze_dicts.py
# Expected: ✅ 0 business dicts

# LOC check
cloc src/ > docs/metrics/phase6_loc.txt
diff docs/metrics/phase5_loc.txt docs/metrics/phase6_loc.txt
# Expected: -100 to -150 LOC
```

---

## 📊 SUCCESS METRICS

### Quantitative
- Business dicts: Baseline count → **0**
- New models created: 5-10
- `@autoschema` applications: 3-5
- LOC reduced: 100-150
- Coverage: ≥79%

### Qualitative
- Type safety improved
- IDE autocomplete works everywhere
- Validation catches errors early
- Code more maintainable

---

## 🔗 DEPENDENCIES

### Requires
- **EPIC-02**: Automation models available
- **EPIC-05**: Stable dispatcher (no churn during migration)

### Blocks
- **EPIC-07**: Final validation needs dict-free codebase

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes for callers | High | Support dict input (quad API) with deprecation warnings |
| Over-modeling (too many tiny models) | Medium | Only model structured data, keep truly dynamic dicts |
| Performance overhead | Low | Models are lightweight, profile if concerned |

---

**Prev**: [EPIC-05: Dispatcher Reduction](./EPIC-05-dispatcher-reduction.md)
**Next**: [EPIC-07: Tests/Docs/Gates](./EPIC-07-tests-docs-gates.md)

**Status**: 🟡 WAITING ON EPIC-05
