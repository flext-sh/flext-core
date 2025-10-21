# Part 4: Test Fixes (Zero Tolerance)

**Status**: ✅ **UNBLOCKED - Tests can now run** (Fix 2.0 was false alarm)
**Priority**: 🔴 CRITICAL (Test failures block make validate)
**Estimated Time**: 1-2 hours (3 known fixes + investigation)
**Impact**: 100% test pass rate (92+ failures to investigate)

**NOTE**: Fix 2.0 was based on incorrect analysis. The DEFAULT_ENABLE_CACHING constant already exists. Tests are not blocked by missing constants.

**Related**:
- Part 1 context: [01-executive-summary.md](./01-executive-summary.md) (test failure analysis)
- **MUST DO FIRST**: Part 2 Fix 2.0: [02-immediate-fixes.md](./02-immediate-fixes.md#fix-20-add-missing-default_enable_caching-constant)
- Pydantic v2 patterns: [03-best-practices.md](./03-best-practices.md)
- Reference: `/home/marlonsc/flext/docs/references/pydantic2/concepts/validators.md`

**Workspace Integration**:
- FLEXT CLAUDE.md: `/home/marlonsc/flext/CLAUDE.md` (quality gates section)
- flext-core CLAUDE.md: `/home/marlonsc/flext/flext-core/CLAUDE.md` (test standards)

---

## Overview

**3 Identified Pyrefly Type Errors** (From initial analysis):
1. Frozen model immutability test - wrong exception type
2. Bus handler type error - protocol mismatch
3. Type checker test error - wrong argument types

**~89 Unknown Test Failures** (Need investigation):
- Likely related to Pydantic v2 behavior changes
- May involve validation, serialization, or model changes
- Actual count: 92 failing tests (out of 1,235 total, 92.7% pass rate)

**Target Outcome**: 100% test pass rate (total test count to be determined)

---

## Fix 4.1: Frozen Model Test Error

**Location**: `tests/unit/test_coverage_models.py:59`
**Error**: `ERROR Cannot set field 'x' [read-only]`
**Est. Time**: 15 minutes

### Problem

**Current Code**:
```python
def test_value_object_immutability(self) -> None:
    class Point(FlextModels.Value):
        x: float
        y: float
    
    point = Point(x=1.0, y=2.0)
    
    # ❌ WRONG - Pydantic v2 raises AttributeError, not ValidationError
    with pytest.raises(ValidationError):
        point.x = 3.0
```

**Pydantic v2 Behavior**: Frozen models raise `AttributeError` (with "frozen" in message), NOT `ValidationError`

### Solution

```python
import pytest
from pydantic import ValidationError

def test_value_object_immutability(self) -> None:
    """Test value objects are immutable (frozen)."""
    
    class Point(FlextModels.Value):
        x: float
        y: float
    
    point = Point(x=1.0, y=2.0)
    
    # ✅ CORRECT - Pydantic v2 frozen models raise AttributeError
    with pytest.raises(AttributeError, match="frozen"):
        point.x = 3.0
```

**Alternative** (more permissive):
```python
# Accept both AttributeError and ValidationError
with pytest.raises((AttributeError, ValidationError)):
    point.x = 3.0
```

### Verification

```bash
cd flext-core
PYTHONPATH=src poetry run pytest tests/unit/test_coverage_models.py::TestFlextModelsCoverageMissingEnhanced::test_value_object_immutability -xvs
# Expected: PASSED
```

---

## Fix 4.2: Bus Handler Type Error

**Location**: `tests/unit/test_bus.py:1244`
**Error**: `Argument 'TestEventHandler()' not assignable to handler parameter`
**Est. Time**: 30 minutes

### Investigation Required

**Step 1**: Check FlextBus.subscribe() signature
```bash
cd flext-core
grep -A10 "def subscribe" src/flext_core/bus.py
```

**Step 2**: Check TestEventHandler implementation  
```bash
grep -B5 -A15 "class TestEventHandler" tests/unit/test_bus.py
```

### Likely Issue

Handler class doesn't implement callable protocol expected by `subscribe()`.

### Solution Option A: Add __call__ Method

```python
class TestEventHandler:
    """Test event handler."""
    
    def __call__(self, event: object) -> object:  # ✅ Make callable
        """Handle event."""
        return {"status": "handled", "event": event}
```

### Solution Option B: Use Function Instead

```python
def test_event_handler(event: object) -> object:  # ✅ Function is callable
    """Handle test event."""
    return {"status": "handled", "event": event}

# In test:
result = bus.subscribe("TestEvent", test_event_handler)
```

### Verification

```bash
cd flext-core
PYTHONPATH=src poetry run pytest tests/unit/test_bus.py::TestFlextBusMissingCoverage::test_event_publishing_functionality -xvs
# Expected: PASSED
```

---

## Fix 4.3: Type Checker Test Error

**Location**: `tests/unit/test_coverage_utilities.py:376`
**Error**: `Argument 'tuple[object, ...]' not assignable to 'tuple[str | type, ...]'`
**Est. Time**: 10 minutes

### Problem

Test passes tuple of `object` instances, but function expects tuple of types.

**Current Code** (approximate):
```python
accepted = (object, object, object)  # ❌ WRONG - instances, not types
FlextUtilities.TypeChecker.can_handle_message_type(accepted, str)
```

### Solution

```python
# ✅ CORRECT - Pass actual types
accepted = (str, int, float)  # tuple[type, type, type]
assert FlextUtilities.TypeChecker.can_handle_message_type(accepted, str) is True

# For negative test:
accepted = (int, float, bool)  # tuple[type, type, type]
assert FlextUtilities.TypeChecker.can_handle_message_type(accepted, str) is False
```

### Verification

```bash
cd flext-core  
PYTHONPATH=src poetry run pytest tests/unit/test_coverage_utilities.py::test_type_checker -xvs
# Expected: PASSED
```

---

## Investigation: Remaining 89 Test Failures

**Status**: REQUIRES INVESTIGATION
**Est. Time**: 1-2 hours

### Step 1: Get Failure Summary

```bash
cd flext-core
PYTHONPATH=src poetry run pytest tests/ --tb=no -q | grep -E "FAILED|ERROR"
# Expected: List of all failing tests
```

### Step 2: Categorize Failures

Create failure report:
```bash
PYTHONPATH=src poetry run pytest tests/ --tb=line -q > test_failures_report.txt 2>&1
```

Analyze patterns:
- Import errors?
- Assertion failures?
- Type errors?
- Fixture issues?

### Step 3: Fix by Category

**Common Patterns**:

1. **Import Errors** - Already fixed in Part 2 (test infrastructure)
2. **Pydantic v2 Behavior Changes** - Like frozen model test
3. **Type Annotation Issues** - Like type checker test
4. **Fixture Dependencies** - Update fixture implementations
5. **API Changes** - Update to new flext-core APIs

### Step 4: Systematic Fix Process

For each failing test:
1. Run test in isolation: `pytest tests/unit/test_X.py::test_Y -xvs`
2. Read error message carefully
3. Check if related to Pydantic v2 changes
4. Apply appropriate fix
5. Verify test passes
6. Move to next test

---

## Test Fix Checklist

### Critical Fixes (Known)
- [ ] Fix frozen model test (test_coverage_models.py:59)
- [ ] Fix bus handler test (test_bus.py:1244)
- [ ] Fix type checker test (test_coverage_utilities.py:376)

### Investigation & Resolution
- [ ] Get complete list of 92 failures
- [ ] Categorize by failure type
- [ ] Create fix plan for each category
- [ ] Fix import errors (should be 0 after Part 2)
- [ ] Fix Pydantic v2 behavior mismatches
- [ ] Fix type annotation issues
- [ ] Fix fixture dependencies
- [ ] Fix API usage errors

### Verification
- [ ] Run full test suite: `make test`
- [ ] Verify 100% pass rate: 1235+ passing, 0 failing
- [ ] Check coverage: 79%+ maintained
- [ ] Run with verbose: `pytest -xvs` to see details

---

## Success Criteria

After completing Part 4:
- ✅ **100% test pass rate** (1235+ passing, 0 failing)
- ✅ **3 known errors fixed** (frozen model, bus handler, type checker)
- ✅ **89 unknown failures resolved**
- ✅ **Coverage maintained** (79%+ for flext-core)
- ✅ **make validate passes** (all quality gates)

---

## Common Pydantic v2 Test Fixes

### Pattern 1: Frozen Model Exceptions

**OLD (v1)**:
```python
with pytest.raises(ValidationError):
    frozen_model.field = new_value
```

**NEW (v2)**:
```python
with pytest.raises(AttributeError, match="frozen"):
    frozen_model.field = new_value
```

### Pattern 2: Model Serialization

**OLD (v1)**:
```python
assert model.dict() == expected_dict
assert model.json() == expected_json
```

**NEW (v2)**:
```python
assert model.model_dump() == expected_dict
assert model.model_dump_json() == expected_json
```

### Pattern 3: Model Validation

**OLD (v1)**:
```python
model = MyModel.parse_obj(data)
model = MyModel.parse_raw(json_str)
```

**NEW (v2)**:
```python
model = MyModel.model_validate(data)
model = MyModel.model_validate_json(json_str)
```

### Pattern 4: Validation Errors

**OLD (v1)**:
```python
try:
    model = MyModel(**data)
except ValidationError as e:
    errors = e.errors()  # Returns list[dict]
```

**NEW (v2)** (same, but error format may differ):
```python
try:
    model = MyModel(**data)
except ValidationError as e:
    errors = e.errors()  # Returns list[ErrorDict] (TypedDict)
    # Error structure may have changed slightly
```

---

## Next Steps

After completing Part 4:
1. ✅ Verify 100% test pass rate
2. ✅ Run full quality gates: `make validate`
3. ➡️ Proceed to Part 5: [Workspace Audit](./05-workspace-audit.md)

---

**Time Estimate**: 1-2 hours (3 known fixes + investigation + resolution)
