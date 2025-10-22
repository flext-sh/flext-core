# Pydantic v2 Modernization: Quick Start (Next Session)

**For**: Starting the Pydantic v2 modernization work
**Duration**: Read in 5 minutes, then execute
**Full Guide**: See [PYDANTIC_V2_EXECUTION_GUIDE.md](./PYDANTIC_V2_EXECUTION_GUIDE.md)

---

## The Problem (In 30 Seconds)

- ❌ Tests are failing (92 failures out of 1,235 tests)
- ❌ Test code expects validation methods that don't exist
- ❌ Pyrefly can't find test imports (6 import errors)
- ⚠️ Code has redundant type casts
- ⚠️ Constants.Config name is confusing

## The Solution (In 3 Phases)

### Phase 1: Critical Fixes (2-4 hours) ← START HERE

**Fix 1**: Update `pyproject.toml` to add Pyrefly test paths
```toml
[tool.pyrefly]
python_version = "3.13"
strict = true
mypy_path = ["src", "tests"]  # ← ADD THIS LINE
```

**Fix 2**: Add 3 missing validation methods to `src/flext_core/utilities.py`
- `validate_string_not_empty()`
- `validate_string_length()`
- `validate_url()`

See the full guide for exact code.

**Verify**: `pytest tests/unit/test_coverage_utilities.py::TestValidation -v` should PASS

### Phase 2: Test Suite Repair (4-6 hours)

Run full test suite and fix any remaining failures:
```bash
make test
```

**Goal**: 100% pass rate (currently 92.7%)

### Phase 3: Code Cleanup (4-6 hours)

1. Remove 9 redundant type casts from `context.py`
2. Rename `Constants.Config` → `Constants.Defaults`
3. Run `make validate` - should pass

### Phase 4: Optimization (8-12 hours - Optional)

Expand Pydantic v2 adoption:
- Add 6+ domain types to `typings.py`
- Deprecate duplicate validation methods
- Update internal code to use new types

---

## Quick Commands Reference

```bash
cd /home/marlonsc/flext/flext-core

# Check current test status
env PYTHONPATH=src poetry run pytest tests/ -q --tb=line | tail -5

# Check for import errors (Pyrefly)
env PYTHONPATH=src poetry run pyrefly check tests/ 2>&1 | grep "import-error"

# Run validation checks
make validate

# Run specific validation tests
env PYTHONPATH=src poetry run pytest tests/unit/test_coverage_utilities.py::TestValidation -v
```

---

## File Locations

**Files to Modify**:
1. `pyproject.toml` (line ~1-50): Add Pyrefly config
2. `src/flext_core/utilities.py` (line ~323-440): Add validation methods
3. `src/flext_core/context.py` (lines ~???): Remove type casts
4. `src/flext_core/constants.py` (line ~512): Rename Config → Defaults

**Files to Review** (don't modify):
- `src/flext_core/config.py`: Uses constants
- `tests/conftest.py`: Test fixtures
- `tests/unit/test_coverage_utilities.py`: Tests (shows what methods are expected)

---

## Success Criteria

**Phase 1 Complete**:
- ✅ `make type-check` shows 0 import errors
- ✅ Validation tests pass (6/6)

**Phase 2 Complete**:
- ✅ `make test` shows 100% pass rate
- ✅ Coverage >= 79%

**Phase 3 Complete**:
- ✅ `make validate` passes (lint + type-check + test + security)
- ✅ No redundant type casts
- ✅ Constants renamed

**Phase 4 Complete** (Optional):
- ✅ All optimization patterns applied
- ✅ Tests still 100% passing

---

## Key Files with Exact Line Numbers

### pyproject.toml
- Lines 1-50: Project metadata
- **INSERT AFTER**: Line 50 (add Pyrefly section)

### utilities.py
- Line 323: `class Validation:`
- Line 346-440: Existing validation methods
- **INSERT AFTER**: Line 440 (add 3 new methods)

### constants.py
- Line 512: `class Config:` ← RENAME TO `class Defaults:`
- Line 516: `DEFAULT_ENABLE_CACHING` (verify it exists)

### context.py
- Search for: `cast(` (should find 9 instances to remove)

---

## If You Get Stuck

1. **"ModuleNotFoundError: No module named 'tests'"**
   - Solution: Export PYTHONPATH=src:tests

2. **"Could not find import of 'fixtures'"**
   - Solution: Verify pyproject.toml Pyrefly config is correct

3. **Tests still failing after Phase 1**
   - Solution: Check error message with `--tb=short` flag
   - Run specific test file to isolate issue

4. **Type check errors after removing casts**
   - Solution: Read error message - may need type annotation instead

---

## Next Steps

1. **Read the full guide** for exact code and context
2. **Execute Phase 1** first (quickest wins)
3. **Verify each fix** before moving to next phase
4. **Use git commits** after each phase
5. **Run `make validate`** frequently

---

## Timeline

- **Single dev**: 1-2 weeks (10-28 hours total)
- **Two devs**: 1 week (can parallelize phases)

**Estimated**:
- Phase 1: 2-4 hours
- Phase 2: 4-6 hours
- Phase 3: 4-6 hours
- Phase 4: 8-12 hours (optional)

---

## Questions?

See the full execution guide: [PYDANTIC_V2_EXECUTION_GUIDE.md](./PYDANTIC_V2_EXECUTION_GUIDE.md)

**Created**: 2025-10-21
**Status**: Ready to execute in next session
**Difficulty**: Medium (mostly code changes, good documentation)

