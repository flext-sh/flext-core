# FLEXT-CORE QA Workflow

## Task Completion Protocol (MANDATORY)

### 1. Before object Code Change

```bash
# Always validate current state first
make check                 # Quick lint + type check
```

### 2. After object Code Change

```bash
# MANDATORY: Run after editing any file
make lint                  # Fix ALL ruff issues before continuing
make type-check            # Ensure zero type errors
```

### 3. Before Commit (ZERO TOLERANCE)

```bash
make validate              # Complete pipeline (lint + type + security + test)
```

### 4. Individual Tool Usage

```bash
# Fix specific issues
poetry run ruff check src tests --fix         # Auto-fix what's possible
poetry run ruff format src tests              # Format code
poetry run mypy src --strict                  # Type checking
poetry run pytest tests --cov=src            # Test with coverage
```

## Quality Gate Sequence

### Phase 1: Foundation Code Quality (ZERO TOLERANCE)

- Ruff: ZERO violations allowed in src/
- MyPy strict: ZERO errors in src/
- PyRight: ZERO errors in src/

### Phase 2: Test Quality

- All tests must pass
- 75%+ coverage minimum (targeting 85%+)
- Real functional tests (minimal mocking)

### Phase 3: Architecture Compliance

- Single class per module
- No helper functions outside classes
- FlextResult error handling everywhere
- No wrappers or fallbacks

## Error Resolution Priority

1. **Critical**: Type errors, import errors, syntax errors
2. **High**: Test failures, coverage drops
3. **Medium**: Lint violations, style issues
4. **Low**: Documentation, minor optimizations

## ZERO TOLERANCE VIOLATIONS

- API breaking changes without deprecation
- Type errors in src/ directory
- Test failures
- Coverage below minimum threshold
- Direct internal imports bypassing **init**.py
