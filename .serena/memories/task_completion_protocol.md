# FLEXT-API Task Completion Protocol

## Mandatory Steps After object Code Changes

### 1. Quality Gates Execution (ZERO TOLERANCE)

```bash
# PHASE 1: Code Quality (MANDATORY)
make lint              # Ruff: ZERO violations in src/
make type-check        # MyPy strict: ZERO errors in src/
make security          # Bandit: ZERO critical vulnerabilities

# PHASE 2: Testing (85% Coverage Required)
make test              # Full test suite with coverage validation

# PHASE 3: Complete Validation
make validate          # All quality gates together
```

### 2. FLEXT Ecosystem Compliance

- **FlextResult Usage**: All operations must return FlextResult[T]
- **flext-core Integration**: Must use foundation library exclusively
- **Type Safety**: Complete type coverage with MyPy strict mode
- **Clean Architecture**: Maintain domain/application/infrastructure separation

### 3. Error Resolution Priority

1. **Type Errors**: Fix all MyPy strict mode errors first
2. **Lint Issues**: Resolve all Ruff violations
3. **Security Issues**: Address all Bandit findings
4. **Test Failures**: Ensure all tests pass with 85%+ coverage

### 4. Pre-Commit Validation

```bash
# Before any commit:
make validate          # Must pass completely
make pre-commit        # Run pre-commit hooks
```

### 5. Documentation Updates

- Update docstrings for modified classes/methods
- Ensure type hints are complete and accurate
- Verify examples in docstrings work correctly

### 6. Integration Testing

- Verify FLEXT ecosystem integration works
- Test HTTP client functionality
- Validate FastAPI application creation
- Confirm plugin system works correctly

## Success Criteria

- ✅ **Zero MyPy Errors**: All type checking passes
- ✅ **Zero Ruff Violations**: Code style compliant
- ✅ **Zero Security Issues**: No critical Bandit findings
- ✅ **85%+ Test Coverage**: Basic test validation
- ✅ **All Tests Pass**: No failing tests
- ✅ **FLEXT Compliance**: Proper ecosystem integration

## Failure Response

If any quality gate fails:

1. **STOP**: Do not proceed with other changes
2. **INVESTIGATE**: Understand root cause
3. **FIX**: Address the specific issue
4. **VALIDATE**: Re-run quality gates
5. **REPEAT**: Until all gates pass
