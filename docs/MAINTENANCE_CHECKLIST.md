# Maintenance Checklist

**Last Updated**: 2025-01-10  
**Purpose**: Ensure documentation accuracy and code quality

## Daily Checks

### Before Committing Code

- [ ] Run `make lint` - Must pass with zero errors
- [ ] Run `make type-check` - Target: zero errors
- [ ] Run `make test` - Must maintain 75%+ coverage
- [ ] Update relevant documentation if API changes
- [ ] Verify all examples still work

### Documentation Updates

- [ ] Test all code examples in changed files
- [ ] Verify all imports are valid
- [ ] Update "Last Updated" dates
- [ ] Check cross-references still work
- [ ] No unverified claims or metrics

## Weekly Maintenance

### Every Monday

- [ ] Review open issues and PRs
- [ ] Check documentation for outdated information
- [ ] Run full test suite: `make validate`
- [ ] Update TODO.md with progress
- [ ] Review and update dependencies if needed

### Code Quality Check

```bash
# Full quality check
make validate

# Individual checks
make lint        # Linting
make type-check  # Type checking
make test        # Tests with coverage
make security    # Security scan
```

## Monthly Review

### First Monday of Month

- [ ] Full documentation audit
- [ ] Test all examples in documentation
- [ ] Update version numbers if releasing
- [ ] Review and update CHANGELOG.md
- [ ] Check all external links
- [ ] Review deprecation warnings
- [ ] Update compatibility matrix

### Performance Review

- [ ] Benchmark core operations
- [ ] Profile memory usage
- [ ] Check import times
- [ ] Review test execution times

## Pre-Release Checklist

### Version Release Preparation

- [ ] All tests passing (75%+ coverage)
- [ ] Zero lint errors
- [ ] MyPy errors documented or fixed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Migration guide updated if breaking changes

### Final Validation

```bash
# Complete validation before release
make clean
make setup
make validate
make build

# Test installation
pip install dist/*.whl
python -c "from flext_core import FlextResult; print('✅ Import successful')"
```

## Documentation Standards

### Required for Code Examples

Every code example must:

- [ ] Import from correct modules
- [ ] Run without errors
- [ ] Produce expected output
- [ ] Use current API
- [ ] Include error handling

### Forbidden Without Evidence

Never claim:

- "100% complete" without verification
- Specific percentages without measurement
- "Production ready" without deployment proof
- "Zero errors" without test confirmation
- Project counts without listing them

### How to Verify Claims

```bash
# Count Python modules
find src/flext_core -name "*.py" -type f | wc -l

# Check test coverage
pytest --cov=src/flext_core --cov-report=term

# Count MyPy errors
mypy src/flext_core --strict 2>&1 | grep -c "error:"

# Count lint issues
ruff check src/flext_core 2>&1 | grep -c "error"
```

## Common Issues

### Import Errors in Examples

```python
# Always test imports
from flext_core import FlextResult  # Must work
from flext_core import FlextContainer  # Must work
from flext_core import FlextConfig  # Must work
```

### Outdated API Usage

```python
# Check for deprecated patterns
# OLD: result.is_success (might be deprecated)
# NEW: result.success (current standard)
```

### Version Mismatches

```bash
# Verify version consistency
grep version pyproject.toml
grep __version__ src/flext_core/__version__.py
grep Version docs/*/
```

## Quality Metrics

### Current Targets

| Metric              | Target | Current | Status |
| ------------------- | ------ | ------- | ------ |
| Test Coverage       | 75%+   | 75%     | ✅     |
| Lint Errors         | 0      | 0       | ✅     |
| MyPy Errors (src)   | 0      | 4       | 🚧     |
| MyPy Errors (tests) | <100   | 1,245   | ❌     |
| Documentation       | 100%   | 100%    | ✅     |

### Tracking Progress

```bash
# Generate metrics report
echo "=== FLEXT Core Metrics ==="
echo "Test Coverage: $(pytest --cov=src/flext_core --cov-report=term | grep TOTAL | awk '{print $4}')"
echo "Lint Errors: $(ruff check src/flext_core 2>&1 | grep -c error || echo 0)"
echo "MyPy Errors: $(mypy src/flext_core 2>&1 | grep -c error || echo 0)"
echo "Python Files: $(find src/flext_core -name '*.py' | wc -l)"
echo "Test Files: $(find tests -name 'test_*.py' | wc -l)"
```

## Automation Tools

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks:
      - id: mypy
```

### CI/CD Integration

Ensure CI runs:

- `make validate` on every PR
- `make test` with coverage reporting
- `make build` for release branches
- Documentation build and validation

## Emergency Procedures

### If Build Breaks

1. Check recent commits: `git log --oneline -10`
2. Run diagnostics: `make doctor`
3. Verify environment: `python --version && pip list`
4. Clean and rebuild: `make clean && make setup`
5. Check for dependency conflicts

### If Documentation Is Wrong

1. Identify incorrect information
2. Test the correct behavior
3. Update documentation immediately
4. Add test to prevent regression
5. Note in CHANGELOG.md

## Team Responsibilities

### Code Owners

- Review all PRs affecting their modules
- Maintain documentation for owned code
- Ensure test coverage remains high
- Address issues promptly

### Documentation Team

- Weekly documentation reviews
- Maintain examples and guides
- Update migration documentation
- Respond to user feedback

### Release Manager

- Coordinate version releases
- Update CHANGELOG.md
- Ensure all checks pass
- Create release tags
- Publish to PyPI

---

**Note**: This checklist is a living document. Update it as processes improve.
