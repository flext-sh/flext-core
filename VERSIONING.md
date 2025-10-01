# FLEXT-Core Semantic Versioning Strategy

**Version**: 1.0.0 | **Date**: 2025-10-01 | **Status**: STABLE

## Overview

FLEXT-Core follows strict [Semantic Versioning 2.0.0](https://semver.org/) to provide API stability guarantees for the entire FLEXT ecosystem (32+ dependent projects).

## Version Format: MAJOR.MINOR.PATCH

### MAJOR Version (1.x.x → 2.0.0)

**When to Increment**: Breaking changes to public API

**Examples of Breaking Changes**:
- Removing public classes, methods, or attributes
- Changing method signatures (parameters, return types)
- Changing expected behavior of core operations
- Removing support for Python versions
- Breaking changes in FlextResult, FlextContainer, FlextModels

**Guarantees**:
- **Minimum 6 months** between major version announcements
- **Deprecation cycle**: Minimum 2 minor versions before removal
- **Migration tools**: Automated migration utilities provided
- **Documentation**: Complete migration guide with examples

**Example**:
```python
# 1.x.x - Current API
result = FlextResult[str].ok("value")
data = result.value  # or result.data (both supported)

# 2.0.0 - Breaking change (hypothetical)
result = FlextResult[str].ok("value")
data = result.data  # .value removed (would require migration)
```

### MINOR Version (1.0.x → 1.1.0)

**When to Increment**: New features without breaking changes

**Examples of Minor Changes**:
- Adding new public classes, methods, or attributes
- Adding optional parameters with defaults
- Deprecating features (with warnings)
- Adding new patterns or utilities
- Performance improvements

**Guarantees**:
- **Zero breaking changes** to existing API
- **Backward compatible**: All 1.x.y code works with 1.(x+1).0
- **Deprecation warnings**: Clear warnings for deprecated features
- **Documentation**: Comprehensive feature documentation

**Example**:
```python
# 1.0.0
result = FlextResult[str].ok("value")

# 1.1.0 - New feature added
result = FlextResult[str].ok("value")
result_with_metadata = result.with_metadata({"key": "value"})  # NEW
```

### PATCH Version (1.0.0 → 1.0.1)

**When to Increment**: Bug fixes without breaking changes

**Examples of Patch Changes**:
- Fixing incorrect behavior
- Security patches
- Documentation corrections
- Performance fixes
- Internal refactoring without API changes

**Guarantees**:
- **Zero API changes**: Exact same public interface
- **Safe upgrades**: Drop-in replacement
- **Quick turnaround**: Released as needed for critical fixes

**Example**:
```python
# 1.0.0 - Bug in validation
result = FlextResult[int].ok(-1)  # Should fail validation
assert result.is_success  # Bug: doesn't validate

# 1.0.1 - Bug fixed
result = FlextResult[int].ok(-1)  # Now properly validates
assert result.is_failure  # Fixed: validation works
```

## Dependency Version Locking

### Runtime Dependencies (Locked for ABI Stability)

All runtime dependencies use **compatible version ranges** to prevent breaking changes:

```toml
dependencies = [
    "pydantic>=2.11.7,<3.0.0",           # Pydantic 2.x API stable
    "pydantic-settings>=2.10.1,<3.0.0",  # Aligned with pydantic
    "pyyaml>=6.0.2,<7.0.0",              # YAML 6.x stable
    "structlog>=25.4.0,<26.0.0",         # CalVer YY.MINOR
    "typing-extensions>=4.12.0,<5.0.0",  # Type system stability
    "colorlog>=6.9.0,<7.0.0",            # Logging stability
]
```

**Rationale**:
- Lower bound: Minimum required version with needed features
- Upper bound: Prevent major version breaking changes
- Range: Allow patch and minor updates within major version

### Python Version Support

```toml
requires-python = ">=3.13,<3.14"
```

**Policy**:
- **1.x series**: Python 3.13 only (current)
- **Future versions**: May support 3.14+ in minor releases
- **Deprecation**: Minimum 12 months notice before dropping Python version

## API Stability Guarantees

### Guaranteed Stable for 1.x Series

The following APIs are **guaranteed stable** throughout the 1.x series:

#### Core Result Pattern
```python
from flext_core import FlextResult

# Guaranteed API (no breaking changes in 1.x)
result = FlextResult[T].ok(value)
result = FlextResult[T].fail(error)
result.is_success  # bool
result.is_failure  # bool
result.value       # T or None
result.data        # T or None (alias for .value)
result.error       # str or None
result.unwrap()    # T or raises
```

#### Dependency Injection
```python
from flext_core import FlextContainer

# Guaranteed API
container = FlextContainer.get_global()
container.register(interface, implementation)
container.resolve(interface)
```

#### Domain-Driven Design Models
```python
from flext_core import FlextModels

# Guaranteed API
class MyEntity(FlextModels.Entity): ...
class MyValue(FlextModels.Value): ...
class MyAggregate(FlextModels.AggregateRoot): ...
```

#### Service Pattern
```python
from flext_core import FlextService

# Guaranteed API
class MyService(FlextService[ConfigType]): ...
```

#### Structured Logging
```python
from flext_core import FlextLogger

# Guaranteed API
logger = FlextLogger(__name__)
logger.info("message", extra={"key": "value"})
```

### Deprecation Policy

**Minimum Lifecycle**: 2 minor versions

**Example Timeline**:
1. **Version 1.0.0**: Feature `old_method()` exists
2. **Version 1.1.0**: Feature deprecated with `DeprecationWarning`
3. **Version 1.2.0**: Feature still works with warning
4. **Version 2.0.0**: Feature removed (major version bump required)

**Deprecation Warnings**:
```python
import warnings

def old_method():
    warnings.warn(
        "old_method() is deprecated and will be removed in 2.0.0. "
        "Use new_method() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_method()
```

## Release Process

### Pre-Release Checklist

- [ ] All tests passing (100% pass rate required)
- [ ] Coverage maintained (≥79% minimum)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Migration guide (if breaking changes)
- [ ] Security audit clean
- [ ] Type checking passing (MyPy + PyRight)
- [ ] Lint checks passing (Ruff)

### Release Steps

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Document all changes in `CHANGELOG.md`
3. **Tag**: Create git tag `v{MAJOR}.{MINOR}.{PATCH}`
4. **Build**: `poetry build`
5. **Test**: Install and test in clean environment
6. **Publish**: `poetry publish` to PyPI
7. **Announce**: Update documentation and notify ecosystem

### Version Tag Format

```bash
git tag -a v1.0.0 -m "Release 1.0.0 - Stable Foundation"
git push origin v1.0.0
```

## Ecosystem Impact

### Dependent Projects (32+)

All FLEXT ecosystem libraries depend on flext-core:

- flext-api, flext-cli, flext-auth, flext-web
- flext-ldap, flext-ldif, flext-db-oracle
- flext-meltano, flext-quality, flext-grpc
- Singer taps and targets (12+ projects)
- DBT projects (4+ projects)
- Enterprise tools (client-a-oud-mig, client-b-meltano-native)

**Testing Requirements**:
- All dependent projects must pass tests with new version
- Integration tests validate backward compatibility
- Performance regression tests ensure no degradation

## Version History

### 0.9.9 (Current) - Release Candidate
- Preparing for 1.0.0 stable release
- API surface stabilized
- Dependency versions locked
- 79% test coverage achieved

### 1.0.0 (Planned: October 2025) - Stable Release
- First stable release with API guarantees
- Complete semantic versioning commitment
- Full ecosystem compatibility validated

## Contact & Support

- **Repository**: https://github.com/flext-sh/flext-core
- **Issues**: https://github.com/flext-sh/flext-core/issues
- **Documentation**: https://github.com/flext-sh/flext-core/blob/main/README.md
- **Team**: team@flext.sh
