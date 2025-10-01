# Changelog

All notable changes to FLEXT-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 1.0.0 Release Preparation
- ABI finalization and dependency version locking
- Comprehensive API stability documentation
- Ecosystem compatibility testing
- Migration guide creation

---

## [1.0.0] - 2025-10-XX (Planned)

**STABLE RELEASE** - First production-ready release with API stability guarantees

### Added

#### Core Foundation
- **API Stability Guarantees**: Complete documentation of stable APIs for 1.x series (see [API_STABILITY.md](API_STABILITY.md))
- **Semantic Versioning Strategy**: Formal SemVer 2.0.0 commitment with deprecation policy (see [VERSIONING.md](VERSIONING.md))
- **Migration Guide**: Complete upgrade documentation from 0.x to 1.0 (see [MIGRATION_0x_TO_1.0.md](MIGRATION_0x_TO_1.0.md))
- **HTTP Primitives** (new in 0.9.9, stable in 1.0.0):
  - `FlextConstants.Http`: HTTP status codes, methods, content types, ports
  - `FlextModels.HttpRequest`: Base HTTP request model with validation
  - `FlextModels.HttpResponse`: Base HTTP response model with computed properties

#### Documentation
- **1.0.0 Roadmap**: Complete 5-week release timeline in README.md
- **Ecosystem Compatibility Report**: Testing results for top 5 dependent projects
- **Stability Guarantees**: Three-level API stability classification (100%, 99%, 95%)

### Changed

#### Dependency Management (ABI Stability)
- **pydantic**: Locked to `>=2.11.7,<3.0.0` (Pydantic 2.x API stable)
- **pydantic-settings**: Locked to `>=2.10.1,<3.0.0` (aligned with pydantic)
- **pyyaml**: Locked to `>=6.0.2,<7.0.0` (YAML 6.x stable)
- **structlog**: Locked to `>=25.4.0,<26.0.0` (CalVer YY.MINOR)
- **typing-extensions**: Locked to `>=4.12.0,<5.0.0` (type system stability)
- **colorlog**: Locked to `>=6.9.0,<7.0.0` (logging stability)

**Rationale**: Version bounds prevent breaking changes from major dependency updates while allowing security patches and minor improvements.

### Guaranteed Stable (Level 1: 100%)

These APIs are **guaranteed stable forever** in the 1.x series:

#### FlextResult - Railway-Oriented Programming
```python
# All methods and properties guaranteed
result = FlextResult[T].ok(value)          # Create success
result = FlextResult[T].fail(error)        # Create failure

# Dual access permanently supported
result.value: T | None                      # Primary access
result.data: T | None                       # ABI compatibility

# Properties (all guaranteed)
result.is_success: bool
result.is_failure: bool
result.error: str | None

# Methods (all guaranteed)
result.unwrap() -> T
result.unwrap_or(default: T) -> T
result.map(func) -> FlextResult[U]
result.bind(func) -> FlextResult[U]
```

#### FlextContainer - Dependency Injection
```python
# All methods guaranteed
container = FlextContainer.get_global()
container.register(interface, impl)
container.register_factory(interface, factory)
container.register_singleton(interface, instance)
container.resolve(interface) -> T
container.resolve_all(interface) -> list[T]
container.clear()
```

#### FlextModels - Domain-Driven Design
```python
# All base classes guaranteed
class MyEntity(FlextModels.Entity): pass
class MyValue(FlextModels.Value): pass
class MyAggregate(FlextModels.AggregateRoot): pass

# HTTP models (new in 0.9.9, stable in 1.0.0)
class MyRequest(FlextModels.HttpRequest): pass
class MyResponse(FlextModels.HttpResponse): pass
```

#### FlextService - Service Pattern
```python
# Service base class guaranteed
class MyService(FlextService[ConfigType]): pass
```

#### FlextLogger - Structured Logging
```python
# All log levels guaranteed
logger = FlextLogger(__name__)
logger.debug/info/warning/error/critical(msg, extra={})
logger.exception(msg, exc_info=True)
```

### Deprecated

**NONE** in 1.0.0 release - All 0.9.9 APIs remain fully supported.

Future deprecations will follow minimum 2 minor version policy (e.g., 1.0 → 1.1 → 1.2).

### Removed

**NONE** in 1.0.0 release - 100% backward compatible with 0.9.9.

### Fixed

- Improved type hints for better IDE support
- Enhanced documentation for all public APIs
- Clarified dual access pattern (`.value` and `.data`) in FlextResult

### Security

- Dependency versions locked to prevent supply chain attacks
- Security patches guaranteed within 48 hours for stability issues

---

## [0.9.9] - 2025-10-01

**RELEASE CANDIDATE** - Pre-1.0.0 release with HTTP primitives and final API surface

### Added

#### HTTP Primitives Foundation
- **FlextConstants.Http**: Complete HTTP constants namespace
  - Status codes (HTTP_OK, HTTP_CREATED, HTTP_BAD_REQUEST, etc.)
  - Status ranges (SUCCESS_MIN/MAX, CLIENT_ERROR_MIN/MAX, etc.)
  - HTTP methods (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS, TRACE, CONNECT)
  - Content types (JSON, XML, HTML, FORM, TEXT, BINARY)
  - Ports (HTTP_PORT=80, HTTPS_PORT=443, HTTP_ALT_PORT=8080, etc.)

- **FlextModels.HttpRequest**: Base HTTP request model
  - Fields: url, method, headers, body, timeout
  - Computed properties: is_secure, has_body
  - Pydantic v2 validation with field validators

- **FlextModels.HttpResponse**: Base HTTP response model
  - Fields: status_code, headers, body, elapsed_time
  - Computed properties: is_success, is_client_error, is_server_error
  - Status range validation

#### Documentation Improvements
- Complete API reference for HTTP primitives
- Examples for extending HTTP base models
- Integration patterns with flext-api and flext-web

### Changed

- Improved FlextResult type hints for better inference
- Enhanced FlextContainer documentation with dependency injection examples
- Updated FlextModels with HTTP base classes

### Fixed

- Type checking improvements for strict MyPy mode
- Documentation corrections in README.md
- Import path consistency across modules

---

## [0.9.0] - 2025-09-18

**MAJOR MILESTONE** - Foundation stable release with complete ecosystem integration

### Added

#### Core Foundation
- **FlextResult[T]**: Railway-oriented programming with dual `.value`/`.data` access
- **FlextContainer**: Singleton dependency injection with typed service keys
- **FlextModels**: Domain-driven design patterns (Entity, Value, AggregateRoot)
- **FlextService**: Domain service base class with Pydantic Generic[T]
- **FlextLogger**: Structured logging with context propagation and correlation IDs

#### Advanced Patterns
- **FlextContext**: Request/operation context with metadata
- **FlextCqrs**: Command/Query/Event patterns
- **FlextBus**: Message bus with middleware pipeline and caching
- **FlextDispatcher**: Unified command/query dispatcher
- **FlextRegistry**: Handler registry management
- **FlextProcessors**: Message processing orchestration

#### Infrastructure
- **FlextConfig**: Layered configuration with .env, TOML, YAML support
- **FlextMixins**: Reusable behaviors (timestamps, serialization, validation)
- **FlextUtilities**: Domain utilities (validation, conversion, type guards)
- **FlextProtocols**: Runtime-checkable interfaces
- **FlextExceptions**: Comprehensive exception hierarchy with error codes
- **FlextConstants**: Centralized constants and enumerations
- **FlextTypes**: Complete type system (50+ TypeVars, Protocols, Aliases)

### Quality Metrics (0.9.0)

- **Test Coverage**: 79% (proven stable baseline)
- **Type Safety**: Python 3.13 + MyPy strict mode + PyRight
- **Code Quality**: Zero Ruff violations in src/
- **Tests**: 1,163 passing (unit + integration + patterns)
- **Ecosystem**: Powers 32+ dependent packages

---

## [0.8.x] - 2025-08-XX

**BETA RELEASES** - Feature development and stabilization

### Added
- Initial implementation of core patterns
- Basic dependency injection
- Domain-driven design foundations
- Configuration management
- Structured logging

### Changed
- Multiple API refinements based on ecosystem feedback
- Performance improvements
- Type system enhancements

---

## [0.7.x and earlier]

**ALPHA RELEASES** - Initial development phase

---

## Migration Guides

### Upgrading to 1.0.0 from 0.9.9

**Complexity**: ⭐ Trivial (0/5 difficulty)
**Time Required**: < 5 minutes

**Steps**:
1. Update dependency: `flext-core>=1.0.0,<2.0.0`
2. Run tests (no changes needed)
3. Deploy with confidence

**See**: [MIGRATION_0x_TO_1.0.md](MIGRATION_0x_TO_1.0.md) for complete guide

### Upgrading to 0.9.9 from 0.9.0

**Added Features**: HTTP primitives (optional enhancement)
**Breaking Changes**: NONE
**Migration Time**: < 5 minutes

---

## Deprecation Policy

### Minimum Deprecation Cycle

**Timeline**: Minimum 2 minor versions before removal

**Example**:
- Version 1.0.0: Feature exists
- Version 1.1.0: Feature deprecated with `DeprecationWarning`
- Version 1.2.0: Feature still works with warning
- Version 2.0.0: Feature removed (major version bump required)

### Deprecation Process

1. Add `DeprecationWarning` with migration instructions
2. Maintain functionality for minimum 2 minor versions
3. Provide alternative solution before deprecation
4. Only remove in major version update

---

## Support Policy

### Version Support

- **1.x Series**: Active development with backward compatibility guaranteed
- **Security Patches**: Critical issues fixed within 48 hours
- **Bug Fixes**: Patch releases (1.0.x) as needed
- **New Features**: Minor releases (1.x.0) with backward compatibility

### End of Support

- **0.9.x**: Superseded by 1.0.0 (upgrade recommended, still functional)
- **0.8.x and earlier**: No longer supported (upgrade to 1.0.0)

---

## Breaking Changes Policy (1.x Series)

**GUARANTEED**: ZERO breaking changes in 1.x releases

### What Changes Are Allowed

✅ **PATCH** (1.0.0 → 1.0.1):
- Bug fixes
- Security patches
- Documentation corrections
- Internal implementation improvements

✅ **MINOR** (1.0.0 → 1.1.0):
- New classes, methods, functions (additions only)
- New optional parameters with defaults
- Deprecation warnings (features still work)
- Performance improvements

❌ **FORBIDDEN** in 1.x (Requires 2.0.0):
- Removing public APIs
- Changing method signatures without backward compatibility
- Changing expected behavior
- Removing deprecated features

---

## Links

- **Repository**: https://github.com/flext-sh/flext-core
- **Documentation**: [README.md](README.md)
- **API Stability**: [API_STABILITY.md](API_STABILITY.md)
- **Versioning**: [VERSIONING.md](VERSIONING.md)
- **Migration Guide**: [MIGRATION_0x_TO_1.0.md](MIGRATION_0x_TO_1.0.md)
- **Issues**: https://github.com/flext-sh/flext-core/issues
- **Discussions**: https://github.com/flext-sh/flext-core/discussions

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/) principles and [Semantic Versioning](https://semver.org/).
