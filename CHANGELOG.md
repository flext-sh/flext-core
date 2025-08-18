# Changelog

All notable changes to FLEXT Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive documentation update across all modules
- CONTRIBUTING.md with development guidelines
- Improved architecture documentation with real-world examples

### Changed

- Updated README.md to reflect current project status
- Enhanced quickstart guide with practical examples
- Modernized documentation to match actual implementation

### Fixed

- Documentation inconsistencies with actual code
- Type hints in various modules

## [0.9.0] - 2025-08-10

### Added

- **Core Patterns**
  - `FlextResult[T]` - Complete railway-oriented programming pattern
  - `FlextContainer` - Enterprise dependency injection system
  - Type-safe error handling throughout the library
- **Domain Patterns**
  - `FlextEntity` - DDD entities with identity
  - `FlextValueObject` - Immutable value objects
  - `FlextAggregateRoot` - Aggregate consistency boundaries
  - `FlextDomainService` - Stateless domain operations
- **Configuration**
  - `FlextSettings` - Pydantic-based configuration management
  - Environment variable support with prefixes
  - Type-safe configuration validation
- **Infrastructure**
  - Structured logging with correlation IDs
  - `FlextPayload` for event/message patterns
  - Observability patterns for monitoring
- **CQRS Foundation**
  - `FlextCommand` - Command pattern implementation
  - `FlextMessageHandler` - Command/query handlers
  - Basic validation framework

### Changed

- Migrated to Python 3.13+ only (no backward compatibility)
- Adopted MyPy strict mode for type checking
- Implemented 75% minimum test coverage requirement
- Restructured modules following Clean Architecture

### Fixed

- Type safety issues in core modules
- Container singleton pattern implementation
- Domain event handling in aggregates

### Security

- Added bandit security scanning
- Implemented pip-audit for dependency scanning
- No known vulnerabilities

## [0.8.0] - 2024-12-01

### Added

- Initial `FlextResult` pattern implementation
- Basic dependency injection container
- Foundation for domain patterns
- Initial test suite

### Changed

- Project structure reorganization
- Adopted Poetry for dependency management

### Deprecated

- Legacy error handling patterns

## [0.7.0] - 2024-10-15

### Added

- Project inception
- Basic module structure
- Initial documentation

---

## Version Guidelines

### Version Format

`MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Pre-release Versions

- Alpha: `0.x.0-alpha.n`
- Beta: `0.x.0-beta.n`
- RC: `0.x.0-rc.n`

### Deprecated Features

Features marked as deprecated will be:

1. Maintained for 2 minor versions
2. Documented with migration path
3. Removed in next major version

## Release Process

1. Update version in `pyproject.toml`
2. Update this CHANGELOG.md
3. Create git tag: `git tag -a v0.9.0 -m "Release version 0.9.0"`
4. Push tag: `git push origin v0.9.0`
5. GitHub Actions publishes to PyPI

## Links

- [Unreleased]: https://github.com/flext-sh/flext-core/compare/v0.9.0...HEAD
- [0.9.0]: https://github.com/flext-sh/flext-core/compare/v0.8.0...v0.9.0
- [0.8.0]: https://github.com/flext-sh/flext-core/compare/v0.7.0...v0.8.0
- [0.7.0]: https://github.com/flext-sh/flext-core/releases/tag/v0.7.0
