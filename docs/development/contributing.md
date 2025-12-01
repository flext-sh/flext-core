# Contributing to FLEXT-Core

Thank you for your interest in contributing to FLEXT-Core! This guide provides comprehensive instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Quick Start

### Prerequisites

- **Python**: 3.13+ (required)
- **Poetry**: Latest version (recommended)
- **Git**: For source checkout
- **Make**: For development commands

### Development Setup

```bash
# Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Setup development environment
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('✅ FLEXT-Core ready')"
```

## How to Contribute

### 1. Reporting Issues

**Before creating an issue:**

- Search existing issues to avoid duplicates
- Use issue templates when available
- Provide clear reproduction steps for bugs
- Include system information (Python version, OS, etc.)

**Issue Types:**

- 🐛 **Bug Reports**: Unexpected behavior or crashes
- 🚀 **Feature Requests**: New functionality suggestions
- 📚 **Documentation**: Documentation improvements
- 🔧 **Enhancements**: Performance or usability improvements

### 2. Suggesting Features

**Guidelines:**

- Check the roadmap in issues/discussions first
- Explain the use case, not just the solution
- Consider backward compatibility impacts
- Be open to alternative suggestions

**Feature Request Template:**

```markdown
## Problem Statement

[Clear description of the problem you're trying to solve]

## Proposed Solution

[Your suggested approach]

## Use Cases

[Specific examples of how this would be used]

## Alternatives Considered

[Other approaches you've considered]
```

### 3. Submitting Pull Requests

**PR Requirements:**

- Fork the repository and create a feature branch
- Follow the development setup instructions
- Write tests for new functionality
- Update documentation as needed
- Ensure all checks pass before submitting
- Keep PRs focused - one feature/fix per PR

**PR Process:**

1. **Create branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Implement your feature or fix
3. **Write tests**: Add comprehensive tests
4. **Update docs**: Update documentation if needed
5. **Run checks**: `make validate` (lint + type-check + tests)
6. **Submit PR**: Create pull request with clear description

## Development Workflow

### Quality Assurance Pipeline

```bash
# Complete validation (required before PR)
make validate

# Individual checks
make lint           # Ruff linting (ZERO tolerance)
make type-check     # MyPy strict + PyRight
make test          # Full test suite with coverage
make security      # Bandit + pip-audit

# Quick validation during development
make check         # lint + type-check only
make format        # Auto-format code
make fix           # Auto-fix linting issues
```

### Testing

**Test Categories:**

- **Unit Tests**: Core functionality (fast, isolated)
- **Integration Tests**: Component interaction
- **Pattern Tests**: Architectural patterns (CQRS, DDD)

**Running Tests:**

```bash
# All tests with coverage
make test

# Specific test types
make test-unit         # Unit tests only
make test-integration  # Integration tests only

# Specific modules
pytest tests/unit/test_result.py -v
pytest tests/unit/test_container.py::TestFlextContainer::test_singleton -v
```

**Test Markers:**

```bash
# Run by marker
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests

# With coverage
pytest tests/unit/test_result.py --cov=src/flext_core/result.py --cov-report=term-missing
```

### Code Quality Standards

**Mandatory Requirements:**

- **Zero Ruff violations** - Code quality enforced
- **Zero MyPy errors** - Type safety guaranteed
- **Zero PyRight errors** - Enhanced type checking
- **PEP 8 compliance** - 79 character line length
- **Python 3.13+** - Modern syntax and features

**Best Practices:**

- Use `FlextResult[T]` for all operations that can fail
- Register services with `FlextContainer.get_global()`
- Follow DDD patterns with `FlextModels.Entity/Value/AggregateRoot`
- Use `FlextLogger` with context propagation
- Write tests using `flext_tests` infrastructure (no mocks)
- Keep functions small and focused (single responsibility)

### Debugging and Troubleshooting

**Common Issues:**

1. **Import Errors**

   ```bash
   # Ensure Python 3.13+
   python --version

   # Reinstall dependencies
   make install
   ```

2. **Type Errors**

   ```bash
   # Run type checker
   make type-check

   # Check specific files
   mypy src/flext_core/your_module.py
   ```

3. **Test Failures**

   ```bash
   # Run with verbose output
   pytest tests/ -v --tb=short

   # Debug specific test
   pytest tests/unit/test_result.py::TestFlextResult::test_ok -v -s
   ```

### Documentation Updates

**When to Update Documentation:**

- New public APIs or breaking changes
- New configuration options
- New architectural patterns or concepts
- Bug fixes that change behavior

**Documentation Standards:**

- Clear, concise language
- Practical examples for each concept
- Consistent formatting and structure
- Links to related documentation

## Architecture Guidelines

### Adding New Features

**1. Foundation Layer Changes**

- Be extremely cautious - affects entire ecosystem
- Consider backward compatibility impact
- Update type definitions in `typings.py`
- Add comprehensive tests

**2. Domain Layer Changes**

- Focus on business logic only
- Use `FlextModels` for entities and value objects
- Implement validation in `model_post_init`
- Add domain events for significant state changes

**3. Application Layer Changes**

- Use CQRS patterns with `FlextDispatcher`
- Implement handlers for commands/queries
- Add middleware for cross-cutting concerns
- Register components in `FlextRegistry`

**4. Infrastructure Layer Changes**

- Abstract external dependencies
- Use `FlextProtocols` for runtime contracts
- Implement proper error handling
- Add configuration options for new features

### Code Organization

**Module Structure:**

```
src/flext_core/
├── __init__.py          # Public API exports
├── result.py           # Railway pattern implementation
├── container.py        # Dependency injection
├── models.py           # DDD base classes
├── service.py          # Domain service base
├── bus.py             # Message bus
├── config.py          # Configuration management
├── loggings.py        # Structured logging
└── ... (other modules)
```

**Import Guidelines:**

```python
# ✅ Good - Direct imports
from flext_core import FlextDispatcher
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

# ❌ Bad - Star imports in production code
from flext_core import *

# ❌ Bad - Relative imports in public APIs
from .result import FlextResult
```

## Review Process

### PR Review Checklist

**Code Quality:**

- [ ] Zero Ruff violations
- [ ] Zero MyPy/PyRight errors
- [ ] Tests pass and provide good coverage
- [ ] Documentation updated
- [ ] No breaking changes without migration guide

**Architecture:**

- [ ] Follows Clean Architecture principles
- [ ] Proper separation of concerns
- [ ] No circular dependencies
- [ ] Appropriate abstraction level

**Testing:**

- [ ] Unit tests for new functionality
- [ ] Integration tests for component interaction
- [ ] Edge cases and error conditions covered
- [ ] Performance impact considered

### Approval Process

1. **Automated Checks**: All PRs must pass CI/CD pipeline
2. **Code Review**: At least one maintainer approval required
3. **Testing**: All tests must pass on multiple Python versions
4. **Documentation**: Documentation updates must be approved
5. **Merge**: Maintainers merge approved PRs

## Community Guidelines

### Communication

**Preferred Channels:**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and discussions
- **Pull Requests**: Code contributions and reviews

**Response Times:**

- Issues: Acknowledged within 2 business days
- PRs: Initial review within 3 business days
- Questions: Answered within 1 business day

### Recognition

Contributors who make significant improvements may be:

- Added to contributors list
- Invited to become maintainers
- Featured in release notes
- Nominated for community awards

## Getting Help

### Resources

- **[Documentation](./)**: Complete documentation
- **[Examples](../examples/)**: Working code examples
- **[Tests](../tests/)**: Usage patterns and best practices
- **[Issues](https://github.com/flext-sh/flext-core/issues)**: Report bugs or ask questions
- **[Discussions](https://github.com/flext-sh/flext-core/discussions)**: Community discussions

### Support Levels

**Community Support:**

- GitHub Issues and Discussions
- Stack Overflow (tag: flext-core)
- Community chat (if available)

**Commercial Support:**

- Priority issue handling
- Direct maintainer access
- Custom feature development
- Training and consulting

---

Thank you for contributing to FLEXT-Core! Your contributions help make this a better framework for the entire ecosystem.

**Happy coding!** 🚀
