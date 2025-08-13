# Contributing to FLEXT Core

Thank you for your interest in contributing to FLEXT Core! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** when available
3. **Provide clear reproduction steps** for bugs
4. **Include system information** (Python version, OS, etc.)

### Suggesting Features

1. **Check the roadmap** in issues/discussions first
2. **Explain the use case** not just the solution
3. **Consider backward compatibility** impacts
4. **Be open to alternatives** suggested by maintainers

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch
2. **Follow the development setup** instructions below
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all checks pass** before submitting
6. **Keep PRs focused** - one feature/fix per PR

## Development Setup

### Prerequisites

- Python 3.13+
- Poetry 1.8+
- Make
- Git

### Initial Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/flext-core.git
cd flext-core

# Add upstream remote
git remote add upstream https://github.com/flext-sh/flext-core.git

# Setup development environment
make setup

# Verify setup
make doctor
```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test iteratively
make test-fast  # Quick tests without coverage

# Run full validation before commit
make validate   # MANDATORY - all checks must pass

# Commit with descriptive message
git commit -m "feat: add new validation pattern"

# Push to your fork
git push origin feature/your-feature-name
```

## Quality Standards

### Required Checks

All pull requests must pass these checks:

```bash
make validate  # Runs all checks below
```

Individual checks:

- **Linting**: `make lint` - Code style with ruff
- **Type Checking**: `make type-check` - MyPy strict mode
- **Testing**: `make test` - 75% minimum coverage
- **Security**: `make security` - Bandit scanning

### Code Style

#### Python Style

- **Line Length**: 79 characters maximum (PEP8)
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions
- **Naming**: `FlextXxx` prefix for public exports

Example:

```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[str]:
    """Process input data with validation.

    Args:
        data: The input string to process.

    Returns:
        FlextResult containing processed data or error message.

    Example:
        >>> result = process_data("hello")
        >>> if result.success:
        ...     print(result.unwrap())
    """
    if not data:
        return FlextResult.fail("Data cannot be empty")

    processed = data.strip().upper()
    return FlextResult.ok(processed)
```

#### Import Order

```python
# Standard library
import os
import sys
from datetime import datetime

# Third-party
import pydantic
from structlog import get_logger

# Local application
from flext_core import FlextResult, FlextContainer
from flext_core.typings import TData
```

### Testing Requirements

#### Test Structure

```python
import pytest
from flext_core import FlextResult

class TestFlextResult:
    """Test FlextResult railway pattern."""

    def test_success_case(self):
        """Test successful result creation."""
        result = FlextResult.ok("test")
        assert result.success
        assert result.data == "test"

    def test_failure_case(self):
        """Test failure result creation."""
        result = FlextResult.fail("error")
        assert result.is_failure
        assert result.error == "error"

    @pytest.mark.parametrize("value,expected", [
        ("test", "TEST"),
        ("hello", "HELLO"),
    ])
    def test_map_operation(self, value, expected):
        """Test map transformation."""
        result = FlextResult.ok(value).map(str.upper)
        assert result.unwrap() == expected
```

#### Test Categories

Use appropriate markers:

```python
@pytest.mark.unit        # Unit tests (default)
@pytest.mark.integration # Integration tests
@pytest.mark.slow        # Long-running tests
@pytest.mark.performance # Performance benchmarks
```

### Documentation

#### Docstring Format

```python
def complex_function(
    param1: str,
    param2: int,
    optional: bool = False
) -> FlextResult[dict]:
    """Brief description of function purpose.

    Longer description explaining behavior, edge cases,
    and important details about the implementation.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.
        optional: Description of optional parameter.
            Defaults to False.

    Returns:
        FlextResult containing a dictionary with keys:
        - key1: Description of key1
        - key2: Description of key2

    Raises:
        Never raises - returns FlextResult.fail() on error.

    Example:
        >>> result = complex_function("test", 42)
        >>> if result.success:
        ...     data = result.unwrap()
        ...     print(data["key1"])

    Note:
        This function is thread-safe.
    """
```

## Commit Messages

Follow conventional commits:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions/changes
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```bash
# Feature
git commit -m "feat(result): add async support for FlextResult"

# Bug fix
git commit -m "fix(container): resolve singleton registration issue"

# Documentation
git commit -m "docs(api): update FlextEntity examples"

# Breaking change
git commit -m "feat(core): redesign FlextResult API

BREAKING CHANGE: FlextResult.unwrap() now raises on failure"
```

## Architecture Guidelines

### Design Principles

1. **Railway-Oriented**: All operations return FlextResult
2. **Type Safety**: Comprehensive type hints with generics
3. **Clean Architecture**: Clear layer separation
4. **DDD Patterns**: Rich domain models
5. **Minimal Dependencies**: Only essential libraries

### Adding New Features

1. **Discuss First**: Open an issue for significant changes
2. **Design Document**: For complex features, write a design doc
3. **Backward Compatibility**: Don't break existing APIs
4. **Migration Path**: Provide upgrade guides for breaking changes
5. **Examples**: Include working examples in docs

### Module Organization

```
src/flext_core/
├── Foundation (core patterns)
│   └── Must not depend on other layers
├── Domain (business logic)
│   └── Depends only on Foundation
├── Application (use cases)
│   └── Depends on Foundation and Domain
└── Infrastructure (external)
    └── Can depend on all layers
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`make type-check`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description

Brief description of changes

## Motivation

Why these changes are needed

## Changes

- Change 1
- Change 2

## Testing

How the changes were tested

## Breaking Changes

List any breaking changes

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All checks pass
```

### Review Process

1. **Automated Checks**: Must pass all CI checks
2. **Code Review**: At least one maintainer approval
3. **Documentation Review**: For API changes
4. **Testing**: Verify test coverage
5. **Merge**: Squash and merge to main

## Release Process

### Version Numbering

We follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI

## Getting Help

### Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

### Contact

- Open an issue for bugs/features
- Start a discussion for questions
- Tag maintainers for urgent issues

## Recognition

Contributors are recognized in:

- CHANGELOG.md (for each release)
- GitHub contributors page
- Release notes

Thank you for contributing to FLEXT Core! 🎉
