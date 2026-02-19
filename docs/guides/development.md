<!-- Generated from docs/guides/development.md for flext-core. -->

<!-- Source of truth: workspace docs/guides/. -->

# flext-core - FLEXT Development Guide

> Project profile: `flext-core`

<!-- TOC START -->

- [Prerequisites](#prerequisites)
- [Development Environment Setup](#development-environment-setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Verify Installation](#3-verify-installation)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
  - [1. Create a Feature Branch](#1-create-a-feature-branch)
  - [2. Make Changes](#2-make-changes)
  - [3. Run Quality Gates](#3-run-quality-gates)
  - [4. Commit Changes](#4-commit-changes)
- [Code Standards](#code-standards)
  - [Type Safety (ZERO TOLERANCE)](#type-safety-zero-tolerance)
  - [Railway-Oriented Programming](#railway-oriented-programming)
  - [Unified Models Pattern](#unified-models-pattern)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Writing Tests](#writing-tests)
- [Quality Gates](#quality-gates)
  - [Pre-commit Hooks](#pre-commit-hooks)
  - [Quality Checks](#quality-checks)
- [Adding New Projects](#adding-new-projects)
  - [1. Create Project Structure](#1-create-project-structure)
  - [2. Implement Core Patterns](#2-implement-core-patterns)
  - [3. Add to Workspace](#3-add-to-workspace)
- [Debugging](#debugging)
  - [Type Errors](#type-errors)
  - [Test Failures](#test-failures)
  - [Import Issues](#import-issues)
- [Documentation](#documentation)
  - [Code Documentation](#code-documentation)
  - [README Updates](#readme-updates)
- [Contributing](#contributing)
  - [Pull Request Process](#pull-request-process)
  - [Code Review Guidelines](#code-review-guidelines)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
- [Resources](#resources)
- [Support](#support)

<!-- TOC END -->

This guide covers setting up a development environment for FLEXT contributions and understanding the development workflow.

## Prerequisites

- **Python 3.13+** (required for all FLEXT projects)
- **Poetry** (for dependency management)
- **Git** (for version control)
- **Docker** (optional, for containerized development)

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/flext-sh/flext.git
cd flext
```

### 2. Install Dependencies

```bash
# Install all dependencies and pre-commit hooks
make setup

# Or install manually
poetry install
pre-commit install
```

### 3. Verify Installation

```bash
# Run quality gates to verify setup
make validate

# Check individual components
make lint-all
make type-check-all
make test-all
```

## Project Structure

FLEXT is organized as a monorepo with the following structure:

```
flext/
├── flext-core/           # Foundation library
├── flext-api/            # HTTP client and FastAPI
├── flext-auth/           # Authentication services
├── flext-ldap/           # LDAP operations
├── flext-ldif/           # LDIF processing
├── flext-grpc/           # gRPC services
├── flext-cli/            # Command-line interface
├── flext-meltano/        # Meltano integration
├── flext-observability/  # Monitoring and metrics
├── flext-quality/        # Quality assurance tools
├── docs/                 # Documentation
├── scripts/              # Development scripts
└── examples/             # Usage examples
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
```

### 2. Make Changes

Follow FLEXT development standards:

- **Use FlextResult[T]** for all operations
- **Follow Clean Architecture** principles
- **Maintain type safety** with MyPy strict mode
- **Write comprehensive tests**

### 3. Run Quality Gates

```bash
# Quick validation (before commit)
make check

# Full validation (before push)
make validate
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(component): add amazing feature"
git push origin feature/amazing-feature
```

## Code Standards

### Type Safety (ZERO TOLERANCE)

```python
# ✅ CORRECT - Complete type annotations
def process_data(data: dict[str, object]) -> FlextResult[ProcessedData]:
    """Process data with type safety."""
    if not data:
        return FlextResult[ProcessedData].fail("Data required")

    return FlextResult[ProcessedData].ok(ProcessedData(**data))

# ❌ WRONG - Missing type annotations
def process_data(data):
    return data
```

### Railway-Oriented Programming

```python
# ✅ CORRECT - Use FlextResult for all operations
def validate_and_process(data: dict) -> FlextResult[ProcessedData]:
    return (
        validate_data(data)
        .flat_map(transform_data)
        .map(enrich_data)
        .map_error(handle_error)
    )

# ❌ WRONG - Exception-based error handling
def validate_and_process(data: dict) -> ProcessedData:
    if not data:
        raise ValueError("Data required")
    return transform_data(data)
```

### Unified Models Pattern

```python
# ✅ CORRECT - Use [Project]Models pattern
class FlextApiModels:
    class Request(BaseModel):
        data: dict[str, object]

    class Response(BaseModel):
        result: FlextResult[object]
        status: int

# ❌ WRONG - Scattered model definitions
class ApiRequest(BaseModel):
    data: dict[str, object]

class ApiResponse(BaseModel):
    result: object
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/        # Unit tests
pytest tests/integration/ # Integration tests
pytest tests/e2e/         # End-to-end tests

# Run with coverage
pytest --cov=src --cov-report=html
```

### Writing Tests

```python
import pytest
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

class TestDataProcessing:
    def test_process_valid_data(self):
        """Test processing valid data."""
        data = {"key": "value"}
        result = process_data(data)

        assert result.is_success
        assert result.unwrap().key == "value"

    def test_process_invalid_data(self):
        """Test processing invalid data."""
        result = process_data(None)

        assert result.is_failure
        assert "Data required" in result.failure()
```

## Quality Gates

### Pre-commit Hooks

FLEXT uses pre-commit hooks to enforce quality standards:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Quality Checks

```bash
# Linting (Ruff)
make lint

# Type checking (MyPy)
make type-check

# Security scanning (Bandit)
make security

# All quality checks
make validate
```

## Adding New Projects

### 1. Create Project Structure

```bash
# Copy from existing project
cp -r flext-api flext-newlib
cd flext-newlib

# Update project metadata
# Edit pyproject.toml, README.md, etc.
```

### 2. Implement Core Patterns

```python
# src/flext_newlib/__init__.py
from flext_core import FlextBus
from flext_core import FlextSettings
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import h
from flext_core import FlextLogger
from flext_core import x
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import p
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import t
from flext_core import u

# Main API class
class FlextNewlib:
    def __init__(self, config: FlextNewlibSettings):
        self.config = config

    def process(self, data: dict) -> FlextResult[dict]:
        """Process data using FlextResult pattern."""
        # Implementation here
        pass

# Models class
class FlextNewlibModels:
    class Config(BaseModel):
        setting: str = "default"

    class Request(BaseModel):
        data: dict[str, object]

    class Response(BaseModel):
        result: FlextResult[object]
```

### 3. Add to Workspace

```bash
# Add to workspace pyproject.toml
# Add to workspace Makefile
# Update documentation
```

## Debugging

### Type Errors

```bash
# Run MyPy with full context
mypy src/module.py --show-error-codes --show-traceback

# Check specific error
mypy src/ --show-error-codes | grep "error-code"
```

### Test Failures

```bash
# Run with verbose output
pytest tests/unit/test_module.py -vv --tb=long

# Debug mode
pytest tests/unit/test_module.py --pdb
```

### Import Issues

```bash
# Verify PYTHONPATH
export PYTHONPATH=src
python -c "import flext_core; print(flext_core.__file__)"

# Check poetry environment
poetry env info
```

## Documentation

### Code Documentation

```python
def process_data(data: dict[str, object]) -> FlextResult[ProcessedData]:
    """
    Process data using the FLEXT pipeline.

    Args:
        data: Input data dictionary

    Returns:
        FlextResult containing processed data or error

    Raises:
        ValidationError: If data validation fails

    Example:
        >>> result = process_data({"key": "value"})
        >>> if result.is_success:
        ...     processed = result.unwrap()
    """
    # Implementation here
```

### README Updates

Update project README.md files when adding new features:

- Add a "New Feature" section with usage and configuration examples.

```python
from flext_newlib import FlextNewlib
from flext_newlib import FlextNewlibSettings

lib = FlextNewlib()
result = lib.new_feature()

config = FlextNewlibSettings(new_setting="value")
```

## Contributing

### Pull Request Process

1. **Fork the repository**
1. **Create a feature branch**
1. **Make your changes**
1. **Run quality gates**
1. **Write tests**
1. **Update documentation**
1. **Submit pull request**

### Code Review Guidelines

- **Follow FLEXT patterns** and architecture
- **Maintain test coverage** above 85%
- **Update documentation** for new features
- **Ensure type safety** with MyPy strict mode
- **Use descriptive commit messages**

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check PYTHONPATH
   export PYTHONPATH=src

   # Reinstall dependencies
   make clean && make setup
   ```

````

2. **Test Failures**

   ```bash
   # Run with debug output
   pytest -vv --tb=long

   # Check specific test
   pytest tests/unit/test_specific.py::test_function -v
````

3. **Build Issues**

   ```bash
   # Clean and rebuild
   make clean-all
   make setup
   make build-all
   ```

## Resources

- FLEXT Core Patterns
- Quality Standards
- Testing Guide
- API Reference
- Examples

## Support

- **Issues**: [GitHub Issues](https://github.com/flext-sh/flext/issues)
- **Discussions**: [GitHub Discussions](https://github.com/flext-sh/flext/discussions)
- **Email**: <dev@flext.com>
