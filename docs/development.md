# Development Guide for FLEXT-Core

Complete development workflow guide for contributing to FLEXT-Core v0.9.9 - the foundation library for the FLEXT ecosystem with strict quality standards and comprehensive testing requirements.

---

## Development Environment

### Prerequisites

- **Python**: 3.13+ (required - uses latest syntax features)
- **Poetry**: Latest version for dependency management
- **Git**: For version control
- **Make**: For automation commands (available on all Unix systems)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Complete setup with pre-commit hooks
make setup

# Or install dependencies only
make install

# Verify installation
python -c "from flext_core import FlextResult; print('✅ FLEXT-Core ready')"
```

**What `make setup` does**:

- Installs Poetry if not present
- Creates virtual environment
- Installs all dependencies (dev, test, security, typings)
- Installs pre-commit hooks
- Runs initial validation

---

## Development Workflow

### Daily Development Commands

```bash
# Format code (auto-fix style issues)
make format              # Ruff formatter (79 char line limit)

# Lint code (check for issues)
make lint               # Ruff linting with comprehensive rules

# Type checking
make type-check         # MyPy strict mode + PyRight validation

# Run tests
make test              # Full test suite with coverage
make test-unit         # Unit tests only (fast feedback)
make test-integration  # Integration tests only
make test-fast         # Tests without coverage (quick iteration)

# Security checks
make security          # Bandit + pip-audit security scanning

# Complete validation (MANDATORY before commits)
make validate          # Runs lint + type-check + security + test
make check             # Quick validation (lint + type-check only)

# Single letter aliases for speed
make f                 # Format
make l                 # Lint
make tc                # Type check
make t                 # Test
make v                 # Validate
make c                 # Clean
```

### Pre-Commit Checklist

**MANDATORY before every commit**:

```bash
# 1. Format code
make format

# 2. Run complete validation
make validate

# 3. Check git status
git status

# 4. Commit changes
git add .
git commit -m "Your commit message"
```

**Quality Gates** (all must pass):

- ✅ Ruff: ZERO violations in `src/`
- ✅ MyPy: ZERO errors in `src/` (strict mode)
- ✅ PyRight: ZERO errors (secondary validation)
- ✅ Tests: ALL 1,163+ tests passing
- ✅ Coverage: Maintain or improve from 75% baseline

---

## Code Quality Standards

### Type Safety Requirements

**MANDATORY for all code in `src/`**:

```python
# ✅ CORRECT - Complete type annotations
from flext_core import FlextResult

def process_data(input_data: FlextTypes.Dict) -> FlextResult[ProcessedData]:
    """Process input data with type-safe error handling."""
    if not input_data:
        return FlextResult[ProcessedData].fail("Input cannot be empty")

    # Processing logic
    result = ProcessedData(**input_data)
    return FlextResult[ProcessedData].ok(result)

# ❌ FORBIDDEN - Missing type annotations
def process_data(input_data):  # No type hints
    # Processing logic
    return result  # Unclear return type
```

### Line Length and Formatting

**PEP 8 Strict Compliance**:

- **Line Length**: 79 characters maximum
- **Imports**: Sorted alphabetically, grouped by standard/third-party/local
- **Strings**: Double quotes preferred
- **Indentation**: 4 spaces (no tabs)

```python
# ✅ CORRECT - Proper formatting
from flext_core import (
    FlextContainer,
    FlextLogger,
    FlextResult,
)

def create_service(
    config: AppConfig,
    logger: FlextLogger,
) -> FlextResult[MyService]:
    """Create service with dependencies.

    Args:
        config: Application configuration
        logger: Logger instance

    Returns:
        FlextResult containing service instance or error
    """
    if not config.api_key:
        return FlextResult[MyService].fail(
            "API key required"
        )

    service = MyService(config=config, logger=logger)
    return FlextResult[MyService].ok(service)
```

### Documentation Requirements

**ALL public APIs must have docstrings**:

```python
from flext_core import FlextResult

class UserService:
    """User management service with validation.

    This service handles user operations including creation,
    validation, and lifecycle management. All operations
    return FlextResult for explicit error handling.

    Example:
        >>> service = UserService()
        >>> result = service.create_user("alice@example.com")
        >>> if result.is_success:
        ...     user = result.unwrap()
    """

    def create_user(self, email: str) -> FlextResult[User]:
        """Create a new user with email validation.

        Args:
            email: User email address (must contain @)

        Returns:
            FlextResult containing User instance on success,
            or error message on validation failure.

        Example:
            >>> result = service.create_user("user@example.com")
            >>> if result.is_success:
            ...     user = result.unwrap()
            ...     print(f"Created: {user.email}")
        """
        if "@" not in email:
            return FlextResult[User].fail("Invalid email format")

        user = User(email=email)
        return FlextResult[User].ok(user)
```

---

## Testing Standards

### Test Organization

```
tests/
├── unit/              # Unit tests (isolated component testing)
│   ├── test_result.py           # FlextResult tests
│   ├── test_container.py        # FlextContainer tests
│   ├── test_models.py           # FlextModels tests
│   └── ... (20+ test modules)
├── integration/       # Integration tests (component interaction)
│   ├── test_config_singleton_integration.py
│   ├── test_service.py
│   └── test_wildcard_exports.py
├── patterns/          # Pattern tests (CQRS, DDD, architectural)
│   ├── test_patterns.py
│   ├── test_patterns_commands.py
│   └── test_advanced_patterns.py
└── conftest.py        # Shared fixtures and configuration
```

### Writing Tests

**Use flext_tests infrastructure** (no mocks, real Docker):

```python
from flext_core import FlextResult, FlextContainer
from flext_tests import UserFactory, FlextMatchers
import pytest

class TestUserService:
    """User service test suite."""

    def test_create_user_success(self, clean_container):
        """Test successful user creation."""
        # Arrange
        service = UserService()
        email = "test@example.com"

        # Act
        result = service.create_user(email)

        # Assert
        assert result.is_success
        user = result.unwrap()
        assert user.email == email

    def test_create_user_invalid_email(self):
        """Test user creation with invalid email."""
        # Arrange
        service = UserService()
        invalid_email = "not-an-email"

        # Act
        result = service.create_user(invalid_email)

        # Assert
        assert result.is_failure
        assert "Invalid email" in result.error

    @pytest.mark.parametrize("email", [
        "user@example.com",
        "test.user@domain.co.uk",
        "user+tag@example.org",
    ])
    def test_create_user_valid_formats(self, email: str):
        """Test user creation with various valid email formats."""
        service = UserService()
        result = service.create_user(email)
        assert result.is_success
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_result.py -v

# Run specific test class
PYTHONPATH=src pytest tests/unit/test_result.py::TestFlextResult -v

# Run specific test method
PYTHONPATH=src pytest tests/unit/test_result.py::TestFlextResult::test_ok -v

# Run with markers
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Exclude slow tests

# Run with coverage for specific module
PYTHONPATH=src pytest tests/unit/test_result.py --cov=src/flext_core/result.py --cov-report=term-missing

# Run last failed tests
pytest --lf --ff -x

# Run in parallel (faster)
pytest -n auto tests/unit/

# Run with verbose output and short traceback
pytest tests/ -v --tb=short
```

### Coverage Requirements

**Current State**: 75% coverage (9,392 statements flext_core, 2,382 missed)
**Baseline Achieved**: 75% coverage target met ✅
**1.0.0 Target**: 79% coverage minimum

**Priority Modules** (need improvement):

- `dispatcher.py`: 45% → 75%+ (298 lines missed)
- `processors.py`: 56% → 75%+ (195 lines missed)
- `mixins.py`: 57% → 75%+ (178 lines missed)
- `exceptions.py`: 62% → 75%+ (182 lines missed)
- `models.py`: 65% → 75%+ (484 lines missed)
- `context.py`: 66% → 75%+ (174 lines missed)
- `handlers.py`: 66% → 75%+ (111 lines missed)
- `utilities.py`: 66% → 75%+ (379 lines missed)

**Coverage Analysis**:

```bash
# Generate coverage report
make test

# Generate HTML report for detailed analysis
make coverage-html

# View module-by-module breakdown
pytest tests/ --cov=src --cov-report=term-missing | grep "src/flext_core"
```

---

## Git Workflow

### Branch Strategy

```bash
# Create feature branch
git checkout -b feature/add-new-validator

# Create bugfix branch
git checkout -b fix/result-unwrap-error

# Create documentation branch
git checkout -b docs/update-api-reference
```

### Commit Message Format

**Format**: `<type>: <subject>`

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build/tooling changes

**Examples**:

```bash
git commit -m "feat: add email validation to FlextModels.User"
git commit -m "fix: handle None values in FlextResult.unwrap()"
git commit -m "docs: update FlextConfig examples in getting-started.md"
git commit -m "test: add coverage for FlextDispatcher error paths"
git commit -m "refactor: simplify FlextContainer registration logic"
```

### Pre-Commit Hooks

**Automatically run** on every commit:

- Ruff formatting
- Ruff linting
- MyPy type checking
- Trailing whitespace removal
- End of file fixer

**If hooks fail**:

```bash
# Fix issues automatically
make format

# Run validation
make validate

# Try commit again
git commit -m "Your message"
```

**Skip hooks** (only when necessary):

```bash
# Not recommended - only for WIP commits
git commit --no-verify -m "WIP: partial implementation"
```

---

## Quality Assurance

### Mandatory QA Order

**ALWAYS follow this order**:

```bash
# 1. Format code (auto-fix style)
make format

# 2. Lint check (catch issues)
make lint

# 3. Type check (validate types)
make type-check

# 4. Security check (vulnerabilities)
make security

# 5. Run tests (validate functionality)
make test

# Or run all at once
make validate
```

### Quality Gate Requirements

**ALL must pass for merge**:

| Check           | Tool               | Requirement             | Current       |
| --------------- | ------------------ | ----------------------- | ------------- |
| **Formatting**  | Ruff format        | 100% compliant          | ✅ Pass       |
| **Linting**     | Ruff check         | Zero violations in src/ | ✅ Pass       |
| **Type Safety** | MyPy strict        | Zero errors in src/     | ✅ Pass       |
| **Type Safety** | PyRight            | Zero errors (secondary) | ✅ Pass       |
| **Security**    | Bandit + pip-audit | Zero critical issues    | ✅ Pass       |
| **Tests**       | Pytest             | All tests passing       | ✅ 1,163 pass |
| **Coverage**    | pytest-cov         | Maintain/improve 75%+   | ✅ 75%        |

### Continuous Integration

**GitHub Actions** (runs on every push):

```yaml
# .github/workflows/qa.yml workflow runs:
- Python 3.13 tests
- Complete validation (lint + type + test)
- Coverage reporting
- Security scanning
- Documentation build
```

**PR Requirements**:

- ✅ All CI checks passing
- ✅ Code review approved
- ✅ Coverage maintained or improved
- ✅ Documentation updated
- ✅ CHANGELOG.md updated (for notable changes)

---

## Development Best Practices

### 1. Railway-Oriented Programming

**ALWAYS use FlextResult for error handling**:

```python
# ✅ CORRECT - Explicit error handling
def process_user(data: dict) -> FlextResult[User]:
    """Process user data with explicit error handling."""
    if not data.get("email"):
        return FlextResult[User].fail("Email required")

    # Railway-oriented composition
    return (
        validate_email(data["email"])
        .flat_map(lambda email: create_user(email))
        .map(lambda user: enrich_user_data(user))
        .map_error(lambda e: f"User processing failed: {e}")
    )

# ❌ FORBIDDEN - Exception-based error handling
def process_user(data: dict) -> User:
    """Don't use exceptions for expected failures."""
    try:
        if not data.get("email"):
            raise ValueError("Email required")
        # Processing logic
        return user
    except ValueError as e:
        # Exception handling for business logic
        return None  # Unclear failure mode
```

### 2. Dependency Injection

**Use FlextContainer.get_global() pattern**:

```python
from flext_core import FlextContainer, FlextLogger

class MyService:
    """Service using dependency injection."""

    def __init__(self) -> None:
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    def process(self, data: dict) -> FlextResult[FlextTypes.Dict]:
        """Process with injected dependencies."""
        # Get configuration from container
        config_result = self._container.get("config")
        if config_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail("Config unavailable")

        config = config_result.unwrap()
        self._logger.info("Processing", extra={"data": data})

        # Use config and process
        return FlextResult[FlextTypes.Dict].ok({"result": "processed"})
```

### 3. Type Safety

**Complete type annotations required**:

```python
from typing import object
from flext_core import FlextResult

# ✅ CORRECT - Complete annotations
def transform_data(
    input_data: FlextTypes.Dict,
    options: FlextTypes.BoolDict | None = None
) -> FlextResult[FlextTypes.Dict]:
    """Transform data with optional configuration."""
    opts = options or {}
    transformed = {k: str(v) for k, v in input_data.items()}
    return FlextResult[FlextTypes.Dict].ok(transformed)

# ❌ FORBIDDEN - Missing annotations
def transform_data(input_data, options=None):
    """Missing type annotations."""
    opts = options or {}
    transformed = {k: str(v) for k, v in input_data.items()}
    return FlextResult.ok(transformed)  # What type is this?
```

### 4. Documentation

**Document all public APIs**:

```python
from flext_core import FlextResult

def validate_and_process(
    data: FlextTypes.Dict,
    strict: bool = False
) -> FlextResult[FlextTypes.Dict]:
    """Validate and process input data.

    This function validates the input data structure and
    processes it according to business rules. In strict mode,
    additional validation checks are performed.

    Args:
        data: Input data dictionary with required 'id' key
        strict: If True, perform additional validation checks

    Returns:
        FlextResult containing processed data on success,
        or validation error message on failure.

    Example:
        >>> result = validate_and_process({"id": "123"})
        >>> if result.is_success:
        ...     processed = result.unwrap()
        ...     print(f"Processed: {processed['id']}")

    Note:
        Strict mode requires 'email' field in input data.
    """
    if "id" not in data:
        return FlextResult[FlextTypes.Dict].fail("ID required")

    if strict and "email" not in data:
        return FlextResult[FlextTypes.Dict].fail("Email required in strict mode")

    # Processing logic
    processed = {"id": data["id"], "status": "processed"}
    return FlextResult[FlextTypes.Dict].ok(processed)
```

---

## Troubleshooting Development Issues

### Import Errors

```bash
# If imports fail, check PYTHONPATH
export PYTHONPATH=src
python -c "from flext_core import FlextResult"

# Or use pytest with PYTHONPATH
PYTHONPATH=src pytest tests/unit/test_result.py
```

### Type Checking Errors

```bash
# Run MyPy with verbose output
mypy src/ --strict --show-error-codes

# Run PyRight for additional validation
pyright src/ --outputformat text

# Check specific file
mypy src/flext_core/result.py --strict --show-error-codes
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -v --tb=short

# Run specific failing test
pytest tests/unit/test_result.py::TestFlextResult::test_ok -xvs

# Run with full traceback
pytest tests/unit/test_result.py::TestFlextResult::test_ok --tb=long

# Run last failed tests only
pytest --lf --ff -x
```

### Coverage Issues

```bash
# Generate detailed coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report for analysis
pytest tests/ --cov=src --cov-report=html:coverage-report

# View in browser
python -m http.server -d coverage-report
```

### Pre-Commit Hook Issues

```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files

# Clear cache if issues persist
pre-commit clean
pre-commit install
```

---

## IDE Configuration

### VS Code

**Recommended settings** (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": ["--strict"],
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.rulers": [79],
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"]
}
```

### PyCharm

**Recommended settings**:

- Enable MyPy plugin
- Set line length to 79
- Enable "Reformat on save"
- Configure Ruff as external tool
- Enable pytest as test runner

---

## Release Process

### Version Bumping

**Update version** in multiple locations:

```bash
# 1. pyproject.toml
[project]
version = "0.9.10"

# 2. src/flext_core/version.py
__version__ = "0.9.10"

# 3. CHANGELOG.md
## [0.9.10] - 2025-XX-XX
### Added
- New feature description
```

### Release Checklist

- [ ] All tests passing
- [ ] Coverage maintained or improved
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all locations
- [ ] Git tag created: `git tag v0.9.10`
- [ ] GitHub release created
- [ ] PyPI package published (if applicable)

---

## Getting Help

### Resources

- **Documentation**: Browse `docs/` directory
- **Examples**: Check `examples/` for working code
- **Tests**: Look at `tests/` for usage patterns
- **Issues**: Report issues on GitHub
- **Discussions**: Use GitHub Discussions for questions

### Common Commands Reference

```bash
# Setup and installation
make setup                # Complete setup with hooks
make install              # Install dependencies only

# Quality checks
make format              # Auto-format code
make lint                # Check linting
make type-check          # Check types
make security            # Security scan
make test                # Run all tests
make validate            # Complete validation

# Testing
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-fast           # Tests without coverage
make coverage-html       # Generate HTML coverage

# Utilities
make clean              # Clean build artifacts
make clean-all          # Deep clean including venv
make reset              # Full reset (clean + setup)
make doctor             # Health check
make diagnose           # Environment diagnostics
make shell              # Python REPL with project

# Single letter aliases
make f l tc t v c i     # format lint type-check test validate clean install
```

---

**FLEXT-Core Development** - Strict quality standards, comprehensive testing, and professional development practices for foundation library excellence.
