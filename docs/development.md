# Development Guide

Professional development guide for contributing to FLEXT-Core foundation library.

## Development Environment

### Prerequisites

- **Python 3.13+**: Required for modern type annotations
- **Poetry 1.7+**: Dependency management and virtual environments
- **Git**: Version control
- **Make**: Build automation (included on Linux/macOS, install on Windows)

### Setup

```bash
# Clone repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Install dependencies and setup environment
make setup

# Verify installation
python -c "from flext_core import FlextResult; print('Development environment ready')"
```

## Development Workflow

### Essential Commands

```bash
# Quality gates (run before every commit)
make validate          # Complete validation (lint + type + test + security)
make check            # Quick validation (lint + type only)

# Individual quality checks
make lint             # Code linting with Ruff
make type-check       # Type checking with MyPy
make test             # Run test suite with coverage
make security         # Security audit with Bandit

# Code formatting
make format           # Auto-format code (Ruff + line length 79)

# Development utilities
make clean            # Clean build artifacts
make docs             # Build documentation
make shell            # Python REPL with project loaded
```

### Quality Standards

FLEXT-Core maintains the highest quality standards as the foundation library:

- **Type Safety**: Zero MyPy errors in strict mode
- **Code Quality**: Zero Ruff violations
- **Test Coverage**: 84% minimum (targeting 85%+)
- **Security**: Zero critical vulnerabilities
- **Line Length**: 79 characters maximum (PEP8 strict)

### Pre-commit Hooks

```bash
# Pre-commit hooks are installed automatically with 'make setup'
# They run automatically on git commit:

# Manual execution
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Code Standards

### Import Organization

**ALWAYS use root-level imports** for public APIs:

```python
# ✅ CORRECT - Root module imports
from flext_core import (
    FlextResult,           # Railway pattern
    FlextContainer,        # Dependency injection
    FlextModels,           # Domain models
    FlextConfig,           # Configuration
    FlextLogger,           # Logging
)

# ❌ FORBIDDEN - Internal module imports
from flext_core.result import FlextResult
from flext_core.container import FlextContainer
```

### Function Signatures

All functions must have complete type annotations:

```python
# ✅ CORRECT - Complete type annotations
def process_data(data: dict[str, Any]) -> FlextResult[ProcessedData]:
    """Process input data with explicit error handling."""
    if not data:
        return FlextResult.fail("Data cannot be empty")

    result = ProcessedData(**data)
    return FlextResult.ok(result)

# ❌ INCORRECT - Missing type annotations
def process_data(data):
    # No type information
    pass
```

### Error Handling

**ALWAYS use FlextResult** for operations that can fail:

```python
# ✅ CORRECT - Explicit error handling with FlextResult
def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Safe division with explicit error handling."""
    if b == 0:
        return FlextResult.fail("Division by zero")
    return FlextResult.ok(a / b)

# Usage with proper error handling
result = divide_numbers(10, 2)
if result.success:
    value = result.unwrap()
    print(f"Result: {value}")
else:
    print(f"Error: {result.error}")

# ❌ INCORRECT - Exception-based error handling
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
```

### Class Design

**Single Responsibility** - one class per module with unified purpose:

```python
# ✅ CORRECT - Unified class with single responsibility
class UserService(FlextDomainService):
    """Unified user service with all user-related operations."""

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    class _ValidationHelper:
        """Nested helper class - no loose functions."""
        @staticmethod
        def validate_email(email: str) -> FlextResult[str]:
            if "@" not in email:
                return FlextResult.fail("Invalid email format")
            return FlextResult.ok(email.lower())

    def create_user(self, data: dict) -> FlextResult[User]:
        """Create user with validation and business logic."""
        email_result = self._ValidationHelper.validate_email(data.get("email", ""))
        if email_result.is_failure:
            return email_result.map(lambda _: User())

        # Continue with user creation
        user = User(email=email_result.unwrap(), **data)
        return FlextResult.ok(user)

# ❌ INCORRECT - Multiple classes per module
class UserValidator:  # Should be nested helper
    pass

class UserService:    # Main service
    pass

class UserEmailer:    # Should be injected dependency
    pass
```

### Documentation

All public APIs require docstrings with examples:

```python
def process_user_registration(data: dict) -> FlextResult[User]:
    """Process complete user registration workflow.

    Validates input data, creates user account, and sends welcome email.
    Uses railway-oriented programming for error handling.

    Args:
        data: Registration data containing username, email, password

    Returns:
        FlextResult[User]: Success with User object, or failure with error message

    Example:
        >>> registration_data = {
        ...     "username": "johndoe",
        ...     "email": "john@example.com",
        ...     "password": "securepass123"
        ... }
        >>> result = process_user_registration(registration_data)
        >>> if result.success:
        ...     user = result.unwrap()
        ...     print(f"User {user.username} registered")
        ... else:
        ...     print(f"Registration failed: {result.error}")
    """
    # Implementation here
    pass
```

## Testing

### Test Organization

```
tests/
├── unit/                    # Fast, isolated unit tests
│   ├── test_result.py       # FlextResult railway pattern
│   ├── test_container.py    # Dependency injection
│   └── test_models.py       # Domain models
├── integration/             # Cross-module integration tests
├── performance/             # Performance and load tests
└── conftest.py             # Shared fixtures and utilities
```

### Test Patterns

#### Unit Testing with FlextResult

```python
import pytest
from flext_core import FlextResult

def test_railway_pattern_success():
    """Test successful railway operations."""
    result = FlextResult[str].ok("success")

    assert result.success
    assert result.unwrap() == "success"
    assert result.value == "success"  # New API
    assert result.data == "success"   # Legacy compatibility

def test_railway_pattern_failure():
    """Test failure railway operations."""
    result = FlextResult[str].fail("error occurred")

    assert result.is_failure
    assert result.error == "error occurred"

    with pytest.raises(ValueError):
        result.unwrap()  # Should raise on failure

def test_railway_chaining():
    """Test operation chaining with map and flat_map."""
    def double(x: int) -> int:
        return x * 2

    def safe_divide(x: int, y: int) -> FlextResult[float]:
        if y == 0:
            return FlextResult.fail("Division by zero")
        return FlextResult.ok(x / y)

    result = (
        FlextResult[int].ok(20)
        .map(double)                           # 40
        .flat_map(lambda x: safe_divide(x, 4)) # 10.0
        .map(lambda x: int(x))                 # 10
    )

    assert result.success
    assert result.unwrap() == 10
```

#### Testing with Dependency Injection

```python
def test_service_with_injected_dependencies():
    """Test service using dependency injection."""
    # Setup clean container
    container = FlextContainer()

    # Register mock dependencies
    mock_database = MockDatabase()
    mock_email_service = MockEmailService()

    container.register("database", mock_database)
    container.register("email", mock_email_service)

    # Test service (would use injected dependencies)
    service = UserService()  # Gets dependencies from container

    result = service.create_user({
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    })

    assert result.success
    user = result.unwrap()
    assert user.username == "testuser"

    # Verify mock interactions
    assert mock_database.save_called
    assert mock_email_service.welcome_email_sent
```

#### Testing Domain Models

```python
def test_domain_entity_business_logic():
    """Test entity business logic and domain events."""
    from datetime import datetime

    user = User(
        id="user_123",
        username="testuser",
        email="test@example.com",
        created_at=datetime.now(),
        is_active=False
    )

    # Test business operation
    activation_result = user.activate()
    assert activation_result.success
    assert user.is_active

    # Test domain events
    events = user.get_domain_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "UserActivated"
    assert events[0]["data"]["user_id"] == "user_123"

    # Test business rule (cannot activate twice)
    second_activation = user.activate()
    assert second_activation.is_failure
    assert "already active" in second_activation.error.lower()
```

### Test Coverage

Run tests with coverage reporting:

```bash
# Full test suite with coverage
make test

# Run specific test file
pytest tests/unit/test_result.py -v

# Coverage report
pytest --cov=src --cov-report=term-missing --cov-report=html

# Coverage by module
pytest --cov=src/flext_core/result.py --cov-report=term
```

**Coverage Target**: 84% minimum (currently achieved), targeting 85%+

## Contributing

### Pull Request Process

1. **Fork and Clone**
   ```bash
   git fork https://github.com/flext-sh/flext-core.git
   git clone https://github.com/yourusername/flext-core.git
   cd flext-core
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Develop with Quality Gates**
   ```bash
   # Make changes
   # Run quality checks frequently
   make check    # Quick validation
   make validate # Full validation
   ```

4. **Test Thoroughly**
   ```bash
   # Run full test suite
   make test

   # Test your specific changes
   pytest tests/unit/test_your_feature.py -v
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Describe changes clearly
   - Include test coverage information
   - Reference any related issues

### Code Review Checklist

Before submitting, ensure:

- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`make type-check`)
- [ ] Linting passes (`make lint`)
- [ ] Security audit passes (`make security`)
- [ ] Code coverage is maintained or improved
- [ ] All public APIs have docstrings with examples
- [ ] Changes follow established patterns
- [ ] No breaking changes to existing APIs

### Foundation Library Responsibilities

As FLEXT-Core is the foundation for 33+ ecosystem projects:

1. **API Stability**: No breaking changes without deprecation cycle
2. **Backward Compatibility**: Maintain .data/.value dual access
3. **Quality Leadership**: Set quality standards for ecosystem
4. **Documentation**: All public APIs fully documented
5. **Performance**: Efficient implementation for high-volume usage

### Ecosystem Impact Testing

Before major changes, test impact on dependent projects:

```bash
# Test core imports work across ecosystem
for project in ../flext-api ../flext-cli ../flext-auth; do
    if [ -d "$project" ]; then
        echo "Testing $project compatibility..."
        cd "$project"
        python -c "from flext_core import FlextResult, FlextContainer; print('✅ OK')"
        cd - > /dev/null
    fi
done

# Test API compatibility patterns
python -c "
from flext_core import FlextResult
result = FlextResult[str].ok('test')
assert hasattr(result, 'data'), '.data API missing (ecosystem breaking)'
assert hasattr(result, 'value'), '.value API missing'
assert result.data == result.value, 'API consistency broken'
print('✅ API compatibility confirmed')
"
```

## Debugging

### Development REPL

```bash
# Start interactive shell with project loaded
make shell

# In the Python shell:
>>> from flext_core import *
>>> result = FlextResult[str].ok("test")
>>> result.success
True
>>> container = FlextContainer.get_global()
>>> container.register("test", "value")
<FlextResult success=True>
```

### Debugging Type Issues

```bash
# Run MyPy with verbose output
mypy src/ --strict --show-error-codes --show-traceback

# Check specific file
mypy src/flext_core/result.py --strict --show-error-codes

# Generate type coverage report
mypy src/ --html-report mypy-report --show-error-codes
```

### Performance Profiling

```python
import cProfile
from flext_core import FlextResult

def benchmark_railway_operations():
    """Benchmark railway pattern operations."""
    for i in range(10000):
        result = (
            FlextResult[int].ok(i)
            .map(lambda x: x * 2)
            .flat_map(lambda x: FlextResult.ok(x + 1))
            .map(lambda x: str(x))
        )
        value = result.unwrap()

# Profile the function
cProfile.run('benchmark_railway_operations()', 'profile_results')
```

## Documentation

### Building Documentation

```bash
# Build documentation locally
make docs

# Serve documentation (if available)
make docs-serve  # Available at http://localhost:8000
```

### Documentation Standards

- All public APIs require complete docstrings
- Include usage examples in docstrings
- Use Google-style docstring format
- Document error conditions and return types
- Provide practical examples

## Release Process

### Version Management

```python
# Update version in src/flext_core/version.py
__version__ = "0.9.1"

class FlextVersionManager:
    VERSION_MAJOR = 0
    VERSION_MINOR = 9
    VERSION_PATCH = 1
```

### Release Checklist

1. **Quality Validation**
   ```bash
   make validate  # All checks must pass
   ```

2. **Test Coverage**
   ```bash
   make test
   # Ensure coverage is 84%+
   ```

3. **Documentation Update**
   - Update CHANGELOG.md
   - Update README.md version references
   - Update documentation version numbers

4. **Ecosystem Compatibility**
   - Test core imports across dependent projects
   - Verify API compatibility maintained
   - Test no regressions in dependent project builds

5. **Release**
   ```bash
   git tag v0.9.1
   git push origin v0.9.1
   make build  # If building distributions
   ```

---

**Development Complete**: Follow these standards to maintain FLEXT-Core as a solid foundation for the entire FLEXT ecosystem. Quality and compatibility are paramount.