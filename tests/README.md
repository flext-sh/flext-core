# Test Suite Documentation

Comprehensive testing framework for FLEXT Core with 75%+ coverage requirement.

## Overview

The test suite validates FLEXT Core's functionality across unit, integration, and end-to-end scenarios. Tests are organized following Clean Architecture principles and ensure quality for all dependent projects in the FLEXT ecosystem.

## Test Organization

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── conftest_integration.py          # Integration test fixtures
├── shared_test_domain.py            # Shared test models
├── test_shared_domain.py            # Tests for shared domain
│
├── unit/                            # Unit tests (75% coverage target)
│   ├── core/                        # Core framework tests
│   │   ├── test_result.py           # FlextResult pattern
│   │   ├── test_container.py        # Dependency injection
│   │   ├── test_config.py           # Configuration management
│   │   ├── test_config_base.py      # Base configuration
│   │   ├── test_commands.py         # Command patterns
│   │   ├── test_handlers.py         # Handler patterns
│   │   ├── test_validation.py       # Validation system
│   │   ├── test_utilities.py        # Utility functions
│   │   ├── test_payload.py          # Payload patterns
│   │   ├── test_mixins.py           # Mixin behaviors
│   │   ├── test_decorators.py       # Decorator patterns
│   │   ├── test_entities.py         # Domain entities
│   │   ├── test_value_objects.py    # Value objects
│   │   ├── test_aggregate_root.py   # Aggregate roots
│   │   ├── test_interfaces.py       # Interface definitions
│   │   ├── test_loggings.py         # Logging system
│   │   ├── test_semantic.py         # Semantic patterns
│   │   ├── test_schema_processing.py # Schema processing
│   │   ├── test_context.py          # Context management
│   │   ├── test_core.py             # Core functionality
│   │   └── test_observability_simple.py # Observability
│   │
│   ├── domain/                      # Domain model tests
│   │   ├── test_domain_entity.py
│   │   ├── test_domain_value_object.py
│   │   ├── test_domain_services.py
│   │   └── test_entities.py
│   │
│   └── patterns/                    # Pattern tests
│       ├── test_patterns_commands.py
│       ├── test_patterns_validation.py
│       └── test_architectural_patterns.py
│
├── integration/                     # Integration tests
│   ├── test_integration.py
│   └── test_integration_examples.py
│
└── e2e/                             # End-to-end tests
    └── test_complete_workflows.py
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Isolated component testing**

- **Purpose**: Validate individual components work correctly
- **Scope**: Single functions, classes, or modules
- **Dependencies**: Uses mocks and fixtures
- **Execution**: Fast (<100ms per test)
- **Coverage Target**: 75% minimum
- **Examples**:
    - `test_result.py`: FlextResult behavior
    - `test_container.py`: DI container operations
    - `test_config.py`: Configuration validation

### Integration Tests (`tests/integration/`)

**Component interaction testing**

- **Purpose**: Verify components work together
- **Scope**: Multiple modules interacting
- **Dependencies**: Real FLEXT Core components
- **Execution**: Medium (100ms-1s)
- **Focus Areas**:
    - Service integration
    - Container lifecycle
    - Configuration loading
    - Cross-module communication

### End-to-End Tests (`tests/e2e/`)

**Complete workflow validation**

- **Purpose**: Test real-world usage scenarios
- **Scope**: Full application workflows
- **Dependencies**: Complete system
- **Execution**: Slower (>1s)
- **Scenarios**:
    - User registration flow
    - Order processing pipeline
    - Configuration and startup

## Running Tests

### By Test Type

```bash
# Unit tests only
pytest tests/unit -m unit

# Integration tests only
pytest tests/integration -m integration

# End-to-end tests only
pytest tests/e2e -m e2e
```

### By Component

```bash
# Core module tests
pytest tests/unit/core/

# Pattern tests
pytest tests/unit/patterns/

# Domain tests
pytest tests/unit/domain/
```

### By Marker

```bash
# PEP8 compliance tests
pytest -m pep8

# Core functionality tests
pytest -m core

# DDD pattern tests
pytest -m ddd

# Architecture tests
pytest -m architecture
```

### Coverage and Quality

```bash
# Check current coverage
pytest --cov=src/flext_core --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src/flext_core --cov-report=html
open htmlcov/index.html  # View report

# Coverage by module
pytest --cov=src/flext_core.result tests/unit/core/test_result.py

# Fail if below threshold (75%)
pytest --cov=src/flext_core --cov-fail-under=75

# Quick quality check
make test  # Runs tests with coverage requirement
```

## Test Markers

### Primary Markers

| Marker        | Description        | Usage                   |
| ------------- | ------------------ | ----------------------- |
| `unit`        | Unit tests         | `pytest -m unit`        |
| `integration` | Integration tests  | `pytest -m integration` |
| `e2e`         | End-to-end tests   | `pytest -m e2e`         |
| `slow`        | Slow-running tests | `pytest -m "not slow"`  |
| `performance` | Performance tests  | `pytest -m performance` |

### Domain Markers

| Marker         | Description            | Usage                    |
| -------------- | ---------------------- | ------------------------ |
| `core`         | Core functionality     | `pytest -m core`         |
| `ddd`          | Domain-driven design   | `pytest -m ddd`          |
| `architecture` | Architectural patterns | `pytest -m architecture` |
| `pep8`         | PEP8 compliance        | `pytest -m pep8`         |

### Combining Markers

```bash
# Fast unit tests only
pytest -m "unit and not slow"

# Core functionality excluding integration
pytest -m "core and not integration"

# All DDD tests
pytest -m ddd
```

## Test Configuration

### Main Configuration (`conftest.py`)

```python
# Key fixtures available globally
@pytest.fixture
def clean_container() -> FlextContainer:
    """Provides clean container for each test."""
    container = FlextContainer()
    yield container
    container.clear()

@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing."""
    return {
        "app_name": "test_app",
        "debug": True,
        "database_url": "sqlite:///:memory:"
    }

@pytest.fixture
def mock_repository(mocker):
    """Mock repository for testing."""
    return mocker.Mock(spec=Repository)
```

### Shared Test Domain (`shared_test_domain.py`)

```python
# Shared domain models for testing
class TestUser(FlextEntity):
    """Test user entity."""
    username: str
    email: str
    is_active: bool = True

class TestOrder(FlextAggregateRoot):
    """Test order aggregate."""
    customer_id: str
    items: list
    total: Decimal
```

## Writing Tests

### Test Structure (AAA Pattern)

```python
def test_user_activation():
    """Test user activation with AAA pattern."""
    # Arrange
    user = TestUser(
        id="user_123",
        username="testuser",
        email="test@example.com",
        is_active=False
    )

    # Act
    result = user.activate()

    # Assert
    assert result.success
    assert user.is_active is True
    assert "UserActivated" in [e.type for e in user.get_events()]
```

### Testing FlextResult Patterns

```python
def test_railway_pattern():
    """Test railway-oriented error handling."""
    result = (
        FlextResult[object].ok(10)
        .map(lambda x: x * 2)
        .flat_map(lambda x: FlextResult[object].ok(x + 5))
        .map_error(lambda e: f"Error: {e}")
    )

    assert result.success
    assert result.value == 25

def test_error_propagation():
    """Test error propagation in chain."""
    result = (
        FlextResult[object].ok(10)
        .flat_map(lambda x: FlextResult[object].fail("Division error"))
        .map(lambda x: x * 2)  # Should not execute
    )

    assert result.is_failure
    assert result.error == "Division error"
```

### Testing Domain Models

```python
def test_value_object_immutability():
    """Test value objects are immutable."""
    email1 = Email(address="test@example.com")
    email2 = Email(address="test@example.com")

    assert email1 == email2  # Value equality
    assert email1 is not email2  # Different instances

    with pytest.raises(AttributeError):
        email1.address = "new@example.com"  # Should be immutable

def test_entity_business_rules():
    """Test entity enforces business rules."""
    account = BankAccount(
        id="acc_123",
        balance=100.0,
        daily_limit=500.0
    )

    # Test successful withdrawal
    result = account.withdraw(50.0)
    assert result.success
    assert account.balance == 50.0

    # Test overdraft protection
    result = account.withdraw(100.0)
    assert result.is_failure
    assert "Insufficient funds" in result.error
```

### Testing Configuration

```python
def test_configuration_loading():
    """Test configuration loads from environment."""
    os.environ["APP_DEBUG"] = "true"
    os.environ["APP_DATABASE_URL"] = "postgresql://localhost/test"

    try:
        config = AppSettings()
        assert config.debug is True
        assert config.database_url == "postgresql://localhost/test"
    finally:
        # Clean up
        os.environ.pop("APP_DEBUG", None)
        os.environ.pop("APP_DATABASE_URL", None)
```

## Test Dependencies

### Required Dependencies

```toml
# pyproject.toml
[tool.poetry.group.test.dependencies]
pytest = "^8.0.0"
pytest-cov = "^5.0.0"
pytest-mock = "^3.12.0"
pytest-asyncio = "^0.21.0"  # For async tests
pytest-timeout = "^2.2.0"   # Prevent hanging tests
```

### Running Without External Services

All tests are self-contained:

- No database required (uses in-memory when needed)
- No external APIs (uses mocks)
- No network calls (uses fixtures)
- No file system dependencies (uses temp directories)

## Common Test Patterns

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (0, FlextResult[object].fail("Cannot divide by zero")),
    (2, FlextResult[object].ok(5.0)),
    (-2, FlextResult[object].ok(-5.0)),
])
def test_division_scenarios(input, expected):
    """Test division with multiple scenarios."""
    result = divide(10, input)
    if expected.success:
        assert result.success
        assert result.value == expected.value
    else:
        assert result.is_failure
        assert result.error == expected.error
```

### Fixture Composition

```python
@pytest.fixture
def user_service(clean_container, mock_repository):
    """Compose fixtures for service testing."""
    service = UserService(repository=mock_repository)
    clean_container.register("user_service", service)
    return service

def test_user_creation(user_service):
    """Test user creation with composed fixtures."""
    result = user_service.create_user("testuser", "test@example.com")
    assert result.success
```

## Troubleshooting

### Common Issues

**Import Errors**

```bash
# Ensure FLEXT Core is installed
pip install -e .
# Or
PYTHONPATH=. pytest tests/
```

**Coverage Not Meeting Threshold**

```bash
# Find uncovered lines
pytest --cov=src/flext_core --cov-report=term-missing

# Focus on specific module
pytest --cov=src/flext_core.result --cov-report=term-missing tests/unit/core/test_result.py
```

**Slow Tests**

```bash
# Skip slow tests
pytest -m "not slow"

# Find slow tests
pytest --durations=10
```

## Contributing Tests

When adding new tests:

1. **Location**: Place in appropriate directory (unit/integration/e2e)
2. **Naming**: Follow `test_{feature}_{scenario}.py` pattern
3. **Markers**: Add appropriate markers
4. **Documentation**: Include docstrings explaining test purpose
5. **Coverage**: Ensure new code has tests
6. **Quality**: Run `make validate` before committing

---

For more information, see the [Contributing Guide](../CONTRIBUTING.md).
