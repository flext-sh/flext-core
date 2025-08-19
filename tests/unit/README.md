# Unit Tests

Isolated component testing for FLEXT Core modules.

## Overview

Unit tests validate individual components in isolation, ensuring correct behavior of all functions, classes, and modules. Tests use mocks and fixtures to eliminate external dependencies.

## Test Organization

```
tests/unit/
├── core/                    # Core framework tests
│   ├── test_result.py      # FlextResult pattern
│   ├── test_container.py   # Dependency injection
│   ├── test_config.py      # Configuration
│   ├── test_utilities.py   # Utility functions
│   └── ...
├── domain/                  # Domain model tests
│   ├── test_entities.py    # Domain entities
│   ├── test_domain_entity.py
│   └── test_domain_value_object.py
├── patterns/                # Pattern tests
│   ├── test_patterns_commands.py
│   ├── test_patterns_validation.py
│   └── test_architectural_patterns.py
└── test_pep8_compliance.py # Code style validation
```

## Running Tests

### Run All Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# With coverage
pytest tests/unit/ --cov=src/flext_core

# Verbose output
pytest tests/unit/ -v

# Parallel execution
pytest tests/unit/ -n auto
```

### Run Specific Categories

```bash
# Core tests only
pytest tests/unit/core/

# Domain tests only
pytest tests/unit/domain/

# Pattern tests only
pytest tests/unit/patterns/

# Single test file
pytest tests/unit/core/test_result.py
```

### Run by Markers

```bash
# Fast tests only
pytest tests/unit/ -m "not slow"

# Core functionality
pytest tests/unit/ -m core

# PEP8 compliance
pytest tests/unit/ -m pep8
```

## Writing Unit Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from flext_core import FlextResult

class TestComponentName:
    """Test suite for ComponentName."""

    @pytest.fixture
    def component(self):
        """Provide component instance for testing."""
        return ComponentName()

    def test_successful_operation(self, component):
        """Test successful operation path."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = component.process(input_data)

        # Assert
        assert result.success
        assert result.unwrap() == expected_value

    def test_error_handling(self, component):
        """Test error handling path."""
        # Arrange
        invalid_input = None

        # Act
        result = component.process(invalid_input)

        # Assert
        assert result.is_failure
        assert "Invalid input" in result.error

    @pytest.mark.parametrize("input,expected", [
        ("value1", "result1"),
        ("value2", "result2"),
        ("value3", "result3"),
    ])
    def test_multiple_scenarios(self, component, input, expected):
        """Test multiple input scenarios."""
        result = component.process(input)
        assert result.unwrap() == expected
```

### Using Mocks

```python
def test_with_external_dependency(mocker):
    """Test component with mocked dependency."""
    # Mock external service
    mock_service = mocker.Mock(spec=ExternalService)
    mock_service.fetch_data.return_value = FlextResult[object].ok({"data": "value"})

    # Inject mock
    component = Component(service=mock_service)

    # Test behavior
    result = component.process()

    # Verify interactions
    mock_service.fetch_data.assert_called_once()
    assert result.success
```

### Testing Async Code

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operation."""
    component = AsyncComponent()
    result = await component.async_process()
    assert result.success
```

## Coverage Requirements

### Target Coverage

- **Minimum**: 75% overall coverage
- **Critical modules**: 90% coverage
    - `result.py`
    - `container.py`
    - `config.py`
- **New code**: 80% minimum

### Check Coverage

```bash
# Generate coverage report
pytest tests/unit/ --cov=src/flext_core --cov-report=term-missing

# HTML report
pytest tests/unit/ --cov=src/flext_core --cov-report=html
open htmlcov/index.html

# Fail if below threshold
pytest tests/unit/ --cov=src/flext_core --cov-fail-under=75
```

## Test Categories

### Core Tests (`core/`)

Tests for fundamental patterns:

- Railway-oriented programming (FlextResult)
- Dependency injection (FlextContainer)
- Configuration management
- Exception handling
- Utility functions

### Domain Tests (`domain/`)

Tests for domain modeling:

- Entities with business logic
- Value objects and immutability
- Aggregate roots and consistency
- Domain services

### Pattern Tests (`patterns/`)

Tests for architectural patterns:

- CQRS commands and queries
- Validation patterns
- Handler chains
- Decorators and mixins

## Best Practices

### DO

- ✅ Test both success and failure paths
- ✅ Use descriptive test names
- ✅ Keep tests focused and simple
- ✅ Use fixtures for common setup
- ✅ Mock external dependencies
- ✅ Test edge cases and boundaries
- ✅ Verify error messages

### DON'T

- ❌ Test implementation details
- ❌ Use real external services
- ❌ Share state between tests
- ❌ Write overly complex tests
- ❌ Ignore flaky tests
- ❌ Test framework code

## Common Patterns

### Testing FlextResult

```python
def test_result_chaining():
    """Test FlextResult chaining operations."""
    result = (
        FlextResult[object].ok(10)
        .map(lambda x: x * 2)
        .flat_map(lambda x: FlextResult[object].ok(x + 5))
    )
    assert result.unwrap() == 25

def test_result_error_propagation():
    """Test error propagation in chain."""
    result = (
        FlextResult[object].ok(10)
        .flat_map(lambda x: FlextResult[object].fail("Error"))
        .map(lambda x: x * 2)  # Should not execute
    )
    assert result.is_failure
    assert result.error == "Error"
```

### Testing Configuration

```python
def test_configuration_loading(monkeypatch):
    """Test configuration from environment."""
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("APP_PORT", "8080")

    config = AppSettings()
    assert config.debug is True
    assert config.port == 8080
```

### Testing Domain Models

```python
def test_entity_business_logic():
    """Test entity enforces business rules."""
    user = User(id="123", email="test@example.com")

    # Test valid operation
    result = user.activate()
    assert result.success
    assert user.is_active

    # Test invalid operation
    result = user.activate()  # Already active
    assert result.is_failure
    assert "already active" in result.error.lower()
```

## Troubleshooting

### Common Issues

**Slow tests:**

```bash
# Find slow tests
pytest tests/unit/ --durations=10

# Skip slow tests
pytest tests/unit/ -m "not slow"
```

**Import errors:**

```bash
# Ensure proper Python path
PYTHONPATH=. pytest tests/unit/
```

**Flaky tests:**

```bash
# Run multiple times to detect flakiness
pytest tests/unit/test_file.py --count=10
```

## Related Documentation

- [Integration Tests](../integration/)
- [E2E Tests](../e2e/)
- [Test Configuration](../conftest.py)
- [Main Test Documentation](../README.md)
