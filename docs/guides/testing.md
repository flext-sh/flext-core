<!-- Generated from docs/guides/testing.md for flext-core. -->

<!-- Source of truth: workspace docs/guides/. -->

# flext-core - FLEXT Testing Guide

> Project profile: `flext-core`

<!-- TOC START -->

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [End-to-End Tests](#end-to-end-tests)
- [Test Markers](#test-markers)
- [Running Tests](#running-tests)
  - [Basic Test Execution](#basic-test-execution)
  - [Coverage Analysis](#coverage-analysis)
  - [Parallel Test Execution](#parallel-test-execution)
- [Test Fixtures](#test-fixtures)
  - [Pytest Fixtures](#pytest-fixtures)
  - [Using Fixtures](#using-fixtures)
- [Mocking and Stubbing](#mocking-and-stubbing)
  - [Unit Test Mocking](#unit-test-mocking)
  - [Integration Test Stubbing](#integration-test-stubbing)
- [Performance Testing](#performance-testing)
  - [Load Testing](#load-testing)
  - [Memory Testing](#memory-testing)
- [Test Data Management](#test-data-management)
  - [Test Fixtures Directory](#test-fixtures-directory)
  - [Loading Test Data](#loading-test-data)
- [Continuous Integration](#continuous-integration)
  - [GitHub Actions Workflow](#github-actions-workflow)
- [Best Practices](#best-practices)
  - [1. Test Naming](#1-test-naming)
  - [2. Test Organization](#2-test-organization)
  - [3. Assertion Quality](#3-assertion-quality)
  - [4. Test Independence](#4-test-independence)
- [Troubleshooting](#troubleshooting)
  - [Common Test Issues](#common-test-issues)
- [Resources](#resources)

<!-- TOC END -->

This guide covers testing strategies, best practices, and procedures for FLEXT applications and libraries.

## Overview

FLEXT maintains comprehensive test coverage across all **33 projects** with the following standards:

- **85%+ coverage** for foundation libraries (flext-core)
- **75%+ coverage** for applications and domain libraries
- **100% test pass rate** across all projects
- **Zero Pyrefly errors** in strict mode (successor to MyPy)
- **Zero Ruff violations** in production code

## Test Structure

FLEXT uses a hierarchical test structure:

````text
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (component interaction)
├── e2e/           # End-to-end tests (full workflow)
├── fixtures/      # Test data and fixtures
└── conftest.py    # Pytest configuration
```text

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:

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
from flext_ldif import FlextLdif

class TestLdifParsing:
    def test_parse_valid_ldif(self):
        """Test parsing valid LDIF content."""
        ldif = FlextLdif()
        content = """dn: cn=test,dc=example,dc=com
cn: test
objectClass: inetOrgPerson"""

        result = ldif.parse(content)

        assert result.is_success
        entries = result.unwrap()
        assert len(entries) == 1
        assert entries[0].dn == "cn=test,dc=example,dc=com"

    def test_parse_invalid_ldif(self):
        """Test parsing invalid LDIF content."""
        ldif = FlextLdif()
        content = "invalid ldif content"

        result = ldif.parse(content)

        assert result.is_failure
        assert "parsing" in str(result.failure()).lower()
```text

### Integration Tests

Test component interactions and workflows:

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
from flext_ldif import FlextLdif, FlextLdifSettings

class TestLdifIntegration:
    def test_ldif_with_container(self):
        """Test LDIF processing with dependency injection."""
        container = FlextContainer.get_global()

        # Register LDIF service
        config = FlextLdifSettings(batch_size=100)
        ldif = FlextLdif(config=config)
        container.register("ldif", ldif)

        # Retrieve and use service
        ldif_result = container.get("ldif")
        assert ldif_result.is_success

        ldif_service = ldif_result.unwrap()
        # Test LDIF operations
        result = ldif_service.parse("dn: test")
        assert result.is_success
```text

### End-to-End Tests

Test complete workflows and user scenarios:

```python
import pytest
from pathlib import Path
from flext_ldif import FlextLdif, FlextLdifSettings

class TestLdifMigration:
    def test_oid_to_oud_migration(self):
        """Test complete OID to OUD migration workflow."""
        # Setup test data
        input_dir = Path("test_data/oid")
        output_dir = Path("test_data/oud")

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create sample LDIF file
        sample_ldif = """dn: cn=test,dc=example,dc=com
cn: test
objectClass: inetOrgPerson"""

        with open(input_dir / "test.ldif", 'w') as f:
            f.write(sample_ldif)

        # Configure and run migration
        config = FlextLdifSettings(
            source_server="oid",
            target_server="oud",
            preserve_oid_modifiers=True
        )

        ldif = FlextLdif(config=config)
        result = ldif.migrate(input_dir, output_dir, "oid", "oud")

        # Verify migration
        assert result.is_success
        report = result.unwrap()
        assert report.successful_entries > 0
        assert (output_dir / "test.ldif").exists()
```text

## Test Markers

FLEXT uses pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """Unit test - fast and isolated."""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test - component interaction."""
    pass

@pytest.mark.e2e
def test_end_to_end_scenario():
    """End-to-end test - complete workflow."""
    pass

@pytest.mark.slow
def test_performance_benchmark():
    """Slow test - performance or load testing."""
    pass
```text

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only
pytest tests/e2e/         # End-to-end tests only

# Run with markers
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests
pytest -m "not slow"     # Skip slow tests
```text

### Coverage Analysis

Coverage thresholds and source directories are configured in each project's `pyproject.toml` under `[tool.coverage]`. Use `make test` which reads these automatically.

```bash
# Run with coverage (reads [tool.coverage] from pyproject.toml)
make test

# HTML coverage report
pytest --cov --cov-report=html
```text

### Parallel Test Execution

```bash
# Run tests in parallel
pytest -n auto

# Specific number of workers
pytest -n 4
```text

## Test Fixtures

### Pytest Fixtures

```python
import pytest
from pathlib import Path
from flext_ldif import FlextLdif, FlextLdifSettings

@pytest.fixture
def ldif_config():
    """Provide LDIF configuration for tests."""
    return FlextLdifSettings(
        batch_size=10,
        strict_validation=False
    )

@pytest.fixture
def ldif_service(ldif_config):
    """Provide LDIF service instance."""
    return FlextLdif(config=ldif_config)

@pytest.fixture
def sample_ldif_content():
    """Provide sample LDIF content for tests."""
    return """dn: cn=test,dc=example,dc=com
cn: test
sn: user
objectClass: inetOrgPerson"""

@pytest.fixture
def temp_directories(tmp_path):
    """Provide temporary directories for file tests."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()
    output_dir.mkdir()

    return input_dir, output_dir
```text

### Using Fixtures

```python
def test_ldif_parsing(ldif_service, sample_ldif_content):
    """Test LDIF parsing with fixtures."""
    result = ldif_service.parse(sample_ldif_content)
    assert result.is_success

def test_file_migration(ldif_service, temp_directories):
    """Test file migration with temporary directories."""
    input_dir, output_dir = temp_directories

    # Create test file
    test_file = input_dir / "test.ldif"
    test_file.write_text("dn: test")

    # Run migration
    result = ldif_service.migrate(input_dir, output_dir, "oid", "oud")
    assert result.is_success
```text

## Mocking and Stubbing

### Unit Test Mocking

```python
from unittest.mock import Mock, patch
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

def test_with_mocked_dependency():
    """Test with mocked external dependency."""
    with patch('flext_ldif.external_service') as mock_service:
        # Configure mock
        mock_service.process.return_value = FlextResult.ok("processed")

        # Test function that uses mock
        result = my_function()

        # Verify mock was called
        mock_service.process.assert_called_once()
        assert result.is_success
```text

### Integration Test Stubbing

```python
from unittest.mock import Mock
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

def test_with_stubbed_service():
    """Test with stubbed service in container."""
    container = FlextContainer.get_global()

    # Create stub service
    stub_service = Mock()
    stub_service.process.return_value = FlextResult.ok("stubbed")

    # Register stub
    container.register("external_service", stub_service)

    # Test integration
    result = integration_function()
    assert result.is_success
```text

## Performance Testing

### Load Testing

```python
import pytest
import time
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.slow
def test_concurrent_processing():
    """Test concurrent processing performance."""
    ldif = FlextLdif()
    content = "dn: test\ncn: test"

    def process_entry():
        return ldif.parse(content)

    # Run concurrent processing
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_entry) for _ in range(100)]
        results = [future.result() for future in futures]

    end_time = time.time()

    # Verify all succeeded
    assert all(result.is_success for result in results)

    # Verify performance (should complete in < 1 second)
    assert (end_time - start_time) < 1.0
```text

### Memory Testing

```python
import pytest
import psutil
import os

@pytest.mark.slow
def test_memory_usage():
    """Test memory usage during large file processing."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Process large dataset
    ldif = FlextLdif()
    large_content = "dn: test\ncn: test\n" * 10000

    result = ldif.parse(large_content)
    assert result.is_success

    # Check memory usage (should not exceed 100MB)
    current_memory = process.memory_info().rss
    memory_used = current_memory - initial_memory

    assert memory_used < 100 * 1024 * 1024  # 100MB
```text

## Test Data Management

### Test Fixtures Directory

```text
tests/
├── fixtures/
│   ├── ldif/
│   │   ├── valid.ldif
│   │   ├── invalid.ldif
│   │   └── large.ldif
│   ├── config/
│   │   ├── dev.yaml
│   │   └── prod.yaml
│   └── data/
│       ├── users.json
│       └── schema.json
```text

### Loading Test Data

```python
import json
from pathlib import Path

def load_test_fixture(fixture_name: str) -> str:
    """Load test fixture from fixtures directory."""
    fixture_path = Path(__file__).parent / "fixtures" / fixture_name
    return fixture_path.read_text()

def load_json_fixture(fixture_name: str) -> dict[str, object]:
    """Load JSON test fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / fixture_name
    return json.loads(fixture_path.read_text())

# Usage
def test_with_fixture():
    """Test using loaded fixture data."""
    ldif_content = load_test_fixture("ldif/valid.ldif")
    config_data = load_json_fixture("config/dev.yaml")

    # Use fixture data in test
    result = process_ldif(ldif_content, config_data)
    assert result.is_success
```text

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.13]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run tests
        run: |
          poetry run pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```text

## Best Practices

### 1. Test Naming

```python
# ✅ GOOD - Descriptive test names
def test_parse_valid_ldif_returns_success():
    """Test that parsing valid LDIF returns success result."""
    pass

def test_parse_invalid_ldif_returns_failure():
    """Test that parsing invalid LDIF returns failure result."""
    pass

# ❌ BAD - Vague test names
def test_parse():
    pass

def test_ldif():
    pass
```text

### 2. Test Organization

```python
class TestLdifParsing:
    """Test LDIF parsing functionality."""

    def test_parse_valid_single_entry(self):
        """Test parsing single valid LDIF entry."""
        pass

    def test_parse_valid_multiple_entries(self):
        """Test parsing multiple valid LDIF entries."""
        pass

    def test_parse_invalid_format(self):
        """Test parsing invalid LDIF format."""
        pass

class TestLdifMigration:
    """Test LDIF migration functionality."""

    def test_migrate_oid_to_oud(self):
        """Test OID to OUD migration."""
        pass
```text

### 3. Assertion Quality

```python
# ✅ GOOD - Specific assertions
def test_parse_result():
    result = ldif.parse(content)

    assert result.is_success
    entries = result.unwrap()
    assert len(entries) == 1
    assert entries[0].dn == "cn=test,dc=example,dc=com"
    assert "cn" in entries[0].attributes

# ❌ BAD - Vague assertions
def test_parse_result():
    result = ldif.parse(content)
    assert result  # Too vague
```text

### 4. Test Independence

```python
# ✅ GOOD - Independent tests
def test_parse_valid_ldif():
    ldif = FlextLdif()  # Fresh instance
    result = ldif.parse("dn: test")
    assert result.is_success

def test_parse_invalid_ldif():
    ldif = FlextLdif()  # Fresh instance
    result = ldif.parse("invalid")
    assert result.is_failure

# ❌ BAD - Dependent tests
ldif = FlextLdif()  # Shared instance

def test_parse_valid_ldif():
    result = ldif.parse("dn: test")
    assert result.is_success

def test_parse_invalid_ldif():
    result = ldif.parse("invalid")
    assert result.is_failure
```text

## Troubleshooting

### Common Test Issues

1. **Import Errors**

   ```bash
   # Set PYTHONPATH
   export PYTHONPATH=src
   pytest
```text

1. **Fixture Not Found**

   ```python
   # Check fixture scope and dependencies
   @pytest.fixture(scope="function")
   def my_fixture():
       return "value"
```text

1. **Test Timeout**

   ```bash
   # Increase timeout
   pytest --timeout=300
```text

1. **Coverage Issues**

   ```bash
   # Check coverage configuration
   pytest --cov=src --cov-report=term-missing
```text

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- FLEXT Quality Standards
- Test Examples
- CI/CD Configuration
````
