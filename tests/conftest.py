"""Comprehensive pytest configuration with full FlextCore integration.

This configuration provides extensive testing infrastructure with all pytest plugins
and FlextCore patterns integrated for maximum testing capabilities.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog
from hypothesis import strategies as st
from pytest_httpx import HTTPXMock
from pytest_mock import MockerFixture

from flext_core import (
    FlextContainer,
    get_flext_container,
)
from tests.support.async_utils import AsyncTestUtils

from tests.support.domain_factories import (
    ConfigurationFactory,
    FlextResultFactory,
    UserDataFactory,
)
from tests.support.http_utils import HTTPTestUtils
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

# Pytest markers for test categorization
pytest_plugins = [
    "pytest_asyncio",
    "benchmark",         # Corrected from pytest_benchmark
    "pytest_httpx",
    "pytest_mock",
    "xdist",             # Corrected from pytest_xdist
    "pytest_cov",        
    "randomly",          # Corrected from pytest_randomly
    "clarity",           # Corrected from pytest_clarity
    "sugar",             # Corrected from pytest_sugar
    "deadfixtures",      # Corrected from pytest_deadfixtures
    "env",               # Corrected from pytest_env
    "timeout",           # Corrected from pytest_timeout
]


# Test markers for organization
def pytest_configure(config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "ddd: Domain-driven design tests")
    config.addinivalue_line("markers", "architecture: Architectural pattern tests")
    config.addinivalue_line("markers", "boundary: Boundary condition tests")
    config.addinivalue_line("markers", "error_path: Error path scenarios")
    config.addinivalue_line("markers", "happy_path: Happy path scenarios")
    config.addinivalue_line("markers", "pep8: PEP8 compliance tests")
    config.addinivalue_line("markers", "parametrize_advanced: Advanced parametrized tests")
    config.addinivalue_line("markers", "guards: Guard and validation tests")
    config.addinivalue_line("markers", "advanced: Advanced pattern tests")
    config.addinivalue_line("markers", "concurrency: Concurrency tests")

# =============================================================================
# CORE FIXTURES
# =============================================================================


@pytest.fixture
def flext_result_factory() -> FlextResultFactory:
    """Factory for creating FlextResult instances."""
    return FlextResultFactory()


@pytest.fixture
def user_data_factory() -> type[UserDataFactory]:
    """Factory for creating user test data."""
    return UserDataFactory


@pytest.fixture
def config_data_factory() -> ConfigurationFactory:
    """Factory for creating configuration test data."""
    return ConfigurationFactory()


@pytest.fixture
def flext_container() -> FlextContainer:
    """Clean FlextContainer instance for testing."""
    container = get_flext_container()
    container.clear()  # Ensure clean state
    return container

# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================


@pytest.fixture
def performance_profiler() -> PerformanceProfiler:
    """Performance profiler for memory and timing analysis."""
    return PerformanceProfiler()


@pytest.fixture
def benchmark_utils() -> BenchmarkUtils:
    """Benchmark utilities for performance testing."""
    return BenchmarkUtils()

# =============================================================================
# ASYNC FIXTURES
# =============================================================================


@pytest.fixture
def async_test_utils() -> AsyncTestUtils:
    """Async testing utilities."""
    return AsyncTestUtils()


@pytest.fixture
def event_loop() -> None:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# =============================================================================
# HTTP FIXTURES
# =============================================================================


@pytest.fixture
def http_test_utils(httpx_mock: HTTPXMock) -> HTTPTestUtils:
    """HTTP testing utilities with httpx mock."""
    return HTTPTestUtils(httpx_mock)

# =============================================================================
# TEMP FILE FIXTURES
# =============================================================================


@pytest.fixture
def temp_file() -> Generator[Path]:
    """Temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

# =============================================================================
# TIME FIXTURES
# =============================================================================


@pytest.fixture
def frozen_time():
    """Frozen time for consistent testing."""
    return datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

# =============================================================================
# LOGGING FIXTURES
# =============================================================================


@pytest.fixture
def logger():
    """Structured logger for testing."""
    return structlog.get_logger("test")

# =============================================================================
# ERROR TESTING FIXTURES
# =============================================================================


@pytest.fixture
def sample_error_data():
    """Sample error data for testing."""
    return {
        "message": "Test error",
        "code": "TEST_ERROR",
        "context": {"field": "test_field", "value": "test_value"},
    }

# =============================================================================
# FACTORY BOY INTEGRATION
# =============================================================================


@pytest.fixture(autouse=True)
def setup_factory_boy() -> None:
    """Setup factory_boy for test isolation."""
    # Reset any factory state between tests
    return
    # Cleanup after test if needed

# =============================================================================
# COMPREHENSIVE PLUGIN INTEGRATION
# =============================================================================


@pytest.fixture
def mock_service(mocker: MockerFixture):
    """Mock service for testing."""
    return mocker.Mock()


@pytest.fixture
def benchmark_config():
    """Benchmark configuration for performance tests."""
    return {
        "min_rounds": 5,
        "max_time": 1.0,
        "timer": time.perf_counter,
        "disable_gc": True,
        "warmup": True,
    }

# =============================================================================
# HYPOTHESIS INTEGRATION
# =============================================================================


@pytest.fixture
def hypothesis_strategies():
    """Hypothesis strategies for property-based testing."""
    return {
        "texts": st.text(min_size=1, max_size=100),
        "integers": st.integers(min_value=1, max_value=1000),
        "floats": st.floats(min_value=0.1, max_value=1000.0),
        "emails": st.emails(),
        "uuids": st.uuids(),
        "dates": st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2030, 12, 31)),
    }

# =============================================================================
# VALIDATION FIXTURES
# =============================================================================


@pytest.fixture
def validation_test_cases():
    """Common validation test cases."""
    return [
        {"input": "", "expected": False, "error": "empty_string"},
        {"input": "valid@example.com", "expected": True, "error": None},
        {"input": "invalid-email", "expected": False, "error": "invalid_format"},
        {"input": None, "expected": False, "error": "none_value"},
    ]

# =============================================================================
# PARAMETRIZE HELPERS
# =============================================================================


def create_parametrize_data(test_cases: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Helper to create parametrize data from test cases."""
    return [(case["input"], case["expected"], case.get("error")) for case in test_cases]

# =============================================================================
# TEST SUPPORT CLASSES AND HELPERS
# =============================================================================


@dataclass
class TestCase:
    """Standard test case structure."""

    name: str
    input_data: Any
    expected: Any
    error_expected: bool = False
    error_message: str | None = None


@dataclass
class TestScenario:
    """Test scenario for complex test patterns."""

    scenario_type: str
    description: str
    test_data: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = "success"


class AssertHelpers:
    """Collection of assertion helpers for tests."""

    @staticmethod
    def assert_result_success(result) -> None:
        """Assert that a FlextResult is successful."""
        assert hasattr(result, "success"), "Object should have 'success' attribute"
        assert result.success, f"Expected success but got failure: {getattr(result, 'error', 'Unknown error')}"

    @staticmethod
    def assert_result_failure(result, expected_error: str | None = None) -> None:
        """Assert that a FlextResult is a failure."""
        assert hasattr(result, "failure"), "Object should have 'failure' attribute"
        assert result.is_failure, "Expected failure but got success"
        if expected_error:
            assert expected_error in str(result.error), f"Expected error '{expected_error}' not found in '{result.error}'"


@dataclass
class PerformanceMetrics:
    """Performance metrics for test monitoring."""

    execution_time: float
    memory_usage: int = 0
    iterations: int = 1
    max_time_allowed: float = 1.0

    def is_within_limits(self) -> bool:
        """Check if performance is within acceptable limits."""
        return self.execution_time <= self.max_time_allowed


def assert_performance(func: Callable, max_time: float = 1.0, iterations: int = 100) -> None:
    """Assert that a function performs within time limits."""
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    assert elapsed < max_time, f"Performance test failed: {elapsed:.3f}s > {max_time}s for {iterations} iterations"

# =============================================================================
# TEST SCENARIO FIXTURES
# =============================================================================


@pytest.fixture
def test_scenarios() -> list[TestScenario]:
    """Provide standard test scenarios for complex testing."""
    return [
        TestScenario(
            scenario_type="success",
            description="Basic success scenario",
            test_data={"input": "valid_input", "expected": "processed_input"},
            expected_outcome="success"
        ),
        TestScenario(
            scenario_type="error",
            description="Basic error scenario",
            test_data={"input": "", "expected": None},
            expected_outcome="failure"
        ),
        TestScenario(
            scenario_type="validation",
            description="Validation scenario",
            test_data={"input": "test@example.com", "expected": True},
            expected_outcome="success"
        ),
    ]
