# ruff: noqa: PLC0415, ANN401
"""Centralized pytest fixtures for flext-core testing.

Provides reusable fixtures following SOLID principles and DRY patterns
for consistent test setup across all test modules.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator

import pytest

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextFields,
    FlextFieldType,
    FlextResult,
    FlextTypes,
    get_flext_container,
)
from tests.support.factories import FlextResultFactory, UserDataFactory

JsonDict = FlextTypes.Core.JsonObject


class FlextTestFixtures:
    """Centralized fixture provider for flext-core testing.

    Implements Factory pattern for creating commonly used test fixtures
    with proper cleanup and isolation.
    """

    @staticmethod
    @pytest.fixture
    def sample_user_data() -> JsonDict:
        """Sample user data for testing."""
        return UserDataFactory.build()

    @staticmethod
    @pytest.fixture
    def sample_config_data() -> JsonDict:
        """Sample configuration data for testing."""
        return {"env": "test", "debug": True, "database_url": "sqlite:///:memory:"}

    @staticmethod
    @pytest.fixture
    def sample_field_data() -> JsonDict:
        """Sample field data for testing."""
        return {"field_name": "test_field", "field_type": "string", "required": True}

    @staticmethod
    @pytest.fixture
    def edge_case_values() -> dict[str, list[object]]:
        """Edge case values for comprehensive testing."""
        return {"edge_cases": [None, "", 0, False, [], {}]}

    @staticmethod
    @pytest.fixture
    def field_types_matrix() -> list[tuple[FlextFieldType, list[object], list[bool]]]:
        """Field types with test values and expected validity."""
        return [
            (FlextFieldType.STRING, ["test", ""], [True, False]),
            (FlextFieldType.INTEGER, [42, "invalid"], [True, False]),
        ]

    @staticmethod
    @pytest.fixture
    def temp_json_file() -> Generator[str]:
        """Temporary JSON file for file-based testing."""
        import json

        content = {"test": "data", "nested": {"value": 123}}
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(content, f)
            yield f.name

        # Cleanup handled by tempfile

    @staticmethod
    @pytest.fixture
    def isolated_container() -> Generator[FlextContainer]:
        """Isolated container instance for testing."""
        # Create fresh container for isolation
        return FlextContainer()
        # Container cleanup is automatic (no global state)

    @staticmethod
    @pytest.fixture
    def populated_container() -> Generator[FlextContainer]:
        """Container pre-populated with test services."""
        container = FlextContainer()

        # Register test services
        container.register("test_service", {"name": "test", "value": 42})
        container.register_factory(
            "test_factory",
            lambda: {"created": True, "id": "test"},
        )

        return container

    @staticmethod
    @pytest.fixture
    def sample_string_field() -> FlextFields.Core.BaseField[object]:
        """Sample string field for testing."""
        from flext_core.fields import create_field

        result = create_field(
            "string",
            "test_string_field",
            min_length=1,
            max_length=100,
            pattern=r"^[a-zA-Z0-9_]+$",
            required=True,
            description="Test string field",
        )
        if result.failure:
            raise ValueError(f"Failed to create string field: {result.error}")
        return result.value

    @staticmethod
    @pytest.fixture
    def sample_integer_field() -> FlextFields.Core.BaseField[object]:
        """Sample integer field for testing."""
        from flext_core.fields import create_field

        result = create_field(
            "integer",
            "test_integer_field",
            min_value=0,
            max_value=1000,
            required=True,
            description="Test integer field",
        )
        if result.failure:
            raise ValueError(f"Failed to create integer field: {result.error}")
        return result.value

    @staticmethod
    @pytest.fixture
    def sample_boolean_field() -> FlextFields.Core.BaseField[object]:
        """Sample boolean field for testing."""
        from flext_core.fields import create_field

        result = create_field(
            "boolean",
            "test_boolean_field",
            required=False,
            default_value=False,
            description="Test boolean field",
        )
        if result.failure:
            raise ValueError(f"Failed to create boolean field: {result.error}")
        return result.value

    @staticmethod
    @pytest.fixture
    def success_result() -> FlextResult[str]:
        """Sample successful FlextResult."""
        return FlextResultFactory.build(success=True, data="test_success_data")

    @staticmethod
    @pytest.fixture
    def failure_result() -> FlextResult[str]:
        """Sample failed FlextResult."""
        return FlextResultFactory.build(
            success=False,
            error="test_failure_message",
            error_code="TEST_ERROR_CODE",
        )

    @staticmethod
    @pytest.fixture
    def mock_validator_success() -> object:
        """Mock validator that always succeeds."""
        return lambda _: True

    @staticmethod
    @pytest.fixture
    def mock_validator_failure() -> object:
        """Mock validator that always fails."""
        return lambda _: False

    @staticmethod
    @pytest.fixture(autouse=True)
    def clear_global_container() -> Generator[None]:
        """Clear global container before each test for isolation."""
        # Reset global container state
        container = get_flext_container()
        container.clear()

        yield

        # Cleanup after test
        container.clear()

    @staticmethod
    @pytest.fixture
    def sample_config_instance() -> FlextConfig:
        """Sample FlextConfig instance for testing."""
        return FlextConfig(
            log_level="DEBUG",
            environment="test",
            debug=True,
        )

    @staticmethod
    @pytest.fixture
    def validation_test_cases() -> list[dict[str, object]]:
        """Test cases for validation scenarios."""
        return [
            {
                "name": "valid_email",
                "value": "test@example.com",
                "validator": "email",
                "expected": True,
            },
            {
                "name": "invalid_email",
                "value": "not_an_email",
                "validator": "email",
                "expected": False,
            },
            {
                "name": "valid_positive_integer",
                "value": 42,
                "validator": "positive_integer",
                "expected": True,
            },
            {
                "name": "invalid_negative_integer",
                "value": -1,
                "validator": "positive_integer",
                "expected": False,
            },
        ]

    @staticmethod
    @pytest.fixture(scope="session")
    def test_data_directory() -> str:
        """Directory containing test data files."""
        return "tests/data"

    @staticmethod
    @pytest.fixture
    def performance_test_size() -> int:
        """Size for performance tests (smaller in CI)."""
        return 1000  # Reasonable size for CI/CD


# Export fixtures for easy import
__all__ = [
    "FlextTestFixtures",
]


# Register custom markers
def pytest_configure(config: object) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "core: Core framework tests")
    config.addinivalue_line("markers", "domain: Domain layer tests")
    config.addinivalue_line("markers", "application: Application layer tests")
    config.addinivalue_line("markers", "infrastructure: Infrastructure layer tests")
    config.addinivalue_line("markers", "boundary: Boundary condition tests")
    config.addinivalue_line("markers", "error_path: Error path tests")
    config.addinivalue_line("markers", "happy_path: Happy path tests")
    config.addinivalue_line("markers", "pep8: PEP8 compliance tests")
