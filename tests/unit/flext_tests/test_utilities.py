"""Unit tests for flext_tests.utilities module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flext_core import FlextResult
from flext_tests.utilities import FlextTestsUtilities


class TestFlextTestsUtilities:
    """Test suite for FlextTestsUtilities class."""

    def test_create_test_result_success_default(self) -> None:
        """Test create_test_result success with default parameters."""
        result = FlextTestsUtilities.create_test_result()

        assert result.is_success
        # Default data is empty dict (None is not valid for FlextResult)
        assert result.value == {}

    def test_create_test_result_success_with_data(self) -> None:
        """Test create_test_result success with data."""
        test_data = {"key": "value"}
        result = FlextTestsUtilities.create_test_result(success=True, data=test_data)

        assert result.is_success
        assert result.value == test_data

    def test_create_test_result_failure_default(self) -> None:
        """Test create_test_result failure with default parameters."""
        result = FlextTestsUtilities.create_test_result(success=False)

        assert result.is_failure
        assert result.error == "Test error"

    def test_create_test_result_failure_custom_error(self) -> None:
        """Test create_test_result failure with custom error."""
        result = FlextTestsUtilities.create_test_result(
            success=False, error="Custom error"
        )

        assert result.is_failure
        assert result.error == "Custom error"

    def test_functional_service_default(self) -> None:
        """Test functional_service with default parameters."""
        service = FlextTestsUtilities.functional_service()

        assert isinstance(service, dict)
        assert service["type"] == "api"
        assert service["name"] == "functional_api_service"
        assert service["enabled"] is True
        assert service["host"] == "localhost"
        assert service["port"] == 8000
        assert service["timeout"] == 30
        assert service["retries"] == 3

    def test_functional_service_custom(self) -> None:
        """Test functional_service with custom parameters."""
        service = FlextTestsUtilities.functional_service(
            "database",
            host="db.example.com",
            port=5432,
            custom_field="custom_value",
        )

        assert service["type"] == "database"
        assert service["name"] == "functional_database_service"
        assert service["host"] == "db.example.com"
        assert service["port"] == 5432
        assert service["custom_field"] == "custom_value"

    def test_test_context_attribute_change(self) -> None:
        """Test test_context changes attribute temporarily."""

        class TestObject:
            def __init__(self) -> None:
                super().__init__()
                self.attribute = "original"

        obj = TestObject()

        with FlextTestsUtilities.test_context(obj, "attribute", "modified"):
            assert obj.attribute == "modified"

        # Should restore original value
        assert obj.attribute == "original"

    def test_test_context_new_attribute(self) -> None:
        """Test test_context adds new attribute temporarily."""

        class TestObject:
            pass

        obj = TestObject()

        with FlextTestsUtilities.test_context(obj, "new_attr", "new_value"):
            assert hasattr(obj, "new_attr")
            assert obj.new_attr == "new_value"

        # Should remove the attribute
        assert not hasattr(obj, "new_attr")

    def test_test_context_delete_after(self) -> None:
        """Test test_context with delete_after option."""

        class TestObject:
            def __init__(self) -> None:
                super().__init__()
                self.temp_attr = "temp"

        obj = TestObject()

        with FlextTestsUtilities.test_context(
            obj,
            "temp_attr",
            "modified",
        ):
            assert obj.temp_attr == "modified"

        # Should restore original value since attribute existed
        assert hasattr(obj, "temp_attr")
        assert obj.temp_attr == "temp"

    def test_test_context_exception_restores(self) -> None:
        """Test test_context restores value even when exception occurs."""

        class TestObject:
            def __init__(self) -> None:
                super().__init__()
                self.attribute = "original"

        obj = TestObject()

        with FlextTestsUtilities.test_context(obj, "attribute", "modified"):
            assert obj.attribute == "modified"
            msg = "Test exception"
            with pytest.raises(RuntimeError):
                raise RuntimeError(msg)

        # Should still restore original value
        assert obj.attribute == "original"


class TestFlextTestsUtilitiesTestUtilities:
    """Test suite for nested TestUtilities class."""

    def test_assert_result_success_passes(self) -> None:
        """Test assert_result_success with successful result."""
        result = FlextResult[str].ok("success")

        # Should not raise
        FlextTestsUtilities.TestUtilities.assert_result_success(result)

    def test_assert_result_success_fails(self) -> None:
        """Test assert_result_success with failed result."""
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success result"):
            FlextTestsUtilities.TestUtilities.assert_result_success(result)

    def test_assert_result_failure_passes(self) -> None:
        """Test assert_result_failure with failed result."""
        result = FlextResult[str].fail("error")

        # Should not raise
        FlextTestsUtilities.TestUtilities.assert_result_failure(result)

    def test_assert_result_failure_fails(self) -> None:
        """Test assert_result_failure with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure result"):
            FlextTestsUtilities.TestUtilities.assert_result_failure(result)

    def test_create_mock_service_no_methods(self) -> None:
        """Test create_mock_service with no methods specified."""
        mock = FlextTestsUtilities.TestUtilities.create_mock_service()

        assert isinstance(mock, MagicMock)

    def test_create_mock_service_with_methods(self) -> None:
        """Test create_mock_service with method implementations."""

        def mock_method() -> str:
            return "mocked"

        mock = FlextTestsUtilities.TestUtilities.create_mock_service(
            test_method=mock_method,
        )

        assert hasattr(mock, "test_method")
        assert mock.test_method() == "mocked"

    def test_generate_test_id_default(self) -> None:
        """Test generate_test_id with default prefix."""
        test_id = FlextTestsUtilities.TestUtilities.generate_test_id()

        assert isinstance(test_id, str)
        assert test_id.startswith("test_")
        assert len(test_id) == 13  # "test_" + 8 hex chars

    def test_generate_test_id_custom_prefix(self) -> None:
        """Test generate_test_id with custom prefix."""
        test_id = FlextTestsUtilities.TestUtilities.generate_test_id("custom")

        assert isinstance(test_id, str)
        assert test_id.startswith("custom_")
        assert len(test_id) == 15  # "custom_" + 8 hex chars

    def test_generate_test_id_uniqueness(self) -> None:
        """Test that generate_test_id produces unique IDs."""
        ids = [FlextTestsUtilities.TestUtilities.generate_test_id() for _ in range(10)]

        # All IDs should be unique
        assert len(set(ids)) == len(ids)
