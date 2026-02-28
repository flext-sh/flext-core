"""Unit tests for flext_tests.utilities module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest
from flext_core import FlextResult, r
from flext_tests.utilities import FlextTestsUtilities

from tests.test_utils import assertion_helpers


class TestFlextTestsUtilitiesResult:
    """Test suite for FlextTestsUtilities.Tests.Result class."""

    def test_assert_success_passes(self) -> None:
        """Test assert_success with successful result."""
        result = FlextResult[str].ok("success")

        # Should not raise and return value
        value = FlextTestsUtilities.Tests.Result.assert_success(result)
        assert value == "success"

    def test_assert_success_fails(self) -> None:
        """Test assert_success with failed result."""
        result: r[str] = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success but got failure"):
            FlextTestsUtilities.Tests.Result.assert_success(result)

    def test_assert_failure_passes(self) -> None:
        """Test assert_failure with failed result."""
        result: r[str] = FlextResult[str].fail("error message")

        # Should not raise and return error
        error = FlextTestsUtilities.Tests.Result.assert_failure(result)
        assert error == "error message"

    def test_assert_failure_fails(self) -> None:
        """Test assert_failure with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure but got success"):
            FlextTestsUtilities.Tests.Result.assert_failure(result)

    def test_assert_failure_with_expected_error(self) -> None:
        """Test assert_failure with expected error substring."""
        result: r[str] = FlextResult[str].fail("validation error occurred")

        # Should not raise when substring matches
        error = FlextTestsUtilities.Tests.Result.assert_failure(result, "validation")
        assert "validation" in error

    def test_assert_failure_with_expected_error_mismatch(self) -> None:
        """Test assert_failure when expected error doesn't match."""
        result: r[str] = FlextResult[str].fail("validation error occurred")

        with pytest.raises(AssertionError, match="Expected error containing"):
            FlextTestsUtilities.Tests.Result.assert_failure(result, "not found")

    def test_assert_success_with_value(self) -> None:
        """Test assert_success_with_value with matching value."""
        result = FlextResult[str].ok("expected")

        # Should not raise
        FlextTestsUtilities.Tests.Result.assert_success_with_value(result, "expected")

    def test_assert_success_with_value_mismatch(self) -> None:
        """Test assert_success_with_value with non-matching value."""
        result = FlextResult[str].ok("actual")

        with pytest.raises(AssertionError):
            FlextTestsUtilities.Tests.Result.assert_success_with_value(
                result,
                "expected",
            )

    def test_assert_failure_with_error(self) -> None:
        """Test assert_failure_with_error with matching error."""
        result: r[str] = FlextResult[str].fail("test error")

        # Should not raise
        FlextTestsUtilities.Tests.Result.assert_failure_with_error(result, "test")

    def test_assert_failure_with_error_mismatch(self) -> None:
        """Test assert_failure_with_error with non-matching error."""
        result: r[str] = FlextResult[str].fail("actual error")

        with pytest.raises(AssertionError):
            FlextTestsUtilities.Tests.Result.assert_failure_with_error(
                result,
                "expected",
            )


class TestFlextTestsUtilitiesTestContext:
    """Test suite for FlextTestsUtilities.Tests.TestContext class."""

    def test_temporary_attribute_change(self) -> None:
        """Test temporary_attribute changes attribute temporarily."""

        class TestObject:
            def __init__(self) -> None:
                super().__init__()
                self.attribute = "original"

        obj = TestObject()

        with FlextTestsUtilities.Tests.TestContext.temporary_attribute(
            obj,
            "attribute",
            "modified",
        ):
            assert obj.attribute == "modified"

        # Should restore original value
        assert obj.attribute == "original"

    def test_temporary_attribute_new(self) -> None:
        """Test temporary_attribute adds new attribute temporarily."""

        class TestObject:
            pass

        obj = TestObject()

        with FlextTestsUtilities.Tests.TestContext.temporary_attribute(
            obj,
            "new_attr",
            "new_value",
        ):
            assert hasattr(obj, "new_attr")
            assert getattr(obj, "new_attr", None) == "new_value"

        # Should remove the attribute
        assert not hasattr(obj, "new_attr")

    def test_temporary_attribute_exception_restores(self) -> None:
        """Test temporary_attribute restores value even when exception occurs."""

        class TestObject:
            def __init__(self) -> None:
                super().__init__()
                self.attribute = "original"

        obj = TestObject()

        with FlextTestsUtilities.Tests.TestContext.temporary_attribute(
            obj,
            "attribute",
            "modified",
        ):
            assert obj.attribute == "modified"
            msg = "Test exception"
            with pytest.raises(RuntimeError):
                raise RuntimeError(msg)

        # Should still restore original value
        assert obj.attribute == "original"


class TestFlextTestsUtilitiesFactory:
    """Test suite for FlextTestsUtilities.Tests.Factory class."""

    def test_create_result_success(self) -> None:
        """Test create_result with value."""
        result = FlextTestsUtilities.Tests.Factory.create_result("test_value")

        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "test_value"

    def test_create_result_failure(self) -> None:
        """Test create_result with error."""
        result: r[str] = FlextTestsUtilities.Tests.Factory.create_result(
            None,
            error="test error",
        )

        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "test error"

    def test_create_result_no_args(self) -> None:
        """Test create_result with no arguments returns failure."""
        result: r[str] = FlextTestsUtilities.Tests.Factory.create_result(None)

        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "No value or error provided"

    def test_create_test_data(self) -> None:
        """Test create_test_data creates dict with kwargs."""
        data = FlextTestsUtilities.Tests.Factory.create_test_data(
            key1="value1",
            key2=42,
            key3=True,
        )

        assert data["key1"] == "value1"
        assert data["key2"] == 42
        assert data["key3"] is True


class TestFlextTestsUtilitiesTestUtilitiesCompat:
    """Test suite for TestUtilities compatibility class."""

    def test_assert_result_success_passes(self) -> None:
        """Test assert_result_success with successful result."""
        result = FlextResult[str].ok("success")

        # Should not raise
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)

    def test_assert_result_success_fails(self) -> None:
        """Test assert_result_success with failed result."""
        result: r[str] = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success but got failure"):
            FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)

    def test_assert_result_failure_passes(self) -> None:
        """Test assert_result_failure with failed result."""
        result: r[str] = FlextResult[str].fail("error")

        # Should not raise
        FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(result)

    def test_assert_result_failure_fails(self) -> None:
        """Test assert_result_failure with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure but got success"):
            FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(result)
