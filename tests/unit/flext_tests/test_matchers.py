"""Unit tests for flext_tests.matchers module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextResult, t
from flext_tests import tm
from flext_tests.constants import c


class TestFlextTestsMatchers:
    """Test suite for FlextTestsMatchers class."""

    def test_assert_result_success_passes(self) -> None:
        """Test tm.ok() with successful result."""
        result = FlextResult[str].ok("success")

        # Should not raise
        value = tm.ok(result)
        assert value == "success"

    def test_assert_result_success_fails(self) -> None:
        """Test tm.ok() with failed result."""
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success but got failure"):
            tm.ok(result)

    def test_assert_result_success_custom_message(self) -> None:
        """Test tm.ok() with custom error message."""
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Custom message"):
            tm.ok(result, msg="Custom message")

    def test_assert_result_failure_passes(self) -> None:
        """Test tm.fail() with failed result."""
        result = FlextResult[str].fail("error")

        # Should not raise
        error = tm.fail(result)
        assert error == "error"

    def test_assert_result_failure_fails(self) -> None:
        """Test tm.fail() with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure but got success"):
            tm.fail(result)

    def test_assert_result_failure_with_expected_error(self) -> None:
        """Test tm.fail() with expected error substring."""
        result = FlextResult[str].fail("Database connection failed")

        # Should not raise
        error = tm.fail(result, contains="connection")
        assert "connection" in error

    def test_assert_result_failure_expected_error_not_found(self) -> None:
        """Test tm.fail() when expected error substring not found."""
        result = FlextResult[str].fail("Database error")

        with pytest.raises(
            AssertionError,
            match=r"Expected.*to contain 'connection'",
        ):
            tm.fail(result, contains="connection")

    def test_assert_dict_contains_passes(self) -> None:
        """Test tm.dict_() with contains parameter."""
        data = {"key1": "value1", "key2": "value2"}
        expected = {"key1": "value1"}

        # Should not raise
        tm.dict_(data, contains=expected)

    def test_assert_dict_contains_missing_key(self) -> None:
        """Test tm.dict_() with missing key."""
        data = {"key1": "value1"}
        expected = {"key2": "value2"}

        with pytest.raises(AssertionError, match="Key 'key2' not found in dict"):
            tm.dict_(data, contains=expected)

    def test_assert_dict_contains_wrong_value(self) -> None:
        """Test tm.dict_() with wrong value."""
        data = {"key1": "value1"}
        expected = {"key1": "wrong_value"}

        with pytest.raises(AssertionError, match="expected wrong_value, got value1"):
            tm.dict_(data, contains=expected)

    def test_assert_list_contains_passes(self) -> None:
        """Test tm.list_() with contains parameter."""
        items = ["item1", "item2", "item3"]

        # Should not raise
        tm.list_(items, contains="item2")

    def test_assert_list_contains_missing_item(self) -> None:
        """Test tm.list_() with item not in list."""
        items = ["item1", "item2"]

        with pytest.raises(AssertionError, match=r"Expected.*to contain 'item3'"):
            tm.list_(items, contains="item3")

    def test_assert_valid_email_passes(self) -> None:
        """Test tm.that() with email pattern match."""
        # Should not raise
        tm.that("test@example.com", match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_valid_email_fails(self) -> None:
        """Test tm.that() with invalid email."""
        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that("invalid-email", match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_valid_email_edge_cases(self) -> None:
        """Test tm.that() with various email edge cases."""
        valid_emails = [
            "user.name@domain.co.uk",
            "test+tag@example.com",
            "a@b.co",
        ]
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test.example.com",  # Missing @
        ]

        for email in valid_emails:
            # Should not raise
            tm.that(email, match=c.Tests.Matcher.EMAIL_PATTERN)

        for email in invalid_emails:
            with pytest.raises(AssertionError):
                tm.that(email, match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_config_valid_passes(self) -> None:
        """Test tm.dict_() with config validation."""
        config: dict[str, t.GeneralValueType] = {
            "service_type": "api",
            "environment": "test",
            "timeout": 30,
        }

        # Should not raise - validate keys and timeout
        tm.dict_(config, has_key=["service_type", "environment", "timeout"])
        tm.that(config["timeout"], is_=int, gt=0)

    def test_assert_config_valid_missing_required_key(self) -> None:
        """Test tm.dict_() with missing required key."""
        config = {"service_type": "api"}  # Missing environment

        with pytest.raises(AssertionError, match="Key 'environment' not found in dict"):
            tm.dict_(config, has_key=["service_type", "environment", "timeout"])

    def test_assert_config_valid_invalid_timeout(self) -> None:
        """Test tm.that() with invalid timeout type."""
        config = {
            "service_type": "api",
            "environment": "test",
            "timeout": "invalid",  # Should be positive int
        }

        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that(config["timeout"], is_=int, gt=0)

    def test_assert_config_valid_zero_timeout(self) -> None:
        """Test tm.that() with zero timeout."""
        config: dict[str, t.GeneralValueType] = {
            "service_type": "api",
            "environment": "test",
            "timeout": 0,  # Should be positive
        }

        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that(config["timeout"], is_=int, gt=0)
