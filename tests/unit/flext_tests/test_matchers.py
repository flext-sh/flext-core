"""Unit tests for flext_tests.matchers module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextResult
from flext_tests.matchers import DataBuilder, FlextTestsMatchers


class TestDataBuilder:
    """Test suite for DataBuilder class."""

    def test_init(self) -> None:
        """Test DataBuilder initialization."""
        builder = DataBuilder()
        assert isinstance(builder, DataBuilder)

    def test_with_users_default(self) -> None:
        """Test with_users with default count."""
        builder = DataBuilder()
        result = builder.with_users()

        assert result is builder  # Returns self for chaining
        data = builder.build()

        assert "users" in data
        users = cast("list[dict[str, str | int]]", data["users"])
        assert len(users) == 5

        # Check first user structure
        first_user = users[0]
        assert first_user["id"] == "USER-0"
        assert first_user["name"] == "User 0"
        assert first_user["email"] == "user0@example.com"
        assert first_user["age"] == 20

    def test_with_users_custom_count(self) -> None:
        """Test with_users with custom count."""
        builder = DataBuilder()
        builder.with_users(count=3)
        data = builder.build()

        users = cast("list[dict[str, str | int]]", data["users"])
        assert len(users) == 3
        assert users[-1]["id"] == "USER-2"

    def test_with_configs_development(self) -> None:
        """Test with_configs in development mode."""
        builder = DataBuilder()
        result = builder.with_configs(production=False)

        assert result is builder
        data = builder.build()

        assert "configs" in data
        config = data["configs"]
        assert config["environment"] == "development"
        assert config["debug"] is True
        assert config["database_url"] == "postgresql://localhost/testdb"
        assert config["api_timeout"] == 30
        assert config["max_connections"] == 10

    def test_with_configs_production(self) -> None:
        """Test with_configs in production mode."""
        builder = DataBuilder()
        builder.with_configs(production=True)
        data = builder.build()

        config = data["configs"]
        assert config["environment"] == "production"
        assert config["debug"] is False

    def test_with_validation_fields_default(self) -> None:
        """Test with_validation_fields with default count."""
        builder = DataBuilder()
        result = builder.with_validation_fields()

        assert result is builder
        data = builder.build()

        assert "validation_fields" in data
        fields = data["validation_fields"]

        assert len(fields["valid_emails"]) == 5
        assert fields["valid_emails"][0] == "user0@example.com"
        assert len(fields["invalid_emails"]) == 3
        assert fields["valid_hostnames"] == ["example.com", "localhost"]

    def test_with_validation_fields_custom_count(self) -> None:
        """Test with_validation_fields with custom count."""
        builder = DataBuilder()
        builder.with_validation_fields(count=3)
        data = builder.build()

        assert len(data["validation_fields"]["valid_emails"]) == 3

    def test_build_empty(self) -> None:
        """Test build with no data added."""
        builder = DataBuilder()
        data = builder.build()

        assert isinstance(data, dict)
        assert data == {}

    def test_build_full_dataset(self) -> None:
        """Test build with all data types added."""
        builder = DataBuilder()
        builder.with_users(2).with_configs(production=True).with_validation_fields(2)
        data = builder.build()

        assert "users" in data
        assert "configs" in data
        assert "validation_fields" in data
        users = cast("list[dict[str, str | int]]", data["users"])
        configs = cast("dict[str, str | int | bool]", data["configs"])
        assert len(users) == 2
        assert configs["environment"] == "production"


class TestFlextTestsMatchers:
    """Test suite for FlextTestsMatchers class."""

    def test_assert_result_success_passes(self) -> None:
        """Test assert_result_success with successful result."""
        matchers = FlextTestsMatchers()
        result = FlextResult[str].ok("success")

        # Should not raise
        matchers.assert_result_success(result)

    def test_assert_result_success_fails(self) -> None:
        """Test assert_result_success with failed result."""
        matchers = FlextTestsMatchers()
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success result"):
            matchers.assert_result_success(result)

    def test_assert_result_success_custom_message(self) -> None:
        """Test assert_result_success with custom error message."""
        matchers = FlextTestsMatchers()
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Custom message"):
            matchers.assert_result_success(result, "Custom message")

    def test_assert_result_failure_passes(self) -> None:
        """Test assert_result_failure with failed result."""
        result = FlextResult[str].fail("error")

        # Should not raise
        FlextTestsMatchers.assert_result_failure(result)

    def test_assert_result_failure_fails(self) -> None:
        """Test assert_result_failure with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure result"):
            FlextTestsMatchers.assert_result_failure(result)

    def test_assert_result_failure_with_expected_error(self) -> None:
        """Test assert_result_failure with expected error substring."""
        result = FlextResult[str].fail("Database connection failed")

        # Should not raise
        FlextTestsMatchers.assert_result_failure(result, "connection")

    def test_assert_result_failure_expected_error_not_found(self) -> None:
        """Test assert_result_failure when expected error substring not found."""
        result = FlextResult[str].fail("Database error")

        with pytest.raises(
            AssertionError, match="Expected error containing 'connection'"
        ):
            FlextTestsMatchers.assert_result_failure(result, "connection")

    def test_assert_dict_contains_passes(self) -> None:
        """Test assert_dict_contains with matching data."""
        data = {"key1": "value1", "key2": "value2"}
        expected = {"key1": "value1"}

        # Should not raise
        FlextTestsMatchers.assert_dict_contains(data, expected)

    def test_assert_dict_contains_missing_key(self) -> None:
        """Test assert_dict_contains with missing key."""
        data = {"key1": "value1"}
        expected = {"key2": "value2"}

        with pytest.raises(AssertionError, match="Key 'key2' not found"):
            FlextTestsMatchers.assert_dict_contains(data, expected)

    def test_assert_dict_contains_wrong_value(self) -> None:
        """Test assert_dict_contains with wrong value."""
        data = {"key1": "value1"}
        expected = {"key1": "wrong_value"}

        with pytest.raises(AssertionError, match="expected wrong_value, got value1"):
            FlextTestsMatchers.assert_dict_contains(data, expected)

    def test_assert_list_contains_passes(self) -> None:
        """Test assert_list_contains with item in list."""
        items = ["item1", "item2", "item3"]

        # Should not raise
        FlextTestsMatchers.assert_list_contains(items, "item2")

    def test_assert_list_contains_missing_item(self) -> None:
        """Test assert_list_contains with item not in list."""
        items = ["item1", "item2"]

        with pytest.raises(AssertionError, match="Expected item 'item3' not found"):
            FlextTestsMatchers.assert_list_contains(items, "item3")

    def test_assert_valid_email_passes(self) -> None:
        """Test assert_valid_email with valid email."""
        # Should not raise
        FlextTestsMatchers.assert_valid_email("test@example.com")

    def test_assert_valid_email_fails(self) -> None:
        """Test assert_valid_email with invalid email."""
        with pytest.raises(AssertionError, match="Invalid email format"):
            FlextTestsMatchers.assert_valid_email("invalid-email")

    def test_assert_valid_email_edge_cases(self) -> None:
        """Test assert_valid_email with various edge cases."""
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
            FlextTestsMatchers.assert_valid_email(email)

        for email in invalid_emails:
            with pytest.raises(AssertionError):
                FlextTestsMatchers.assert_valid_email(email)

    def test_assert_config_valid_passes(self) -> None:
        """Test assert_config_valid with valid config."""
        config = {
            "service_type": "api",
            "environment": "test",
            "timeout": 30,
        }

        # Should not raise
        FlextTestsMatchers.assert_config_valid(config)

    def test_assert_config_valid_missing_required_key(self) -> None:
        """Test assert_config_valid with missing required key."""
        config = {"service_type": "api"}  # Missing environment

        with pytest.raises(
            AssertionError, match="Required config key 'environment' missing"
        ):
            FlextTestsMatchers.assert_config_valid(config)

    def test_assert_config_valid_invalid_timeout(self) -> None:
        """Test assert_config_valid with invalid timeout."""
        config = {
            "service_type": "api",
            "environment": "test",
            "timeout": "invalid",  # Should be positive int
        }

        with pytest.raises(
            AssertionError, match="Config timeout must be positive integer"
        ):
            FlextTestsMatchers.assert_config_valid(config)

    def test_assert_config_valid_zero_timeout(self) -> None:
        """Test assert_config_valid with zero timeout."""
        config = {
            "service_type": "api",
            "environment": "test",
            "timeout": 0,  # Should be positive
        }

        with pytest.raises(
            AssertionError, match="Config timeout must be positive integer"
        ):
            FlextTestsMatchers.assert_config_valid(config)

    def test_nested_test_data_builder(self) -> None:
        """Test the nested TestDataBuilder class (legacy) in FlextTestsMatchers."""
        builder = FlextTestsMatchers.TestDataBuilder()
        result = builder.with_users(count=2).with_configs()

        assert result is builder
        data = builder.build()

        assert "users" in data
        assert "configs" in data
        users = cast("list[dict[str, str | int]]", data["users"])
        configs = cast("dict[str, str | int | bool]", data["configs"])
        assert len(users) == 2
        assert configs["environment"] == "development"
