"""Unit tests for flext_tests.builders module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

from flext_tests.builders import FlextTestsBuilders


class TestFlextTestsBuilders:
    """Test suite for FlextTestsBuilders class."""

    def test_init(self) -> None:
        """Test FlextTestsBuilders initialization."""
        builder = FlextTestsBuilders()

        assert builder is not None
        data = builder.build()
        assert isinstance(data, dict)
        assert data == {}

    def test_with_users_default(self) -> None:
        """Test with_users with default count."""
        builder = FlextTestsBuilders()
        result = builder.with_users()

        assert result is builder
        data = builder.build()

        assert "users" in data
        users = cast("list[dict[str, str | bool]]", data["users"])
        assert len(users) == 5

        first_user = users[0]
        assert "id" in first_user
        assert "name" in first_user
        assert "email" in first_user
        assert "active" in first_user
        assert first_user["name"] == "User 0"

    def test_with_users_custom_count(self) -> None:
        """Test with_users with custom count."""
        builder = FlextTestsBuilders()
        builder.with_users(count=3)
        data = builder.build()

        users = cast("list[dict[str, str | bool]]", data["users"])
        assert len(users) == 3

    def test_with_configs_development(self) -> None:
        """Test with_configs in development mode."""
        builder = FlextTestsBuilders()
        result = builder.with_configs(production=False)

        assert result is builder
        data = builder.build()

        assert "configs" in data
        config = cast("dict[str, str | int | bool]", data["configs"])
        assert config["environment"] == "development"
        assert config["debug"] is True
        assert config["service_type"] == "api"
        assert config["timeout"] == 30

    def test_with_configs_production(self) -> None:
        """Test with_configs in production mode."""
        builder = FlextTestsBuilders()
        builder.with_configs(production=True)
        data = builder.build()

        config = cast("dict[str, str | int | bool]", data["configs"])
        assert config["environment"] == "production"
        assert config["debug"] is False

    def test_with_validation_fields_default(self) -> None:
        """Test with_validation_fields with default count."""
        builder = FlextTestsBuilders()
        result = builder.with_validation_fields()

        assert result is builder
        data = builder.build()

        assert "validation_fields" in data
        fields = cast("dict[str, object]", data["validation_fields"])

        valid_emails = cast("list[str]", fields["valid_emails"])
        assert len(valid_emails) == 5
        assert valid_emails[0] == "user0@example.com"

        invalid_emails = cast("list[str]", fields["invalid_emails"])
        assert len(invalid_emails) == 3

        assert fields["valid_hostnames"] == ["example.com", "localhost"]

    def test_with_validation_fields_custom_count(self) -> None:
        """Test with_validation_fields with custom count."""
        builder = FlextTestsBuilders()
        builder.with_validation_fields(count=3)
        data = builder.build()

        validation_fields = cast("dict[str, object]", data["validation_fields"])
        valid_emails = cast("list[str]", validation_fields["valid_emails"])
        assert len(valid_emails) == 3

    def test_build_empty(self) -> None:
        """Test build with no data added."""
        builder = FlextTestsBuilders()
        data = builder.build()

        assert isinstance(data, dict)
        assert data == {}

    def test_build_full_dataset(self) -> None:
        """Test build with all data types added."""
        builder = FlextTestsBuilders()
        builder.with_users(2).with_configs(production=True).with_validation_fields(2)
        data = builder.build()

        assert "users" in data
        assert "configs" in data
        assert "validation_fields" in data

        users = cast("list[dict[str, str | bool]]", data["users"])
        configs = cast("dict[str, str | int | bool]", data["configs"])
        assert len(users) == 2
        assert configs["environment"] == "production"

    def test_reset(self) -> None:
        """Test reset clears builder state."""
        builder = FlextTestsBuilders()
        builder.with_users(3).with_configs()

        data_before = builder.build()
        assert "users" in data_before
        assert "configs" in data_before

        result = builder.reset()
        assert result is builder

        data_after = builder.build()
        assert data_after == {}

    def test_method_chaining(self) -> None:
        """Test fluent interface method chaining."""
        builder = FlextTestsBuilders()
        result = (
            builder.with_users(2)
            .with_configs(production=False)
            .with_validation_fields(3)
            .build()
        )

        assert isinstance(result, dict)
        assert "users" in result
        assert "configs" in result
        assert "validation_fields" in result

    def test_multiple_calls_overwrite(self) -> None:
        """Test multiple calls to same method overwrite previous data."""
        builder = FlextTestsBuilders()
        builder.with_users(2)
        data1 = builder.build()
        users1 = cast("list[dict[str, str | bool]]", data1["users"])
        assert len(users1) == 2

        builder.with_users(5)
        data2 = builder.build()
        users2 = cast("list[dict[str, str | bool]]", data2["users"])
        assert len(users2) == 5
