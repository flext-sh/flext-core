"""Unit tests for flext_tests.domains module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

from flext_tests.domains import FlextTestsDomains


class TestFlextTestsDomains:
    """Test suite for FlextTestsDomains class."""

    def test_create_configuration_default(self) -> None:
        """Test create_configuration with default parameters."""
        config = FlextTestsDomains.create_configuration()

        assert isinstance(config, dict)
        assert config["service_type"] == "api"
        assert config["environment"] == "test"
        assert config["debug"] is True
        assert config["log_level"] == "DEBUG"
        assert config["timeout"] == 30
        assert config["max_retries"] == 3
        assert config["storage_backend"] == "memory"
        assert config["enable_caching"] is True
        assert config["cache_ttl"] == 300
        assert "namespace" in config

    def test_create_configuration_custom(self) -> None:
        """Test create_configuration with custom parameters."""
        config = FlextTestsDomains.create_configuration(
            service_type="database",
            environment="production",
            custom_field="custom_value",
        )

        assert config["service_type"] == "database"
        assert config["environment"] == "production"
        assert config["custom_field"] == "custom_value"

    def test_create_payload_user_default(self) -> None:
        """Test create_payload with user data type default."""
        payload = FlextTestsDomains.create_payload()

        assert isinstance(payload, dict)
        assert "id" in payload
        assert payload["name"] == "Test User"
        assert payload["email"] == "test@example.com"
        assert payload["active"] is True

    def test_create_payload_order(self) -> None:
        """Test create_payload with order data type."""
        payload = FlextTestsDomains.create_payload("order")

        assert isinstance(payload, dict)
        assert "order_id" in payload
        assert "user_id" in payload
        assert payload["amount"] == 99.99
        assert payload["currency"] == "USD"
        assert payload["status"] == "pending"

    def test_create_payload_api_request(self) -> None:
        """Test create_payload with api_request data type."""
        payload = FlextTestsDomains.create_payload("api_request")

        assert isinstance(payload, dict)
        assert payload["method"] == "GET"
        assert payload["url"] == "/api/test"
        assert payload["headers"] == {"Content-Type": "application/json"}
        assert payload["body"] is None

    def test_create_payload_custom_fields(self) -> None:
        """Test create_payload with custom field overrides."""
        payload = FlextTestsDomains.create_payload("user", custom_field="custom_value")

        assert payload["custom_field"] == "custom_value"
        assert payload["name"] == "Test User"  # Default still present

    def test_create_payload_unknown_type(self) -> None:
        """Test create_payload with unknown data type."""
        payload = FlextTestsDomains.create_payload("unknown")

        assert isinstance(payload, dict)
        assert payload == {}  # Empty dict for unknown types

    def test_api_response_data_success_default(self) -> None:
        """Test api_response_data with success status default."""
        response = FlextTestsDomains.api_response_data()

        assert isinstance(response, dict)
        assert response["status"] == "success"
        assert "timestamp" in response
        assert "request_id" in response
        assert "data" not in response  # include_data defaults to None

    def test_api_response_data_success_with_data(self) -> None:
        """Test api_response_data with success status and data."""
        response = FlextTestsDomains.api_response_data(
            status="success", include_data=True
        )

        assert response["status"] == "success"
        assert response["data"] == {"test": "data"}

    def test_api_response_data_error(self) -> None:
        """Test api_response_data with error status."""
        response: dict[str, object] = FlextTestsDomains.api_response_data(
            status="error"
        )

        assert response["status"] == "error"
        assert "error" in response
        error_obj = cast("dict[str, object]", response["error"])
        assert error_obj["code"] == "TEST_ERROR"
        assert error_obj["message"] == "Test error message"

    def test_api_response_data_custom_fields(self) -> None:
        """Test api_response_data with custom field overrides."""
        response = FlextTestsDomains.api_response_data(custom_field="custom_value")

        assert response["custom_field"] == "custom_value"
        assert response["status"] == "success"  # Default still present

    def test_valid_email_cases(self) -> None:
        """Test valid_email_cases returns correct test cases."""
        cases = FlextTestsDomains.valid_email_cases()

        assert isinstance(cases, list)
        assert len(cases) == 7

        # Check some specific cases
        valid_emails = [email for email, is_valid in cases if is_valid]
        invalid_emails = [email for email, is_valid in cases if not is_valid]

        assert "test@example.com" in valid_emails
        assert "invalid-email" in invalid_emails
        assert "" in invalid_emails

    def test_create_service_default(self) -> None:
        """Test create_service with default parameters."""
        service = FlextTestsDomains.create_service()

        assert isinstance(service, dict)
        assert service["type"] == "api"
        assert service["name"] == "test_api_service"
        assert service["enabled"] is True
        assert "config" in service

    def test_create_service_custom(self) -> None:
        """Test create_service with custom parameters."""
        service = FlextTestsDomains.create_service(
            "database", custom_field="custom_value"
        )

        assert service["type"] == "database"
        assert service["name"] == "test_database_service"
        assert service["custom_field"] == "custom_value"

    def test_create_user_default(self) -> None:
        """Test create_user with default parameters."""
        user = FlextTestsDomains.create_user()

        assert isinstance(user, dict)
        assert "id" in user
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["first_name"] == "Test"
        assert user["last_name"] == "User"
        assert user["active"] is True
        assert "created_at" in user
        assert "updated_at" in user

    def test_create_user_overrides(self) -> None:
        """Test create_user with field overrides."""
        user = FlextTestsDomains.create_user(username="customuser", active=False)

        assert user["username"] == "customuser"
        assert user["active"] is False
        assert user["email"] == "test@example.com"  # Default still present
