"""Comprehensive tests for FlextCqrs - CQRS Pattern Implementation.

This module tests the CQRS (Command Query Responsibility Segregation) pattern
implementation provided by FlextCqrs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core import FlextCqrs, FlextModels


class TestFlextCqrs:
    """Test suite for FlextCqrs CQRS pattern implementation."""

    def test_cqrs_initialization(self) -> None:
        """Test CQRS initialization."""
        # FlextCqrs is a utility class with static methods
        assert FlextCqrs is not None
        assert hasattr(FlextCqrs, "Results")
        assert hasattr(FlextCqrs, "Operations")

    def test_cqrs_results_success(self) -> None:
        """Test CQRS Results success method."""
        data = {"user_id": "123", "name": "John Doe"}
        result = FlextCqrs.Results.success(data)

        assert result.is_success
        assert result.value == data

    def test_cqrs_results_success_with_config(self) -> None:
        """Test CQRS Results success method with configuration."""
        data = {"user_id": "123", "name": "John Doe"}
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
        )
        result = FlextCqrs.Results.success(data, config)

        assert result.is_success
        assert result.value == data
        assert hasattr(result, "_metadata")

    def test_cqrs_results_failure(self) -> None:
        """Test CQRS Results failure method."""
        result = FlextCqrs.Results.failure("Test error")

        assert result.is_failure
        assert result.error == "Test error"

    def test_cqrs_results_failure_with_error_code(self) -> None:
        """Test CQRS Results failure method with error code."""
        result = FlextCqrs.Results.failure("Test error", error_code="TEST_ERROR")

        assert result.is_failure
        assert result.error == "Test error"
        assert result.error_code == "TEST_ERROR"

    def test_cqrs_results_failure_with_error_data(self) -> None:
        """Test CQRS Results failure method with error data."""
        error_data: dict[str, object] = {"field": "value", "code": 400}
        result = FlextCqrs.Results.failure("Test error", error_data=error_data)

        assert result.is_failure
        assert result.error == "Test error"
        assert result.error_data == error_data

    def test_cqrs_results_failure_with_config(self) -> None:
        """Test CQRS Results failure method with configuration."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
        )
        result = FlextCqrs.Results.failure("Test error", config=config)

        assert result.is_failure
        assert result.error == "Test error"
        assert "handler_id" in result.error_data

    def test_cqrs_operations_create_command(self) -> None:
        """Test CQRS Operations create_command method."""
        command_data: dict[str, object] = {"command_type": "CreateUser"}
        result = FlextCqrs.Operations.create_command(command_data)

        assert result.is_success
        assert isinstance(result.value, FlextModels.Command)
        assert result.value.command_type == "CreateUser"

    def test_cqrs_operations_create_command_with_config(self) -> None:
        """Test CQRS Operations create_command method with configuration."""
        command_data: dict[str, object] = {"command_type": "CreateUser"}
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
        )
        result = FlextCqrs.Operations.create_command(command_data, config)

        assert result.is_success
        assert isinstance(result.value, FlextModels.Command)
        assert result.value.command_type == "CreateUser"

    def test_cqrs_operations_create_query(self) -> None:
        """Test CQRS Operations create_query method."""
        query_data: dict[str, object] = {"query_type": "GetUser"}
        result = FlextCqrs.Operations.create_query(query_data)

        assert result.is_success
        assert isinstance(result.value, FlextModels.Query)

    def test_cqrs_operations_create_query_with_config(self) -> None:
        """Test CQRS Operations create_query method with configuration."""
        query_data: dict[str, object] = {"query_type": "GetUser"}
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler", handler_name="TestHandler", handler_type="query"
        )
        result = FlextCqrs.Operations.create_query(query_data, config)

        assert result.is_success
        assert isinstance(result.value, FlextModels.Query)

    def test_cqrs_operations_create_handler_config(self) -> None:
        """Test CQRS Operations create_handler_config method."""
        result = FlextCqrs.Operations.create_handler_config("command")

        assert result.is_success
        config = result.value
        assert config.handler_type == "command"
        assert config.handler_id is not None
        assert config.handler_name is not None

    def test_cqrs_operations_create_handler_config_query(self) -> None:
        """Test CQRS Operations create_handler_config method for query."""
        result = FlextCqrs.Operations.create_handler_config("query")

        assert result.is_success
        config = result.value
        assert config.handler_type == "query"
        assert config.handler_id is not None
        assert config.handler_name is not None

    def test_cqrs_operations_create_handler_config_with_custom_id(self) -> None:
        """Test CQRS Operations create_handler_config method with custom ID."""
        result = FlextCqrs.Operations.create_handler_config(
            "command", handler_id="custom_id", handler_name="CustomHandler"
        )

        assert result.is_success
        config = result.value
        assert config.handler_type == "command"
        assert config.handler_id == "custom_id"
        assert config.handler_name == "CustomHandler"

    def test_cqrs_performance(self) -> None:
        """Test CQRS operations performance."""
        start_time = time.time()

        # Test performance of CQRS operations
        for i in range(1000):
            command_data: dict[str, object] = {"command_type": f"CreateUser{i}"}
            FlextCqrs.Operations.create_command(command_data)

            query_data: dict[str, object] = {"query_type": f"GetUser{i}"}
            FlextCqrs.Operations.create_query(query_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert execution_time < 1.0

    def test_cqrs_type_safety(self) -> None:
        """Test type safety of CQRS operations."""
        # Test type safety
        command_data: dict[str, object] = {"command_type": "CreateUser"}
        result = FlextCqrs.Operations.create_command(command_data)

        assert result.is_success
        assert isinstance(result.value, FlextModels.Command)
        assert result.value.command_type == "CreateUser"

    def test_cqrs_error_handling(self) -> None:
        """Test error handling in CQRS operations."""
        # Test with invalid command data that violates Pydantic validation
        invalid_command_data: dict[str, object] = {"invalid_field": "invalid_value"}
        result = FlextCqrs.Operations.create_command(invalid_command_data)

        # Should fail due to Pydantic validation
        assert result.is_failure
        assert result.error is not None
        assert "Extra inputs are not permitted" in result.error
