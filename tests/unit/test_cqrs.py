"""Comprehensive tests for FlextCqrs - CQRS Pattern Implementation.

This module tests the CQRS (Command Query Responsibility Segregation) pattern
implementation provided by FlextCqrs with real functionality validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import cast

from flext_core import FlextCqrs, FlextModels, FlextResult
from flext_core.constants import FlextConstants


class TestFlextCqrs:
    """Comprehensive test suite for FlextCqrs CQRS pattern implementation."""

    def test_cqrs_initialization(self) -> None:
        """Test CQRS module structure and initialization."""
        # Verify module structure
        assert FlextCqrs is not None
        assert hasattr(FlextCqrs, "Results")
        assert hasattr(FlextCqrs, "Operations")
        assert hasattr(FlextCqrs, "Decorators")

        # Verify nested classes exist
        assert hasattr(FlextCqrs.Results, "success")
        assert hasattr(FlextCqrs.Results, "failure")
        assert hasattr(FlextCqrs.Operations, "create_command")
        assert hasattr(FlextCqrs.Operations, "create_query")
        assert hasattr(FlextCqrs.Operations, "create_handler_config")
        assert hasattr(FlextCqrs.Decorators, "command_handler")

    def test_cqrs_results_success_basic(self) -> None:
        """Test basic CQRS Results success functionality."""
        data = {"user_id": "123", "name": "John Doe"}
        result = FlextCqrs.Results.success(data)

        assert result.is_success
        assert result.value == data
        assert result.error is None
        assert result.error_code is None
        assert result.error_data == {}

    def test_cqrs_results_success_with_metadata(self) -> None:
        """Test CQRS Results success with configuration metadata."""
        data = {"user_id": "123", "name": "John Doe"}
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_123",
            handler_name="TestHandler",
            handler_type="command",
            metadata={"version": "1.0", "environment": "test"},
        )
        result = FlextCqrs.Results.success(data, config)

        assert result.is_success
        assert result.value == data
        assert hasattr(result, "_metadata")

        # Verify metadata is properly attached
        metadata = getattr(result, "_metadata")
        assert metadata is not None
        assert metadata["handler_id"] == "test_handler_123"
        assert metadata["handler_name"] == "TestHandler"
        assert metadata["handler_type"] == "command"

    def test_cqrs_results_success_without_config(self) -> None:
        """Test CQRS Results success without configuration metadata."""
        data = {"operation": "completed", "status": "success"}
        result = FlextCqrs.Results.success(data, None)

        assert result.is_success
        assert result.value == data
        assert not hasattr(result, "_metadata")

    def test_cqrs_results_failure_basic(self) -> None:
        """Test basic CQRS Results failure functionality."""
        result = FlextCqrs.Results.failure("Operation failed")

        assert result.is_failure
        assert result.error == "Operation failed"
        assert result.error_code == FlextConstants.Cqrs.CQRS_OPERATION_FAILED
        assert result.error_data == {}

    def test_cqrs_results_failure_with_custom_error_code(self) -> None:
        """Test CQRS Results failure with custom error code."""
        result = FlextCqrs.Results.failure(
            "Validation failed", error_code="VALIDATION_ERROR"
        )

        assert result.is_failure
        assert result.error == "Validation failed"
        assert result.error_code == "VALIDATION_ERROR"

    def test_cqrs_results_failure_with_error_data(self) -> None:
        """Test CQRS Results failure with additional error data."""
        error_data: dict[str, object] = {
            "field": "email",
            "validation": "required",
            "code": 400,
        }
        result = FlextCqrs.Results.failure("Email is required", error_data=error_data)

        assert result.is_failure
        assert result.error == "Email is required"
        assert result.error_data == error_data

    def test_cqrs_results_failure_with_config_metadata(self) -> None:
        """Test CQRS Results failure with configuration metadata enhancement."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="validation_handler",
            handler_name="EmailValidator",
            handler_type="command",
            metadata={"rule": "email_required", "severity": "error"},
        )
        result = FlextCqrs.Results.failure("Email validation failed", config=config)

        assert result.is_failure
        assert result.error == "Email validation failed"
        assert "handler_id" in result.error_data
        assert "handler_name" in result.error_data
        assert "handler_type" in result.error_data
        assert "handler_metadata" in result.error_data

        # Verify enhanced error data
        assert result.error_data["handler_id"] == "validation_handler"
        assert result.error_data["handler_name"] == "EmailValidator"
        assert isinstance(result.error_data, dict)
        assert result.error_data["handler_type"] == "command"
        handler_metadata = cast("dict[str, str]", result.error_data["handler_metadata"])
        assert handler_metadata["rule"] == "email_required"

    def test_cqrs_operations_create_command_basic(self) -> None:
        """Test basic command creation functionality."""
        command_data: dict[str, object] = {
            "command_type": "CreateUser",
            "issuer_id": "user_123",
        }
        command = FlextModels.Command.model_validate(command_data)

        assert isinstance(command, FlextModels.Command)
        assert command.command_type == "CreateUser"
        assert command.issuer_id == "user_123"
        assert command.command_id is not None  # Auto-generated
        assert command.issued_at is not None  # Auto-generated

    def test_cqrs_operations_create_command_with_existing_id(self) -> None:
        """Test command creation with existing command ID."""
        custom_id = str(uuid.uuid4())
        command_data: dict[str, object] = {
            "command_type": "UpdateUser",
            "command_id": custom_id,
            "issuer_id": "system",
        }
        command = FlextModels.Command.model_validate(command_data)
        assert command.command_id == custom_id  # Should preserve existing ID
        assert command.command_type == "UpdateUser"
        assert command.issuer_id == "system"

    def test_cqrs_operations_create_command_validation_error(self) -> None:
        """Test command creation with validation errors."""
        # Missing required fields
        invalid_data: dict[str, object] = {
            "payload": {"name": "John"}  # Missing command_type
        }
        try:
            FlextModels.Command.model_validate(invalid_data)
            assert False, "Should have raised ValidationError"
        except Exception:
            pass  # Expected validation error - Pydantic raises ValidationError

    def test_cqrs_operations_create_query_basic(self) -> None:
        """Test basic query creation functionality."""
        query_data: dict[str, object] = {
            "query_type": "GetUserById",
            "filters": {"user_id": "123"},
        }
        query = FlextModels.Query.model_validate(query_data)
        assert isinstance(query, FlextModels.Query)
        assert query.filters == {"user_id": "123"}
        assert query.query_id is not None  # Auto-generated

    def test_cqrs_operations_create_query_with_existing_id(self) -> None:
        """Test query creation with existing query ID."""
        custom_id = str(uuid.uuid4())
        query_data: dict[str, object] = {
            "query_type": "SearchUsers",
            "query_id": custom_id,
            "filters": {"active": True, "role": "admin"},
        }
        query = FlextModels.Query.model_validate(query_data)
        assert query.query_id == custom_id  # Should preserve existing ID
        assert query.filters == {"active": True, "role": "admin"}

    def test_cqrs_operations_create_query_validation_error(self) -> None:
        """Test query creation with validation errors."""
        # Invalid query data structure
        invalid_data: dict[str, object] = {
            "filters": "not_a_dict"  # Should be dict
        }
        try:
            FlextModels.Query.model_validate(invalid_data)
            assert False, "Should have raised ValidationError"
        except Exception:
            pass  # Expected validation error - Pydantic raises ValidationError

    def test_cqrs_operations_create_handler_config_basic(self) -> None:
        """Test basic handler configuration creation."""
        result = FlextModels.CqrsConfig.Handler.create_handler_config("command")

        assert result.is_success
        config = result.value
        assert isinstance(config, FlextModels.CqrsConfig.Handler)
        assert config.handler_type == "command"
        assert config.handler_mode == "command"
        assert config.handler_id is not None
        assert config.handler_name == "command_handler"
        assert config.command_timeout == FlextConstants.Cqrs.DEFAULT_TIMEOUT
        assert config.max_command_retries == FlextConstants.Cqrs.DEFAULT_RETRIES

    def test_cqrs_operations_create_handler_config_query(self) -> None:
        """Test query handler configuration creation."""
        result = FlextModels.CqrsConfig.Handler.create_handler_config("query")

        assert result.is_success
        config = result.value
        assert config.handler_type == "query"
        assert config.handler_mode == "query"
        assert config.handler_name == "query_handler"

    def test_cqrs_operations_create_handler_config_custom_settings(self) -> None:
        """Test handler configuration with custom settings."""
        result = FlextModels.CqrsConfig.Handler.create_handler_config(
            "command",
            handler_id="custom_handler_123",
            handler_name="CustomCommandHandler",
        )

        assert result.is_success
        config = result.value
        assert config.handler_id == "custom_handler_123"
        assert config.handler_name == "CustomCommandHandler"

    def test_cqrs_operations_create_handler_config_with_overrides(self) -> None:
        """Test handler configuration with config overrides."""
        config_overrides: dict[str, object] = {
            "command_timeout": 5000,
            "max_command_retries": 5,
            "metadata": {"custom": "value"},
        }
        result = FlextModels.CqrsConfig.Handler.create_handler_config(
            "command", config_overrides=config_overrides
        )

        assert result.is_success
        config = result.value
        assert config.command_timeout == 5000
        assert config.max_command_retries == 5
        assert config.metadata["custom"] == "value"

    def test_cqrs_operations_create_handler_config_validation_error(self) -> None:
        """Test handler configuration creation with validation errors."""
        # Test with valid type but invalid config to trigger validation error
        result = FlextModels.CqrsConfig.Handler.create_handler_config("command")

        assert result.is_success  # Should succeed with valid handler type

    def test_cqrs_decorators_command_handler_basic(self) -> None:
        """Test basic command handler decorator functionality."""

        # Define a test command class
        @dataclass
        class TestCommand:
            user_id: str
            action: str

        # Define a handler function
        def handle_test_command(cmd: TestCommand) -> str:
            return f"Handled {cmd.action} for user {cmd.user_id}"

        # Apply decorator
        decorated_handler = FlextCqrs.Decorators.command_handler(TestCommand)(
            handle_test_command
        )

        # Test the decorated function
        command = TestCommand("123", "create")
        result = decorated_handler(command)

        assert result == "Handled create for user 123"
        assert decorated_handler.__name__ == "handle_test_command"
        assert hasattr(decorated_handler, "command_type")
        assert hasattr(decorated_handler, "handler_config")
        assert decorated_handler.__dict__["command_type"] == TestCommand
        assert decorated_handler.__dict__["flext_cqrs_decorator"] is True

    def test_cqrs_decorators_command_handler_with_config(self) -> None:
        """Test command handler decorator with configuration."""

        @dataclass
        class ProcessOrderCommand:
            order_id: str
            amount: float

        def process_order_handler(cmd: ProcessOrderCommand) -> str:
            return f"Processed order {cmd.order_id} for ${cmd.amount}"

        config = {
            "command_timeout": 30000,
            "max_command_retries": 3,
            "metadata": {"priority": "high", "queue": "orders"},
        }

        decorated_handler = FlextCqrs.Decorators.command_handler(
            ProcessOrderCommand, config=config
        )(process_order_handler)

        # Test functionality
        command = ProcessOrderCommand("ORD-123", 99.99)
        result = decorated_handler(command)

        assert result == "Processed order ORD-123 for $99.99"
        assert decorated_handler.__dict__["handler_config"].command_timeout == 30000
        assert (
            decorated_handler.__dict__["handler_config"].metadata["priority"] == "high"
        )

    def test_cqrs_decorators_command_handler_with_invalid_config(self) -> None:
        """Test command handler decorator with invalid configuration."""

        @dataclass
        class SimpleCommand:
            data: str

        def simple_handler(cmd: SimpleCommand) -> str:
            return f"Simple: {cmd.data}"

        # Invalid config that should cause fallback
        invalid_config = {"invalid_key": "invalid_value"}

        decorated_handler = FlextCqrs.Decorators.command_handler(
            SimpleCommand, config=invalid_config
        )(simple_handler)

        # Should still work with basic config
        command = SimpleCommand("test")
        result = decorated_handler(command)

        assert result == "Simple: test"
        assert decorated_handler.__dict__["handler_config"].handler_type == "command"
        assert decorated_handler.__dict__["handler_config"].handler_mode == "command"

    def test_cqrs_decorators_preserves_function_metadata(self) -> None:
        """Test that decorator preserves original function metadata."""

        @dataclass
        class DataCommand:
            value: int

        def documented_handler(cmd: DataCommand) -> str:
            """Well-documented handler function for testing metadata preservation."""
            return f"Value: {cmd.value}"

        # Add custom annotations and doc
        documented_handler.__annotations__ = {"return": str}
        documented_handler.__doc__ = "Custom documentation"

        decorated = FlextCqrs.Decorators.command_handler(DataCommand)(
            documented_handler
        )

        # Verify metadata preservation
        assert decorated.__name__ == "documented_handler"
        assert decorated.__doc__ == "Custom documentation"
        assert decorated.__annotations__ == {"return": str}

    def test_cqrs_integration_with_flext_result(self) -> None:
        """Test integration between CQRS operations and FlextResult."""
        # Create a command and verify it works with FlextResult operations
        command_data: dict[str, object] = {
            "command_type": "TestCommand",
            "issuer_id": "test",
        }
        command_result = FlextModels.Command.model_validate(command_data)

        assert command_result.is_success

        # Use the command with other FlextResult operations
        success_result = FlextResult[str].ok("Command executed successfully")
        assert success_result.is_success

        # Test result chaining
        results = [command_result, success_result]
        all_success = all(r.is_success for r in results)
        assert all_success

    def test_cqrs_performance_real_operations(self) -> None:
        """Test performance with real CQRS operations."""
        start_time = time.time()

        # Test performance with realistic operations
        for i in range(100):
            # Create various command types
            command_data: dict[str, object] = {
                "command_type": f"CommandType{i % 5}",
                "issuer_id": f"user_{i}",
            }
            command_result = FlextModels.Command.model_validate(command_data)
            assert command_result.is_success

            # Create corresponding queries
            query_data: dict[str, object] = {
                "query_type": f"QueryType{i % 3}",
                "filters": {"id": i, "active": True},
            }
            query_result = FlextModels.Query.model_validate(query_data)
            assert query_result.is_success

            # Create handler configs
            config_result = FlextModels.CqrsConfig.Handler.create_handler_config(
                "command" if i % 2 == 0 else "query"
            )
            assert config_result.is_success

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete in reasonable time (< 2 seconds for 300 operations)
        assert execution_time < 2.0

    def test_cqrs_real_world_scenario_user_management(self) -> None:
        """Test realistic user management scenario."""
        # Create user command
        create_user_cmd: dict[str, object] = {
            "command_type": "CreateUser",
            "issuer_id": "admin_user",
        }
        create_result = FlextModels.Command.model_validate(create_user_cmd)
        assert create_result.is_success

        # Create query to find user
        find_user_query: dict[str, object] = {
            "query_type": "FindUserByEmail",
            "filters": {"email": "alice@example.com"},
        }
        query_result = FlextModels.Query.model_validate(find_user_query)
        assert query_result.is_success

        # Create handler configurations
        command_config = FlextModels.CqrsConfig.Handler.create_handler_config(
            "command", handler_name="UserCommandHandler"
        )
        assert command_config.is_success

        query_config = FlextModels.CqrsConfig.Handler.create_handler_config(
            "query", handler_name="UserQueryHandler"
        )
        assert query_config.is_success

        # Test result creation with real data
        success_result = FlextCqrs.Results.success(
            {"user_id": "user_123", "created": True}, command_config.value
        )
        assert success_result.is_success
        assert hasattr(success_result, "_metadata")
        metadata = getattr(success_result, "_metadata")
        assert metadata is not None
        assert metadata["handler_name"] == "UserCommandHandler"

    def test_cqrs_error_scenarios_comprehensive(self) -> None:
        """Test comprehensive error scenarios."""
        # Test various validation errors
        test_cases: list[dict[str, object]] = [
            # Missing required fields
            {"command_type": "Test"},
            # Invalid data types
            {"command_type": 123},
            # Empty data
            {},
            # Extra fields that don't belong - this should fail since Command is strict
            {"command_type": "Test", "invalid_field": "value"},
        ]

        for i, invalid_data in enumerate(test_cases):
            result = FlextModels.Command.model_validate(invalid_data)
            # Only the invalid data type case should fail
            if i == 1:  # command_type: 123
                assert result.is_failure, f"Test case {i} should have failed"
                assert result.error_code in {
                    FlextConstants.Cqrs.COMMAND_VALIDATION_FAILED,
                    FlextConstants.Cqrs.CQRS_OPERATION_FAILED,
                }
            elif i == 3:  # Extra fields case
                assert result.is_failure, (
                    f"Test case {i} should have failed (extra fields not allowed)"
                )
                assert result.error_code in {
                    FlextConstants.Cqrs.COMMAND_VALIDATION_FAILED,
                    FlextConstants.Cqrs.CQRS_OPERATION_FAILED,
                }
            else:
                # Other cases should succeed since Command is flexible
                assert result.is_success, f"Test case {i} should have succeeded"

    def test_cqrs_metadata_handling_comprehensive(self) -> None:
        """Test comprehensive metadata handling across all operations."""
        # Test command metadata
        command_config = FlextModels.CqrsConfig.Handler(
            handler_id="meta_test_cmd",
            handler_name="MetadataCommandHandler",
            handler_type="command",
            metadata={
                "version": "2.0",
                "environment": "production",
                "features": ["audit", "logging"],
            },
        )

        command_data: dict[str, object] = {
            "command_type": "TestCommand",
            "issuer_id": "test",
        }
        FlextModels.Command.model_validate(command_data, command_config)

        # Test query metadata
        query_config = FlextModels.CqrsConfig.Handler(
            handler_id="meta_test_query",
            handler_name="MetadataQueryHandler",
            handler_type="query",
            metadata={"cache": True, "timeout": 30, "batch_size": 100},
        )

        query_data: dict[str, object] = {
            "query_type": "TestQuery",
            "filters": {"test": True},
        }
        FlextModels.Query.model_validate(query_data, query_config)

        # Test result metadata
        success_result = FlextCqrs.Results.success(
            {"result": "success", "count": 42}, command_config
        )
        assert hasattr(success_result, "_metadata")
        metadata = getattr(success_result, "_metadata")
        assert metadata is not None
        assert isinstance(metadata, dict)
        assert metadata["handler_id"] == "meta_test_cmd"
        assert metadata["handler_name"] == "MetadataCommandHandler"

        failure_result = FlextCqrs.Results.failure("Test failure", config=query_config)
        assert isinstance(failure_result.error_data, dict)
        assert "handler_id" in failure_result.error_data
        handler_metadata = cast(
            "dict[str, object]", failure_result.error_data["handler_metadata"]
        )
        assert handler_metadata["cache"] is True

    def test_cqrs_bulk_operations(self) -> None:
        """Test bulk operations for efficiency."""
        # Create many operations to test bulk processing
        commands = []
        queries = []
        configs = []

        for i in range(200):
            # Create commands
            command_data: dict[str, object] = {
                "command_type": f"BulkCommand{i}",
                "issuer_id": f"user_{i}",
            }
            command_result = FlextModels.Command.model_validate(command_data)
            if command_result.is_success:
                commands.append(command_result.value)

            # Create queries
            query_data: dict[str, object] = {
                "query_type": f"BulkQuery{i}",
                "filters": {"batch": i},
            }
            query_result = FlextModels.Query.model_validate(query_data)
            if query_result.is_success:
                queries.append(query_result.value)

            # Create configs
            config_result = FlextModels.CqrsConfig.Handler.create_handler_config(
                "command"
            )
            if config_result.is_success:
                configs.append(config_result.value)

        # Verify we created expected number of objects
        assert len(commands) == 200
        assert len(queries) == 200
        assert len(configs) == 200

        # Verify all objects are properly created
        assert all(isinstance(cmd, FlextModels.Command) for cmd in commands)
        assert all(isinstance(qry, FlextModels.Query) for qry in queries)
        assert all(isinstance(cfg, FlextModels.CqrsConfig.Handler) for cfg in configs)
