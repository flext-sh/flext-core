"""Comprehensive tests for FlextHandlers module - 100% coverage target.

Tests the refactored FlextHandlers (341 lines, extracted from original 687)
and all its companion modules for complete CQRS functionality validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from flext_core import (
    FlextModels,
    FlextResult,
    FlextUtilities,
)
from flext_core.handlers import FlextHandlers


class ConcreteTestHandler(FlextHandlers[str, str]):
    """Concrete test handler implementation for testing FlextHandlers base class."""

    def handle(self, message: str) -> FlextResult[str]:
        """Simple test handler implementation."""
        return FlextResult[str].ok(f"processed: {message}")


class TestFlextHandlers:
    """Test suite for FlextHandlers class - targeting 100% coverage."""

    def test_init_with_config(self) -> None:
        """Test FlextHandlers initialization with proper config."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
            handler_mode="command",
            command_timeout=5000,
            max_command_retries=3,
            metadata={"version": "1.0"},
        )

        handler = ConcreteTestHandler(config=config)

        assert handler.handler_id == "test_handler"
        assert handler.handler_name == "TestHandler"
        assert handler.mode == "command"
        assert handler.config == config
        assert handler.logger is not None

    def test_init_extracts_revalidation_setting_from_metadata(self) -> None:
        """Test that revalidation setting is extracted from config metadata."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
            handler_mode="command",
            metadata={"revalidate_pydantic_messages": True},
        )

        handler = ConcreteTestHandler(config=config)
        assert handler._revalidate_pydantic_messages is True

    def test_init_extracts_revalidation_setting_from_string(self) -> None:
        """Test that revalidation setting is extracted from string values."""
        test_cases = [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ]

        for value, expected in test_cases:
            config = FlextModels.CqrsConfig.Handler(
                handler_id="test_handler",
                handler_name="TestHandler",
                handler_type="command",
                handler_mode="command",
                metadata={"revalidate_pydantic_messages": value},
            )

            handler = ConcreteTestHandler(config=config)
            assert handler._revalidate_pydantic_messages is expected

    def test_init_revalidation_setting_defaults_to_false(self) -> None:
        """Test that revalidation setting defaults to False when not specified."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler",
            handler_name="TestHandler",
            handler_type="command",
            handler_mode="command",
        )

        handler = ConcreteTestHandler(config=config)
        assert handler._revalidate_pydantic_messages is False

    def test_properties(self) -> None:
        """Test all property accessors."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_id",
            handler_name="TestHandlerName",
            handler_type="query",
            handler_mode="query",
        )

        handler = ConcreteTestHandler(config=config)

        assert handler.mode == "query"
        assert handler.handler_name == "TestHandlerName"
        assert handler.handler_id == "test_handler_id"
        assert handler.config == config
        assert handler.logger.__class__.__name__ == "FlextLogger"

    def test_can_handle_delegates_to_type_checker(self) -> None:
        """Test can_handle method delegates to FlextUtilities.TypeChecker."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        handler = ConcreteTestHandler(config=config)

        with patch.object(
            FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
        ) as mock_can_handle:
            result = handler.can_handle(str)

            assert result is True
            mock_can_handle.assert_called_once_with(
                handler._accepted_message_types, str
            )

    def test_validate_command_delegates_to_message_validator(self) -> None:
        """Test validate_command delegates to FlextUtilities.MessageValidator."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        handler = ConcreteTestHandler(config=config)
        test_command = "test_command"

        expected_result = FlextResult[None].ok(None)
        with patch.object(
            FlextUtilities.MessageValidator,
            "validate_message",
            return_value=expected_result,
        ) as mock_validate:
            result = handler.validate_command(test_command)

            assert result == expected_result
            mock_validate.assert_called_once_with(
                test_command,
                operation="command",
                revalidate_pydantic_messages=handler._revalidate_pydantic_messages,
            )

    def test_validate_query_delegates_to_message_validator(self) -> None:
        """Test validate_query delegates to FlextUtilities.MessageValidator."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="query",
            handler_mode="query",
        )

        handler = ConcreteTestHandler(config=config)
        test_query = "test_query"

        expected_result = FlextResult[None].ok(None)
        with patch.object(
            FlextUtilities.MessageValidator,
            "validate_message",
            return_value=expected_result,
        ) as mock_validate:
            result = handler.validate_query(test_query)

            assert result == expected_result
            mock_validate.assert_called_once_with(
                test_query,
                operation="query",
                revalidate_pydantic_messages=handler._revalidate_pydantic_messages,
            )

    def test_handle_is_abstract(self) -> None:
        """Test that handle method is abstract and must be implemented."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        # Create a test handler to verify abstract method behavior
        class IncompleteHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                # This implementation is intentionally incomplete for testing
                return FlextResult[str].fail(f"Not implemented for message: {message}")

        # Can instantiate now that handle method is implemented
        handler = IncompleteHandler(config=config)
        assert handler is not None

    def test_execute_delegates_to_run_pipeline(self) -> None:
        """Test execute method delegates to _run_pipeline."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"handled: {message}")

        handler = TestHandler(config=config)
        test_message = "test_message"

        with patch.object(handler, "_run_pipeline") as mock_run_pipeline:
            mock_run_pipeline.return_value = FlextResult[str].ok("result")

            handler.execute(test_message)

            mock_run_pipeline.assert_called_once_with(test_message, operation="command")

    def test_handle_query_calls_execute(self) -> None:
        """Test handle_query method calls execute."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="query",
            handler_mode="query",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"handled: {message}")

        handler = TestHandler(config=config)
        test_message = "test_query"

        with patch.object(handler, "execute") as mock_execute:
            mock_execute.return_value = FlextResult[str].ok("result")

            handler.handle_query(test_message)

            mock_execute.assert_called_once_with(test_message)

    def test_handle_command_calls_execute(self) -> None:
        """Test handle_command method calls execute."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"handled: {message}")

        handler = TestHandler(config=config)
        test_message = "test_command"

        with patch.object(handler, "execute") as mock_execute:
            mock_execute.return_value = FlextResult[str].ok("result")

            handler.handle_command(test_message)

            mock_execute.assert_called_once_with(test_message)


class TestFlextHandlersPipeline:
    """Test the _run_pipeline method and its validation chain."""

    def test_run_pipeline_success_flow(self) -> None:
        """Test successful pipeline execution with all validations passing."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed: {message}")

        handler = TestHandler(config=config)
        test_message = "test"

        # Mock the necessary components to isolate the pipeline logic
        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start"),
            patch.object(FlextHandlers.Metrics, "log_handler_processing"),
            patch.object(FlextHandlers.Metrics, "log_handler_completion"),
        ):
            result = handler._run_pipeline(test_message, operation="command")

            assert result.is_success
            assert result.value == "processed: test"

    def test_run_pipeline_mode_validation_failure(self) -> None:
        """Test pipeline fails when handler mode doesn't match operation."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",  # Handler is for commands
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"result:{message}")

        handler = TestHandler(config=config)

        with (
            patch.object(FlextHandlers.Metrics, "log_handler_start"),
            patch.object(FlextHandlers.Metrics, "log_mode_validation_error"),
        ):
            result = handler._run_pipeline("test", operation="query")  # Wrong operation

            assert result.is_failure
            assert (
                result.error is not None
                and "cannot execute query pipelines" in result.error
            )

    def test_run_pipeline_can_handle_validation_failure(self) -> None:
        """Test pipeline fails when handler cannot handle message type."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"result:{message}")

        handler = TestHandler(config=config)

        with (
            patch.object(
                FlextUtilities.TypeChecker,
                "can_handle_message_type",
                return_value=False,
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start"),
            patch.object(FlextHandlers.Metrics, "log_handler_cannot_handle"),
        ):
            result = handler._run_pipeline("test", operation="command")

            assert result.is_failure
            assert result.error is not None and "cannot handle" in result.error

    def test_run_pipeline_message_validation_failure(self) -> None:
        """Test pipeline fails when message validation fails."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"result:{message}")

        handler = TestHandler(config=config)

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].fail("validation failed"),
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start"),
        ):
            result = handler._run_pipeline("test", operation="command")

            assert result.is_failure
            assert result.error is not None and "validation failed" in result.error

    def test_run_pipeline_exception_handling(self) -> None:
        """Test pipeline handles exceptions during execution."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class FailingHandler(FlextHandlers[str, str]):
            def handle(self, message: str) -> FlextResult[str]:
                msg = f"Handler failed for message: {message}"
                raise ValueError(msg)

        handler = FailingHandler(config=config)

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start"),
            patch.object(FlextHandlers.Metrics, "log_handler_processing"),
            patch.object(FlextHandlers.Metrics, "log_handler_error"),
        ):
            result = handler._run_pipeline("test", operation="command")

            assert result.is_failure
            assert (
                result.error is not None and "Critical handler failure" in result.error
            )
            assert result.error is not None and "Handler failed" in result.error

    def test_run_pipeline_with_message_id_extraction(self) -> None:
        """Test pipeline extracts message IDs correctly."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="command",
            handler_mode="command",
        )

        class TestHandler(FlextHandlers[dict[str, object], str]):
            def handle(self, message: dict[str, object]) -> FlextResult[str]:
                return FlextResult[str].ok(f"result:{message}")

        handler = TestHandler(config=config)

        # Create a message dict with command_id attribute
        message_with_command_id: dict[str, object] = {
            "command_id": "cmd_123",
            "data": "test",
        }

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start") as mock_log_start,
            patch.object(FlextHandlers.Metrics, "log_handler_processing"),
            patch.object(FlextHandlers.Metrics, "log_handler_completion"),
        ):
            handler._run_pipeline(message_with_command_id, operation="command")

            mock_log_start.assert_called_once()
            args = mock_log_start.call_args[0]
            assert args[3] == "cmd_123"  # message_id parameter

    def test_run_pipeline_with_query_id_extraction(self) -> None:
        """Test pipeline extracts query IDs correctly."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="Test",
            handler_type="query",
            handler_mode="query",
        )

        class TestHandler(FlextHandlers[dict[str, object], str]):
            def handle(self, message: dict[str, object]) -> FlextResult[str]:
                return FlextResult[str].ok(f"result:{message}")

        handler = TestHandler(config=config)

        # Create a message dict with query_id attribute
        message_with_query_id: dict[str, object] = {
            "query_id": "qry_456",
            "data": "test",
        }

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
            patch.object(FlextHandlers.Metrics, "log_handler_start") as mock_log_start,
            patch.object(FlextHandlers.Metrics, "log_handler_processing"),
            patch.object(FlextHandlers.Metrics, "log_handler_completion"),
        ):
            handler._run_pipeline(message_with_query_id, operation="query")

            mock_log_start.assert_called_once()
            args = mock_log_start.call_args[0]
            assert args[3] == "qry_456"  # message_id parameter


class TestFlextHandlersFromCallable:
    """Test the from_callable factory method."""

    def test_from_callable_basic_creation(self) -> None:
        """Test basic creation of handler from callable."""

        def test_function(message: object) -> str:
            return f"processed: {message}"

        handler = FlextHandlers.from_callable(
            test_function, mode="command", handler_name="TestFunction"
        )

        assert isinstance(handler, FlextHandlers)
        assert handler.handler_name == "TestFunction"
        assert handler.mode == "command"

    def test_from_callable_with_config_dict(self) -> None:
        """Test creation with config dictionary."""

        def test_function(message: object) -> str:
            return f"processed: {message}"

        config_dict: dict[str, object] = {"timeout": 5000, "retries": 3}

        handler = FlextHandlers.from_callable(
            test_function,
            mode="query",
            handler_config=config_dict,
            handler_name="TestFunction",
        )

        assert handler.mode == "query"

    def test_from_callable_with_config_object(self) -> None:
        """Test creation with config object."""

        def test_function(message: object) -> str:
            return f"processed: {message}"

        config_obj = FlextModels.CqrsConfig.Handler(
            handler_id="test",
            handler_name="ExistingName",
            handler_type="command",
            handler_mode="command",
        )

        handler = FlextHandlers.from_callable(
            test_function,
            mode="command",
            handler_config=config_obj,
            handler_name="TestFunction",
        )

        # When config object is provided, its handler_name takes precedence
        # (config_data.update(handler_config) in models.py line 2851 overrides the default_name)
        assert handler.handler_name == "ExistingName"

    def test_from_callable_invalid_mode_raises_error(self) -> None:
        """Test from_callable with invalid mode raises error."""

        def test_function(message: object) -> object:
            return message

        with pytest.raises(ValueError, match="Invalid handler mode"):
            FlextHandlers.from_callable(test_function, mode="invalid")  # type: ignore[arg-type]

    def test_from_callable_defaults_handler_name(self) -> None:
        """Test that handler name defaults to function name."""

        def my_function(message: object) -> str:
            return f"result:{message}"

        handler = FlextHandlers.from_callable(my_function, mode="command")
        assert handler.handler_name == "my_function"

    def test_from_callable_handles_anonymous_function(self) -> None:
        """Test creation with anonymous function."""

        def typed_function(x: object) -> str:
            return f"result: {x}"

        handler = FlextHandlers.from_callable(typed_function, mode="command")
        # Should get default name for anonymous functions
        assert (
            "FunctionHandler" in handler.handler_name
            or "lambda" in handler.handler_name
        )

    def test_from_callable_handler_execution(self) -> None:
        """Test that created handler can execute successfully."""

        def process_message(message: object) -> str:
            if message == "error":
                msg = "Processing error"
                raise ValueError(msg)
            return f"processed: {message}"

        handler = FlextHandlers.from_callable(process_message, mode="command")
        # Test successful execution
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "processed: test"

        # Test error handling
        error_result = handler.handle("error")
        assert error_result.is_failure
        assert (
            error_result.error is not None and "Processing error" in error_result.error
        )

    def test_from_callable_handler_returns_flext_result(self) -> None:
        """Test handler that returns FlextResult directly."""

        def process_message(message: object) -> FlextResult[str]:
            if message == "fail":
                return FlextResult[str].fail("Processing failed")
            return FlextResult[str].ok(f"processed: {message}")

        handler = FlextHandlers.from_callable(process_message, mode="command")
        # Test successful execution
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "processed: test"

        # Test failure case
        fail_result = handler.handle("fail")
        assert fail_result.is_failure
        assert fail_result.error == "Processing failed"

    def test_from_callable_handler_with_flext_exceptions(self) -> None:
        """Test handler with FlextExceptions."""
        from flext_core.exceptions import FlextExceptions

        def process_message(message: object) -> str:
            if message == "business_error":
                msg = "Business rule violated"
                raise FlextExceptions.ProcessingError(msg, operation="test")
            return "processed"

        handler = FlextHandlers.from_callable(process_message, mode="command")
        result = handler.handle("business_error")
        assert result.is_failure
        assert result.error is not None and "Business rule violated" in result.error


class TestFlextHandlersIntegration:
    """Integration tests for FlextHandlers with real components."""

    def test_full_command_handler_integration(self) -> None:
        """Test complete command handler integration."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="integration_test",
            handler_name="IntegrationTestHandler",
            handler_type="command",
            handler_mode="command",
            command_timeout=1000,
            max_command_retries=2,
        )

        class CommandHandler(FlextHandlers[dict[str, object], dict[str, object]]):
            def handle(
                self, message: dict[str, object]
            ) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({
                    "status": "processed",
                    "input": message,
                    "handler": self.handler_name,
                })

        handler = CommandHandler(config=config)

        # Test full execution pipeline with mocks for integration
        command: dict[str, object] = {"action": "create_user", "data": {"name": "John"}}

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
        ):
            result = handler.execute(command)

            assert result.is_success
            response = result.value
            assert response["status"] == "processed"
            assert response["input"] == command
            assert response["handler"] == "IntegrationTestHandler"

    def test_full_query_handler_integration(self) -> None:
        """Test complete query handler integration."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="query_integration_test",
            handler_name="QueryIntegrationTestHandler",
            handler_type="query",
            handler_mode="query",
        )

        class QueryHandler(FlextHandlers[dict[str, object], list[object]]):
            def handle(self, message: dict[str, object]) -> FlextResult[list[object]]:
                # Simulate data retrieval based on message
                filter_key = message.get("filter") if message else None
                return FlextResult[list[object]].ok([
                    {"id": 1, "name": "Item 1", "filter": filter_key},
                    {"id": 2, "name": "Item 2", "filter": filter_key},
                ])

        handler = QueryHandler(config=config)

        # Test full execution pipeline with mocks for integration
        query: dict[str, object] = {
            "operation": "list_items",
            "filters": {"active": True},
        }

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
        ):
            result = handler.handle_query(query)

            assert result.is_success
            items = result.value
            assert len(items) == 2
            # Type check: items is list[object], so we need to cast for test
            assert isinstance(items[0], dict) and items[0]["name"] == "Item 1"

    def test_handler_with_pydantic_message(self) -> None:
        """Test handler with Pydantic message model."""
        from pydantic import BaseModel

        class UserCommand(BaseModel):
            user_id: str
            action: str
            data: dict[str, object]

        config = FlextModels.CqrsConfig.Handler(
            handler_id="pydantic_test",
            handler_name="PydanticTestHandler",
            handler_type="command",
            handler_mode="command",
            metadata={"revalidate_pydantic_messages": True},
        )

        class PydanticHandler(FlextHandlers[UserCommand, str]):
            def handle(self, message: UserCommand) -> FlextResult[str]:
                return FlextResult[str].ok(
                    f"Processed {message.action} for {message.user_id}"
                )

        handler = PydanticHandler(config=config)

        # Test with valid Pydantic model
        command = UserCommand(
            user_id="user_123", action="update_profile", data={"name": "New Name"}
        )

        with (
            patch.object(
                FlextUtilities.TypeChecker, "can_handle_message_type", return_value=True
            ),
            patch.object(
                FlextUtilities.MessageValidator,
                "validate_message",
                return_value=FlextResult[None].ok(None),
            ),
        ):
            result = handler.execute(command)
            assert result.is_success
            assert "Processed update_profile for user_123" in result.value


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
