"""Tests for SOLID principles implementation in handlers.

Tests specifically for the SOLID refactoring of handlers.py:
- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- Liskov Substitution Principle (LSP)
- Interface Segregation Principle (ISP)
- Dependency Inversion Principle (DIP)
"""

from __future__ import annotations

import pytest

from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult


class TestSOLIDPrinciples:
    """Test SOLID principles in refactored handlers."""

    @pytest.mark.architecture
    def test_interface_segregation_principle(self) -> None:
        """Test ISP - handlers only implement needed interfaces."""

        class CustomValidator:
            """Custom validator implementing only validation protocol."""

            def validate_message(self, message: object) -> FlextResult[object]:
                if isinstance(message, str) and len(message) > 100:
                    return FlextResult.fail("Message too long")
                return FlextResult.ok(message)

        class CustomAuthorizer:
            """Custom authorizer implementing only authorization protocol."""

            def authorize_query(self, query: object) -> FlextResult[None]:
                if isinstance(query, dict) and query.get("REDACTED_LDAP_BIND_PASSWORD_only"):
                    return FlextResult.fail("Admin access required")
                return FlextResult.ok(None)

        # Test that we can inject specific functionality
        validator = CustomValidator()
        authorizer = CustomAuthorizer()

        # Command handler with validator
        command_handler = FlextHandlers.CommandHandler(
            handler_name="test_cmd", validator=validator
        )

        # Query handler with authorizer
        query_handler: FlextHandlers.QueryHandler[object, object] = (
            FlextHandlers.QueryHandler(handler_name="test_query", authorizer=authorizer)
        )

        # Test validation works
        long_message = "x" * 101
        validation_result = command_handler.validate_command(long_message)
        assert validation_result.is_failure
        assert "too long" in (validation_result.error or "")

        # Test authorization works
        REDACTED_LDAP_BIND_PASSWORD_query = {"REDACTED_LDAP_BIND_PASSWORD_only": True}
        auth_result = query_handler.authorize_query(REDACTED_LDAP_BIND_PASSWORD_query)
        assert auth_result.is_failure
        assert "Admin access required" in (auth_result.error or "")

    def test_single_responsibility_principle(self) -> None:
        """Test SRP - each handler has single responsibility."""

        # CommandHandler is only responsible for command processing
        command_handler = FlextHandlers.CommandHandler("cmd_handler")

        # It handles commands
        result = command_handler.handle("test_command")
        assert result.success

        # It validates commands (single responsibility)
        validation_result = command_handler.validate_command("test")
        assert validation_result.success

        # It provides metrics (single responsibility)
        metrics = command_handler.get_metrics()
        assert "handler_type" in metrics
        assert metrics["handler_type"] == "CommandHandler"
        assert "commands_processed" in metrics
        assert "success_rate" in metrics

    def test_open_closed_principle(self) -> None:
        """Test OCP - handlers are open for extension, closed for modification."""

        class ExtendedCommandHandler(FlextHandlers.CommandHandler):
            """Extended command handler without modifying base."""

            def handle(self, command: object) -> FlextResult[object]:
                # Extended functionality without modifying base class
                if isinstance(command, str) and command.startswith("SPECIAL_"):
                    return FlextResult.ok(f"Special handling: {command}")
                return super().handle(command)

        # Original handler works as before
        original_handler = FlextHandlers.CommandHandler("original")
        result = original_handler.handle("normal_command")
        assert result.success
        assert result.data == "normal_command"

        # Extended handler adds functionality without breaking original
        extended_handler = ExtendedCommandHandler("extended")

        # Normal command works
        normal_result = extended_handler.handle("normal_command")
        assert normal_result.success

        # Special command gets extended handling
        special_result = extended_handler.handle("SPECIAL_test")
        assert special_result.success
        assert "Special handling" in str(special_result.data or "")

    def test_dependency_inversion_principle(self) -> None:
        """Test DIP - handlers depend on abstractions, not concretions."""

        class MockValidator:
            """Mock validator for testing DIP."""

            def validate_message(self, message: object) -> FlextResult[object]:
                return FlextResult.fail("Mock validation failed")

        class MockMetricsCollector:
            """Mock metrics collector for testing DIP."""

            def get_metrics(self) -> dict[str, object]:
                return {"custom_metric": "mock_value"}

        # Handler depends on abstractions (protocols), not concrete classes
        mock_validator = MockValidator()
        mock_metrics = MockMetricsCollector()

        handler = FlextHandlers.CommandHandler(
            handler_name="di_test",
            validator=mock_validator,
            metrics_collector=mock_metrics,
        )

        # Validation uses injected dependency
        validation_result = handler.validate_command("test")
        assert validation_result.is_failure
        assert "Mock validation failed" in (validation_result.error or "")

        # Metrics use injected dependency
        metrics = handler.get_metrics()
        assert "custom_metric" in metrics
        assert metrics["custom_metric"] == "mock_value"

    def test_liskov_substitution_principle(self) -> None:
        """Test LSP - derived handlers can substitute base handlers."""

        class SpecialCommandHandler(FlextHandlers.CommandHandler):
            """Special command handler that substitutes base."""

            def handle(self, command: object) -> FlextResult[object]:
                return FlextResult.ok(f"SPECIAL: {command}")

        def process_with_any_handler(
            handler: FlextHandlers.CommandHandler, command: object
        ) -> FlextResult[object]:
            """Function that works with any command handler."""
            return handler.handle_with_hooks(command)

        # Base handler works
        base_handler = FlextHandlers.CommandHandler("base")
        base_result = process_with_any_handler(base_handler, "test")
        assert base_result.success

        # Derived handler can substitute base handler
        special_handler = SpecialCommandHandler("special")
        special_result = process_with_any_handler(special_handler, "test")
        assert special_result.success
        assert "SPECIAL:" in str(special_result.data or "")

        # Both handlers have same interface contracts
        assert hasattr(base_handler, "handle_with_hooks")
        assert hasattr(special_handler, "handle_with_hooks")
        assert hasattr(base_handler, "get_metrics")
        assert hasattr(special_handler, "get_metrics")

    def test_metrics_collection_improvement(self) -> None:
        """Test that SOLID refactoring improved metrics collection."""

        handler = FlextHandlers.CommandHandler("metrics_test")

        # Initial metrics
        initial_metrics = handler.get_metrics()
        assert initial_metrics["commands_processed"] == 0
        assert initial_metrics["successful_commands"] == 0
        assert initial_metrics["success_rate"] == 0.0

        # Process successful command
        result = handler.handle_with_hooks("success_command")
        assert result.success

        # Metrics updated
        updated_metrics = handler.get_metrics()
        assert updated_metrics["commands_processed"] == 1
        assert updated_metrics["successful_commands"] == 1
        assert updated_metrics["success_rate"] == 1.0

        # Process failed command by creating failing handler
        class FailingHandler(FlextHandlers.CommandHandler):
            def handle(self, command: object) -> FlextResult[object]:
                return FlextResult.fail("Intentional failure")

        failing_handler = FailingHandler("failing")
        fail_result = failing_handler.handle_with_hooks("fail_command")
        assert fail_result.is_failure

        # Failing handler metrics
        fail_metrics = failing_handler.get_metrics()
        assert fail_metrics["commands_processed"] == 1
        assert fail_metrics["successful_commands"] == 0
        assert fail_metrics["success_rate"] == 0.0

    def test_protocol_compliance(self) -> None:
        """Test that protocols are properly implemented."""

        # Test that handlers implement expected protocols
        handler = FlextHandlers.CommandHandler("protocol_test")

        # Should implement MessageHandler protocol
        assert hasattr(handler, "handle")
        assert hasattr(handler, "can_handle")

        # Should implement MetricsCollector protocol
        assert hasattr(handler, "get_metrics")

        # Test protocol methods work correctly
        assert callable(handler.handle)
        assert callable(handler.can_handle)
        assert callable(handler.get_metrics)

        # Test actual protocol compliance
        message = {"command": "test_message"}  # Use valid command object
        assert handler.can_handle(message) is True

        handle_result = handler.handle(message)
        assert isinstance(handle_result, FlextResult)

        metrics_result = handler.get_metrics()
        assert isinstance(metrics_result, dict)


class TestSOLIDIntegration:
    """Test integration of SOLID principles."""

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_complete_solid_workflow(self) -> None:
        """Test complete workflow using all SOLID principles."""

        class ProductionValidator:
            """Production-grade validator."""

            def validate_message(self, message: object) -> FlextResult[object]:
                if message is None:
                    return FlextResult.fail("Message cannot be None")
                if isinstance(message, str) and len(message) == 0:
                    return FlextResult.fail("Message cannot be empty")
                return FlextResult.ok(message)

        class ProductionAuthorizer:
            """Production-grade authorizer."""

            def authorize_query(self, query: object) -> FlextResult[None]:
                if isinstance(query, dict) and query.get("user_level", 0) < 1:
                    return FlextResult.fail("Insufficient user level")
                return FlextResult.ok(None)

        class ProductionMetrics:
            """Production-grade metrics collector."""

            def get_metrics(self) -> dict[str, object]:
                return {
                    "environment": "production",
                    "version": "1.0.0",
                    "feature_flags": ["solid_refactoring"],
                }

        # Create handlers with production dependencies
        validator = ProductionValidator()
        authorizer = ProductionAuthorizer()
        metrics = ProductionMetrics()

        command_handler = FlextHandlers.CommandHandler(
            handler_name="production_cmd",
            validator=validator,
            metrics_collector=metrics,
        )

        query_handler: FlextHandlers.QueryHandler[object, object] = (
            FlextHandlers.QueryHandler(
                handler_name="production_query", authorizer=authorizer
            )
        )

        # Test complete workflow
        # 1. Command processing with validation
        valid_command = "valid_command_data"
        cmd_result = command_handler.handle_with_hooks(valid_command)
        assert cmd_result.success

        # 2. Invalid command rejected
        invalid_command = ""
        invalid_result = command_handler.handle_with_hooks(invalid_command)
        assert invalid_result.is_failure
        assert invalid_result.error is not None
        assert "empty" in invalid_result.error

        # 3. Query processing with authorization
        authorized_query = {"user_level": 5, "data": "query_data"}
        query_result = query_handler.pre_handle(authorized_query)
        assert query_result.success

        # 4. Unauthorized query rejected
        unauthorized_query = {"user_level": 0, "data": "secret_data"}
        unauth_result = query_handler.pre_handle(unauthorized_query)
        assert unauth_result.is_failure
        assert unauth_result.error is not None
        assert "Insufficient user level" in unauth_result.error

        # 5. Metrics include production data
        cmd_metrics = command_handler.get_metrics()
        assert "environment" in cmd_metrics
        assert cmd_metrics["environment"] == "production"
        assert "version" in cmd_metrics
        processed_count = cmd_metrics.get("commands_processed", 0)
        assert isinstance(processed_count, int)
        assert processed_count >= 1
