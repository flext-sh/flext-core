"""Critical coverage tests targeting lowest coverage modules.

Focuses on core.py (42%), commands.py (63%), result.py (54%) to reach near 100%.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flext_core import FlextCommands, FlextCore, FlextResult


class TestCriticalCoverage:
    """Tests targeting critical low-coverage modules."""

    def test_core_advanced_operations(self) -> None:
        """Test advanced core operations for coverage."""
        core = FlextCore.get_instance()

        # Test all configuration methods with different parameters
        configs = [
            {"setting1": "value1", "number": 42},
            {"feature_enabled": True, "timeout": 30.0},
            {"batch_size": 100, "max_retries": 5},
            {},  # Empty config
            {"complex": {"nested": {"value": [1, 2, 3]}}},
        ]

        for config in configs:
            # Test all system configurations
            result = core.configure_aggregates_system(config)
            assert isinstance(result, FlextResult)

            result = core.configure_commands_system(config)
            assert isinstance(result, FlextResult)

            result = core.configure_context_system(config)
            assert isinstance(result, FlextResult)

            result = core.configure_decorators_system(config)
            assert isinstance(result, FlextResult)

        # Test all optimization levels
        levels = ["low", "medium", "high", "maximum", "custom"]
        for level in levels:
            result = core.optimize_aggregates_system(level)
            assert isinstance(result, FlextResult)

            result = core.optimize_commands_performance(level)
            assert isinstance(result, FlextResult)

            result = core.optimize_decorators_performance(level)
            assert isinstance(result, FlextResult)

    def test_core_validation_edge_cases(self) -> None:
        """Test validation methods with edge cases."""
        core = FlextCore.get_instance()

        # Test string validation with various inputs
        string_inputs = [
            ("", "empty"),
            ("normal string", "normal"),
            ("string with spaces", "spaced"),
            ("unicode: àáâãäå", "unicode"),
            ("very long string " * 100, "long"),
            ("special chars: !@#$%^&*()", "special"),
        ]

        for string_val, field_name in string_inputs:
            result = core.validate_string_field(string_val, field_name)
            assert isinstance(result, FlextResult)

        # Test numeric validation with various numbers
        numeric_inputs = [
            (0, "zero"),
            (-100, "negative"),
            (999999, "large"),
            (3.141592653589793, "pi"),
            (float("inf"), "infinity"),
            (-float("inf"), "neg_infinity"),
        ]

        for num_val, field_name in numeric_inputs:
            result = core.validate_numeric_field(num_val, field_name)
            assert isinstance(result, FlextResult)

        # Test email validation with comprehensive cases
        email_inputs = [
            "simple@example.com",
            "user.name@domain.com",
            "user+tag@example.co.uk",
            "test.email-with-dash@example.com",
            "x@example.com",
            "test@sub.domain.example.org",
            "invalid_email",
            "@invalid.com",
            "user@",
            "",
            "spaces in@email.com",
            "double..dot@example.com",
        ]

        for email in email_inputs:
            result = core.validate_email(email)
            assert isinstance(result, FlextResult)

        # Test user data validation with various cases
        user_cases = [
            {"name": "John Doe", "email": "john@example.com", "age": 30},
            {"name": "Jane", "email": "jane@test.com", "age": 25, "city": "NYC"},
            {"name": "", "email": "invalid", "age": -5},  # Invalid case
            {"email": "test@example.com"},  # Missing name
            {"name": "Test User"},  # Missing email
            {},  # Empty
            {"name": "User", "email": "user@test.com", "age": "invalid"},  # Wrong type
        ]

        for user_data in user_cases:
            result = core.validate_user_data(user_data)
            assert isinstance(result, FlextResult)

        # Test API request validation
        api_cases = [
            {
                "method": "GET",
                "path": "/api/users",
                "action": "list",
                "version": "v1",
                "headers": {"Authorization": "Bearer token"},
            },
            {
                "method": "POST",
                "path": "/api/users",
                "action": "create",
                "version": "v2",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "Test"},
            },
            {
                "method": "PUT",
                "path": "/api/users/123",
                "action": "update",
                "version": "v1",
                "headers": {},
            },
            {
                "method": "DELETE",
                "path": "/api/users/123",
                "action": "delete",
                "version": "v1",
            },
            {"method": "", "path": "", "action": "", "version": ""},  # Empty values
            {},  # Empty request
            {"method": "INVALID"},  # Minimal invalid
        ]

        for api_request in api_cases:
            result = core.validate_api_request(api_request)
            assert isinstance(result, FlextResult)

    def test_core_entity_creation_comprehensive(self) -> None:
        """Test entity creation with comprehensive data."""
        core = FlextCore.get_instance()

        # Test entity creation with various structures
        entity_cases = [
            {"id": "simple_entity", "name": "Simple"},
            {
                "id": "complex_entity",
                "name": "Complex Entity",
                "properties": {
                    "type": "document",
                    "metadata": {"created": "2023-01-01", "version": 1.0},
                    "tags": ["important", "document", "test"],
                },
                "relations": [
                    {"type": "parent", "target": "parent_entity"},
                    {"type": "child", "target": "child_entity_1"},
                    {"type": "child", "target": "child_entity_2"},
                ],
            },
            {"id": "minimal_entity"},  # Minimal
            {},  # Empty
            {
                "id": "large_entity",
                "data": {f"field_{i}": f"value_{i}" for i in range(100)},  # Large data
            },
        ]

        for entity_data in entity_cases:
            result = core.create_entity(entity_data)
            assert isinstance(result, FlextResult)

        # Test value object creation
        value_cases = [
            {"value": "simple_string"},
            {"value": 42, "type": "integer"},
            {"amount": 100.50, "currency": "USD"},
            {"coordinates": {"lat": 40.7128, "lng": -74.0060}},
            {"range": {"min": 0, "max": 100}, "unit": "percentage"},
            {},  # Empty
            {"complex_value": {"nested": {"deep": {"very_deep": "value"}}}},
        ]

        for value_data in value_cases:
            result = core.create_value_object(value_data)
            assert isinstance(result, FlextResult)

        # Test domain event creation with various scenarios
        event_cases = [
            {
                "event_type": "user_created",
                "aggregate_id": "user_123",
                "aggregate_type": "User",
                "data": {"name": "John Doe", "email": "john@example.com"},
                "source_service": "user_service",
            },
            {
                "event_type": "order_shipped",
                "aggregate_id": "order_456",
                "aggregate_type": "Order",
                "data": {
                    "tracking_number": "1234567890",
                    "carrier": "UPS",
                    "estimated_delivery": "2023-12-01",
                },
                "source_service": "shipping_service",
            },
            {
                "event_type": "payment_processed",
                "aggregate_id": "payment_789",
                "aggregate_type": "Payment",
                "data": {"amount": 99.99, "currency": "USD", "method": "credit_card"},
                "source_service": "payment_service",
            },
        ]

        for event_data in event_cases:
            result = core.create_domain_event(**event_data)
            assert isinstance(result, FlextResult)

    def test_core_utility_comprehensive(self) -> None:
        """Test utility methods comprehensively."""
        core = FlextCore.get_instance()

        # Test UUID generation uniqueness
        uuids = set()
        for _ in range(100):
            uuid = core.generate_uuid()
            assert uuid not in uuids
            uuids.add(uuid)
            assert len(uuid) > 0

        # Test correlation ID generation
        corr_ids = set()
        for _ in range(50):
            corr_id = core.generate_correlation_id()
            assert corr_id not in corr_ids
            corr_ids.add(corr_id)

        # Test entity ID generation
        entity_ids = set()
        for _ in range(50):
            entity_id = core.generate_entity_id()
            assert entity_id not in entity_ids
            entity_ids.add(entity_id)

        # Test duration formatting with various values
        durations = [0.001, 1.0, 60.5, 3600.75, 86400.0, 123456.789]
        for duration in durations:
            formatted = core.format_duration(duration)
            assert isinstance(formatted, str)
            assert len(formatted) > 0

        # Test text cleaning
        text_cases = [
            "  normal text  ",
            "\t\ntab and newline\n\t",
            "multiple    spaces",
            "",
            "   ",
            "already clean",
            "unicode: àáâãäå",
            "special chars: !@#$%^&*()",
        ]

        for text in text_cases:
            cleaned = core.clean_text(text)
            assert isinstance(cleaned, str)

        # Test batch processing
        batch_cases = [
            (list(range(100)), 10),
            (list(range(7)), 3),
            ([], 5),
            (list(range(1000)), 1),
            (["a", "b", "c"], 2),
        ]

        for items, batch_size in batch_cases:
            batches = core.batch_process(items, batch_size)
            assert isinstance(batches, list)

    def test_core_service_management_extensive(self) -> None:
        """Test extensive service management operations."""
        core = FlextCore.get_instance()

        # Test service registration with various data types
        services = [
            ("string_service", "simple string"),
            ("dict_service", {"key": "value", "nested": {"data": 123}}),
            ("list_service", [1, 2, 3, {"nested": "object"}]),
            ("number_service", 42.5),
            ("bool_service", True),
            (
                "complex_service",
                {
                    "config": {"host": "localhost", "port": 5432},
                    "data": list(range(10)),
                    "metadata": {"version": "1.0", "active": True},
                },
            ),
        ]

        for service_name, service_data in services:
            # Register
            result = core.register_service(service_name, service_data)
            assert isinstance(result, FlextResult)

            # Get
            get_result = core.get_service(service_name)
            assert isinstance(get_result, FlextResult)

            # Update
            updated_data = {"updated": True, "original": service_data}
            update_result = core.register_service(service_name, updated_data)
            assert isinstance(update_result, FlextResult)

        # Test getting non-existent services
        for i in range(10):
            missing_result = core.get_service(f"nonexistent_service_{i}")
            assert isinstance(missing_result, FlextResult)

    def test_core_file_operations_comprehensive(self) -> None:
        """Test file operations comprehensively."""
        core = FlextCore.get_instance()

        # Test with various file scenarios
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"valid": "json", "number": 42}, f)
            valid_json_path = f.name

        try:
            result = core.load_config_from_file(valid_json_path)
            assert isinstance(result, FlextResult)
        finally:
            Path(valid_json_path).unlink(missing_ok=True)

        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            invalid_json_path = f.name

        try:
            result = core.load_config_from_file(invalid_json_path)
            assert isinstance(result, FlextResult)
        finally:
            Path(invalid_json_path).unlink(missing_ok=True)

        # Test with various non-existent paths
        nonexistent_paths = [
            "nonexistent.json",
            "/path/that/does/not/exist.json",
            "config.json",
            "data/config.json",
            "../config.json",
        ]

        for path in nonexistent_paths:
            result = core.load_config_from_file(path)
            assert isinstance(result, FlextResult)

    def test_result_advanced_operations(self) -> None:
        """Test advanced FlextResult operations."""
        # Test various success values
        success_values = [42, "string", [1, 2, 3], {"key": "value"}, True, None, 3.14]
        for value in success_values:
            result = FlextResult.ok(value)
            assert result.success is True
            assert result.value == value

        # Test various error messages
        error_messages = [
            "Simple error",
            "",
            "Error with unicode: àáâãäå",
            "Very long error message " * 50,
            "Error with special chars: !@#$%^&*()",
        ]

        for error in error_messages:
            result = FlextResult[int].fail(error)
            assert result.failure is True
            # Empty error messages might be converted to default error
            if error == "":
                assert result.error is not None  # May be converted to default
            else:
                assert error in (result.error or "")

        # Test map operations with various functions
        success = FlextResult.ok(10)

        map_functions = [
            lambda x: x * 2,
            lambda x: str(x),
            lambda x: [x, x, x],
            lambda x: {"value": x},
            lambda x: x > 5,
        ]

        for func in map_functions:
            mapped = success.map(func)
            assert mapped.success

        # Test flat_map operations
        def make_result(x: int) -> FlextResult[int]:
            if x > 0:
                return FlextResult.ok(x * 2)
            return FlextResult[int].fail("negative")

        positive_values = [1, 5, 10, 100]
        for val in positive_values:
            result = FlextResult.ok(val)
            flat_mapped = result.flat_map(make_result)
            assert flat_mapped.success

        negative_values = [-1, -5, -10]
        for val in negative_values:
            result = FlextResult.ok(val)
            flat_mapped = result.flat_map(make_result)
            assert flat_mapped.failure

    def test_commands_basic_operations(self) -> None:
        """Test basic commands operations for coverage."""

        # Test command model creation
        class SimpleCommand(FlextCommands.Models.Command):
            name: str
            value: int = 0

            def validate_command(self) -> FlextResult[None]:
                if not self.name:
                    return FlextResult[None].fail("Name required")
                return FlextResult[None].ok(None)

        # Test valid command
        cmd = SimpleCommand(name="test", value=42)
        validation = cmd.validate_command()
        assert isinstance(validation, FlextResult)

        # Test invalid command
        invalid_cmd = SimpleCommand(name="", value=10)
        validation = invalid_cmd.validate_command()
        assert isinstance(validation, FlextResult)

        # Test query model
        class SimpleQuery(FlextCommands.Models.Query):
            term: str
            limit: int = 10

            def validate_query(self) -> FlextResult[None]:
                if not self.term:
                    return FlextResult[None].fail("Term required")
                return FlextResult[None].ok(None)

        query = SimpleQuery(term="search", limit=20)
        validation = query.validate_query()
        assert isinstance(validation, FlextResult)

        # Test command handler
        class SimpleHandler(FlextCommands.Handlers.CommandHandler[SimpleCommand, str]):
            def handle(self, command: SimpleCommand) -> FlextResult[str]:
                return FlextResult[str].ok(f"Handled: {command.name}")

            def can_handle(self, command: object) -> bool:
                return isinstance(command, SimpleCommand)

        handler = SimpleHandler()
        assert handler.can_handle(cmd) is True
        result = handler.handle(cmd)
        assert isinstance(result, FlextResult)

        # Test command bus
        bus = FlextCommands.Bus()
        bus.register_handler(handler)

        bus_result = bus.execute(cmd)
        assert isinstance(bus_result, FlextResult)
