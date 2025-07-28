"""Comprehensive tests for FlextUtilities consolidated functionality.

Tests all consolidated features following "entregar mais com muito menos" approach:
- FlextTypeGuards: Runtime type checking with TypeGuard support
- FlextGenerators: ID generation, timestamps, and entity metadata
- FlextFormatters: String formatting, data display, and sanitization
- FlextUtilities: Multiple inheritance consolidation with orchestration
- Performance tracking: Function metrics and observability
- DelegationMixin: Automatic method delegation patterns
"""

from __future__ import annotations

import math
import re
import time

import pytest

from flext_core.utilities import (
    DelegationMixin,
    FlextFormatters,
    FlextGenerators,
    FlextTypeGuards,
    FlextUtilities,
    clear_performance_metrics,
    get_performance_metrics,
    record_performance,
    track_performance,
)

pytestmark = [pytest.mark.unit, pytest.mark.patterns]


class TestFlextTypeGuards:
    """Test FlextTypeGuards runtime type checking functionality."""

    def test_has_attribute_detection(self) -> None:
        """Test has_attribute() with various object types."""

        # Object with attribute
        class TestObject:
            def __init__(self) -> None:
                self.test_attr = "value"

            def test_method(self) -> str:
                return "method_result"

        obj = TestObject()
        assert FlextTypeGuards.has_attribute(obj, "test_attr")
        assert FlextTypeGuards.has_attribute(obj, "test_method")
        assert not FlextTypeGuards.has_attribute(obj, "nonexistent_attr")

        # Built-in types
        test_list = [1, 2, 3]
        assert FlextTypeGuards.has_attribute(test_list, "append")
        assert FlextTypeGuards.has_attribute(test_list, "count")
        assert not FlextTypeGuards.has_attribute(test_list, "nonexistent")

        # Dictionary
        test_dict = {"key": "value"}
        assert FlextTypeGuards.has_attribute(test_dict, "keys")
        assert FlextTypeGuards.has_attribute(test_dict, "get")
        assert not FlextTypeGuards.has_attribute(test_dict, "nonexistent")

    def test_is_instance_of_validation(self) -> None:
        """Test is_instance_of() with various type checks."""
        # Basic types
        assert FlextTypeGuards.is_instance_of("test", str)
        assert FlextTypeGuards.is_instance_of(42, int)
        assert FlextTypeGuards.is_instance_of(math.pi, float)
        assert FlextTypeGuards.is_instance_of([1, 2, 3], list)
        assert FlextTypeGuards.is_instance_of({"key": "value"}, dict)

        # Negative cases
        assert not FlextTypeGuards.is_instance_of("test", int)
        assert not FlextTypeGuards.is_instance_of(42, str)
        assert not FlextTypeGuards.is_instance_of([1, 2, 3], dict)

        # Custom classes
        class CustomClass:
            pass

        obj = CustomClass()
        assert FlextTypeGuards.is_instance_of(obj, CustomClass)
        assert not FlextTypeGuards.is_instance_of(obj, str)

        # Inheritance
        class DerivedClass(CustomClass):
            pass

        derived_obj = DerivedClass()
        assert FlextTypeGuards.is_instance_of(derived_obj, DerivedClass)
        assert FlextTypeGuards.is_instance_of(derived_obj, CustomClass)  # inheritance

    def test_is_list_of_validation(self) -> None:
        """Test is_list_of() with homogeneous list validation."""
        # Homogeneous lists
        assert FlextTypeGuards.is_list_of([1, 2, 3, 4], int)
        assert FlextTypeGuards.is_list_of(["a", "b", "c"], str)
        assert FlextTypeGuards.is_list_of([1.1, 2.2, 3.3], float)
        assert FlextTypeGuards.is_list_of([True, False, True], bool)

        # Empty list (should be valid for any type)
        assert FlextTypeGuards.is_list_of([], int)
        assert FlextTypeGuards.is_list_of([], str)

        # Heterogeneous lists (should fail)
        assert not FlextTypeGuards.is_list_of([1, "a", 3], int)
        assert not FlextTypeGuards.is_list_of(["a", 2, "c"], str)
        assert not FlextTypeGuards.is_list_of([1, 2.5, 3], int)

        # Non-list types
        assert not FlextTypeGuards.is_list_of("not_a_list", str)
        assert not FlextTypeGuards.is_list_of(42, int)
        assert not FlextTypeGuards.is_list_of({"key": "value"}, dict)

        # Complex types
        list_of_dicts = [{"a": 1}, {"b": 2}, {"c": 3}]
        assert FlextTypeGuards.is_list_of(list_of_dicts, dict)

        list_of_lists = [[1, 2], [3, 4], [5, 6]]
        assert FlextTypeGuards.is_list_of(list_of_lists, list)

    def test_is_callable_with_return_validation(self) -> None:
        """Test is_callable_with_return() for callable validation."""

        # Functions
        def test_function() -> str:
            return "test"

        assert FlextTypeGuards.is_callable_with_return(test_function, str)
        assert FlextTypeGuards.is_callable_with_return(
            test_function,
            int,
        )  # return_type is for docs only

        # Lambda functions
        def lambda_func(x: int) -> int:
            return x * 2

        assert FlextTypeGuards.is_callable_with_return(lambda_func, int)

        # Built-in functions
        assert FlextTypeGuards.is_callable_with_return(len, int)
        assert FlextTypeGuards.is_callable_with_return(str, str)

        # Methods
        test_list = [1, 2, 3]
        assert FlextTypeGuards.is_callable_with_return(test_list.append, type(None))

        # Non-callable objects
        assert not FlextTypeGuards.is_callable_with_return("not_callable", str)
        assert not FlextTypeGuards.is_callable_with_return(42, int)
        assert not FlextTypeGuards.is_callable_with_return([1, 2, 3], list)

        # Class constructors
        assert FlextTypeGuards.is_callable_with_return(str, str)
        assert FlextTypeGuards.is_callable_with_return(list, list)


class TestFlextGenerators:
    """Test FlextGenerators ID generation and timestamp functionality."""

    def test_generate_uuid_format(self) -> None:
        """Test generate_uuid() produces valid UUID format."""
        uuid1 = FlextGenerators.generate_uuid()
        uuid2 = FlextGenerators.generate_uuid()

        # UUID format: 8-4-4-4-12 hexadecimal characters
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        assert uuid_pattern.match(uuid1)
        assert uuid_pattern.match(uuid2)
        assert uuid1 != uuid2  # Should be unique

    def test_generate_short_id_variations(self) -> None:
        """Test generate_short_id() with different lengths."""
        # Default length (8)
        short_id_8 = FlextGenerators.generate_short_id()
        assert len(short_id_8) == 8
        assert short_id_8.isalnum()

        # Custom lengths
        short_id_4 = FlextGenerators.generate_short_id(4)
        assert len(short_id_4) == 4
        assert short_id_4.isalnum()

        short_id_16 = FlextGenerators.generate_short_id(16)
        assert len(short_id_16) == 16
        assert short_id_16.isalnum()

        # Uniqueness test
        ids = [FlextGenerators.generate_short_id(8) for _ in range(10)]
        assert len(set(ids)) == 10  # All should be unique

    def test_generate_timestamp_accuracy(self) -> None:
        """Test generate_timestamp() provides accurate timestamps."""
        before = time.time()
        timestamp = FlextGenerators.generate_timestamp()
        after = time.time()

        assert isinstance(timestamp, float)
        assert before <= timestamp <= after

        # Multiple timestamps should be increasing
        timestamp1 = FlextGenerators.generate_timestamp()
        time.sleep(0.001)  # Small delay
        timestamp2 = FlextGenerators.generate_timestamp()
        assert timestamp2 > timestamp1

    def test_generate_correlation_id_format(self) -> None:
        """Test generate_correlation_id() format and uniqueness."""
        corr_id1 = FlextGenerators.generate_correlation_id()
        corr_id2 = FlextGenerators.generate_correlation_id()

        # Format: timestamp-shortid (13+ digits, dash, 6 alphanumeric)
        correlation_pattern = re.compile(r"^\d{13,}-[a-zA-Z0-9]{6}$")

        assert correlation_pattern.match(corr_id1)
        assert correlation_pattern.match(corr_id2)
        assert corr_id1 != corr_id2

        # Timestamp part should be reasonable
        timestamp_part = int(corr_id1.split("-")[0])
        current_time_ms = int(time.time() * 1000)
        assert abs(timestamp_part - current_time_ms) < 1000  # Within 1 second

    def test_generate_prefixed_id_patterns(self) -> None:
        """Test generate_prefixed_id() with various prefixes."""
        # Default length
        user_id = FlextGenerators.generate_prefixed_id("USER")
        assert user_id.startswith("USER_")
        assert len(user_id) == len("USER_") + 8  # prefix + underscore + 8 chars

        # Custom length
        REDACTED_LDAP_BIND_PASSWORD_id = FlextGenerators.generate_prefixed_id("ADMIN", 12)
        assert REDACTED_LDAP_BIND_PASSWORD_id.startswith("ADMIN_")
        assert len(REDACTED_LDAP_BIND_PASSWORD_id) == len("ADMIN_") + 12

        # Various prefixes
        prefixes = ["ORDER", "INVOICE", "CUSTOMER", "PRODUCT"]
        for prefix in prefixes:
            prefixed_id = FlextGenerators.generate_prefixed_id(prefix, 6)
            assert prefixed_id.startswith(f"{prefix}_")
            assert len(prefixed_id.split("_")[1]) == 6

    def test_generate_entity_id_format(self) -> None:
        """Test generate_entity_id() FLEXT-prefixed format."""
        entity_id = FlextGenerators.generate_entity_id()

        assert entity_id.startswith("FLEXT_")
        assert len(entity_id) == len("FLEXT_") + 12

        # Multiple entities should be unique
        entity_ids = [FlextGenerators.generate_entity_id() for _ in range(5)]
        assert len(set(entity_ids)) == 5

    def test_generate_id_alias(self) -> None:
        """Test generate_id() alias functionality."""
        id1 = FlextGenerators.generate_id()
        id2 = FlextGenerators.generate_id()

        # Should be UUID format (same as generate_uuid)
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        assert uuid_pattern.match(id1)
        assert uuid_pattern.match(id2)
        assert id1 != id2

    def test_generate_iso_timestamp_format(self) -> None:
        """Test generate_iso_timestamp() ISO 8601 format."""
        iso_timestamp = FlextGenerators.generate_iso_timestamp()

        # ISO 8601 format validation
        iso_pattern = re.compile(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00$",
        )

        assert iso_pattern.match(iso_timestamp)
        assert "T" in iso_timestamp  # Date/time separator
        assert iso_timestamp.endswith("+00:00")  # UTC timezone

    def test_generate_session_id_format(self) -> None:
        """Test generate_session_id() format."""
        session_id = FlextGenerators.generate_session_id()

        assert session_id.startswith("sess_")
        assert len(session_id) == len("sess_") + 16

        # Session ID part should be alphanumeric
        session_part = session_id.split("_")[1]
        assert session_part.isalnum()
        assert len(session_part) == 16

    def test_generate_hash_id_consistency(self) -> None:
        """Test generate_hash_id() consistency and format."""
        test_data = "test_data_for_hashing"

        hash1 = FlextGenerators.generate_hash_id(test_data)
        hash2 = FlextGenerators.generate_hash_id(test_data)

        # Same input should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 characters of SHA-256
        assert all(c in "0123456789abcdef" for c in hash1)  # Hexadecimal

        # Different input should produce different hash
        different_hash = FlextGenerators.generate_hash_id("different_data")
        assert different_hash != hash1


class TestFlextFormatters:
    """Test FlextFormatters string formatting and sanitization functionality."""

    def test_format_duration_various_scales(self) -> None:
        """Test format_duration() with various time scales."""
        # Milliseconds
        assert FlextFormatters.format_duration(0.001) == "1.00ms"
        assert FlextFormatters.format_duration(0.5) == "500.00ms"

        # Seconds
        assert FlextFormatters.format_duration(1.0) == "1.00s"
        assert FlextFormatters.format_duration(1.5) == "1.50s"
        assert FlextFormatters.format_duration(30.25) == "30.25s"

        # Minutes
        assert FlextFormatters.format_duration(60.0) == "1m 0.00s"
        assert FlextFormatters.format_duration(90.5) == "1m 30.50s"
        assert FlextFormatters.format_duration(125.75) == "2m 5.75s"

        # Hours
        assert FlextFormatters.format_duration(3600.0) == "1h 0m"
        assert FlextFormatters.format_duration(3661.0) == "1h 1m"
        assert FlextFormatters.format_duration(7322.5) == "2h 2m"

    def test_format_size_byte_scales(self) -> None:
        """Test format_size() with various byte scales."""
        # Bytes
        assert FlextFormatters.format_size(0) == "0.00B"
        assert FlextFormatters.format_size(512) == "512.00B"
        assert FlextFormatters.format_size(1000) == "1000.00B"

        # Kilobytes
        assert FlextFormatters.format_size(1024) == "1.00KB"
        assert FlextFormatters.format_size(1536) == "1.50KB"
        assert FlextFormatters.format_size(2048) == "2.00KB"

        # Megabytes
        assert FlextFormatters.format_size(1024 * 1024) == "1.00MB"
        assert FlextFormatters.format_size(int(1.5 * 1024 * 1024)) == "1.50MB"

        # Gigabytes
        assert FlextFormatters.format_size(1024 * 1024 * 1024) == "1.00GB"

        # Terabytes
        assert FlextFormatters.format_size(1024 * 1024 * 1024 * 1024) == "1.00TB"

    def test_sanitize_string_security_patterns(self) -> None:
        """Test sanitize_string() security pattern detection."""
        # Credit card patterns
        assert (
            FlextFormatters.sanitize_string("My card is 1234-5678-9012-3456")
            == "My card is [CARD]"
        )
        assert (
            FlextFormatters.sanitize_string("Card: 1234 5678 9012 3456")
            == "Card: [CARD]"
        )
        assert (
            FlextFormatters.sanitize_string("Payment: 1234567890123456")
            == "Payment: [CARD]"
        )

        # SSN patterns
        assert FlextFormatters.sanitize_string("SSN: 123-45-6789") == "SSN: [SSN]"
        assert FlextFormatters.sanitize_string("Social: 987-65-4321") == "Social: [SSN]"

        # Email patterns
        assert (
            FlextFormatters.sanitize_string("Contact: user@example.com")
            == "Contact: [EMAIL]"
        )
        assert (
            FlextFormatters.sanitize_string("Email: john.doe+test@domain.co.uk")
            == "Email: [EMAIL]"
        )

        # Multiple patterns
        sensitive_text = (
            "User john@example.com has card 1234-5678-9012-3456 and SSN 123-45-6789"
        )
        sanitized = FlextFormatters.sanitize_string(sensitive_text)
        assert "[EMAIL]" in sanitized
        assert "[CARD]" in sanitized
        assert "[SSN]" in sanitized
        assert "john@example.com" not in sanitized

        # Length truncation
        long_text = "a" * 200
        truncated = FlextFormatters.sanitize_string(long_text, max_length=50)
        assert len(truncated) == 50
        assert truncated.endswith("...")

    def test_format_dict_display(self) -> None:
        """Test format_dict() for human-readable display."""
        # Simple dictionary
        simple_dict = {"name": "John", "age": 30, "active": True}
        formatted = FlextFormatters.format_dict(simple_dict)
        assert "name='John'" in formatted
        assert "age=30" in formatted
        assert "active=True" in formatted

        # Dictionary with long strings
        long_dict = {"description": "a" * 100, "short": "test"}
        formatted = FlextFormatters.format_dict(long_dict)
        assert "description='" in formatted
        assert "..." in formatted  # Truncated long string
        assert "short='test'" in formatted

        # Mixed types
        mixed_dict = {"string": "test", "number": 42, "list": [1, 2, 3], "none": None}
        formatted = FlextFormatters.format_dict(mixed_dict)
        assert "string='test'" in formatted
        assert "number=42" in formatted
        assert "list=[1, 2, 3]" in formatted
        assert "none=None" in formatted

    def test_truncate_text_handling(self) -> None:
        """Test truncate() text truncation functionality."""
        # Short text (no truncation)
        short_text = "This is short"
        assert FlextFormatters.truncate(short_text, 50) == short_text
        assert FlextFormatters.truncate(short_text) == short_text  # default max_length

        # Exact length
        exact_text = "a" * 50
        assert FlextFormatters.truncate(exact_text, 50) == exact_text

        # Long text (truncation needed)
        long_text = "a" * 200
        truncated = FlextFormatters.truncate(long_text, 50)
        assert len(truncated) == 50
        assert truncated.endswith("...")
        assert truncated[:-3] == "a" * 47  # 47 a's + "..."

        # Custom length
        custom_truncated = FlextFormatters.truncate(long_text, 20)
        assert len(custom_truncated) == 20
        assert custom_truncated.endswith("...")

    def test_snake_to_camel_conversion(self) -> None:
        """Test snake_to_camel() case conversion."""
        # Simple cases
        assert FlextFormatters.snake_to_camel("hello_world") == "helloWorld"
        assert FlextFormatters.snake_to_camel("user_name") == "userName"
        assert FlextFormatters.snake_to_camel("first_last_name") == "firstLastName"

        # Edge cases
        assert FlextFormatters.snake_to_camel("single") == "single"  # No underscores
        assert FlextFormatters.snake_to_camel("") == ""  # Empty string
        assert (
            FlextFormatters.snake_to_camel("_leading") == "Leading"
        )  # Leading underscore
        assert (
            FlextFormatters.snake_to_camel("trailing_") == "trailing"
        )  # Trailing underscore

        # Multiple underscores
        assert FlextFormatters.snake_to_camel("a_b_c_d") == "aBCD"

    def test_camel_to_snake_conversion(self) -> None:
        """Test camel_to_snake() case conversion."""
        # Simple cases
        assert FlextFormatters.camel_to_snake("helloWorld") == "hello_world"
        assert FlextFormatters.camel_to_snake("userName") == "user_name"
        assert FlextFormatters.camel_to_snake("firstName") == "first_name"

        # Edge cases
        assert FlextFormatters.camel_to_snake("single") == "single"  # No capitals
        assert FlextFormatters.camel_to_snake("") == ""  # Empty string
        assert (
            FlextFormatters.camel_to_snake("HTMLParser") == "htmlparser"
        )  # Consecutive caps become lowercase
        assert FlextFormatters.camel_to_snake("XMLHttpRequest") == "xmlhttp_request"

        # Numbers
        assert FlextFormatters.camel_to_snake("user123Name") == "user123_name"

    def test_format_error_message_with_context(self) -> None:
        """Test format_error_message() context integration."""
        # Simple message without context
        simple_msg = FlextFormatters.format_error_message("Operation failed")
        assert simple_msg == "Operation failed"

        # Message with context
        context = {"field": "email", "value": "invalid", "reason": "malformed"}
        formatted = FlextFormatters.format_error_message("Validation failed", context)
        assert "Validation failed" in formatted
        assert "Context:" in formatted
        assert "field=email" in formatted
        assert "value=invalid" in formatted
        assert "reason=malformed" in formatted

        # Empty context
        empty_context_msg = FlextFormatters.format_error_message("Error", {})
        assert empty_context_msg == "Error"

        # None context
        none_context_msg = FlextFormatters.format_error_message("Error", None)
        assert none_context_msg == "Error"

    def test_format_entity_reference(self) -> None:
        """Test format_entity_reference() string formatting."""
        # Simple entity reference
        ref = FlextFormatters.format_entity_reference("User", "123")
        assert ref == "User(123)"

        # Various entity types
        assert (
            FlextFormatters.format_entity_reference("Order", "ORD-456")
            == "Order(ORD-456)"
        )
        assert (
            FlextFormatters.format_entity_reference("Product", "PROD_789")
            == "Product(PROD_789)"
        )
        assert (
            FlextFormatters.format_entity_reference("Customer", "uuid-123-456")
            == "Customer(uuid-123-456)"
        )


class TestFlextUtilitiesOrchestration:
    """Test FlextUtilities consolidated functionality and orchestration patterns."""

    def test_safe_call_error_handling(self) -> None:
        """Test safe_call() FlextResult error handling."""
        # Successful operation
        success_result = FlextUtilities.safe_call(lambda: 42 * 2)
        assert success_result.is_success
        assert success_result.data == 84

        # ValueError
        value_error_result = FlextUtilities.safe_call(lambda: int("invalid"))
        assert value_error_result.is_failure
        assert "invalid literal" in value_error_result.error.lower()

        # TypeError
        type_error_result = FlextUtilities.safe_call(lambda: "test" + 42)
        assert type_error_result.is_failure
        assert (
            "unsupported operand" in type_error_result.error.lower()
            or "can only concatenate" in type_error_result.error.lower()
        )

        # AttributeError
        attr_error_result = FlextUtilities.safe_call(lambda: None.invalid_method)
        assert attr_error_result.is_failure
        assert (
            "attributeerror" in attr_error_result.error.lower()
            or "none" in attr_error_result.error.lower()
        )

        # RuntimeError
        runtime_error_result = FlextUtilities.safe_call(
            lambda: (_ for _ in ()).throw(RuntimeError("test error")),
        )
        assert runtime_error_result.is_failure
        assert "test error" in runtime_error_result.error

    def test_is_not_none_guard_type_safety(self) -> None:
        """Test is_not_none_guard() TypeGuard functionality."""
        # None value
        none_value = None
        assert not FlextUtilities.is_not_none_guard(none_value)

        # Non-None values
        assert FlextUtilities.is_not_none_guard("test")
        assert FlextUtilities.is_not_none_guard(42)
        assert FlextUtilities.is_not_none_guard([])
        assert FlextUtilities.is_not_none_guard({})
        # False is not None
        false_value = False
        assert FlextUtilities.is_not_none_guard(false_value)
        assert FlextUtilities.is_not_none_guard(0)  # 0 is not None

    def test_safe_parse_int_validation(self) -> None:
        """Test safe_parse_int() with various inputs."""
        # Valid integers
        assert FlextUtilities.safe_parse_int("42").unwrap() == 42
        assert FlextUtilities.safe_parse_int("-123").unwrap() == -123
        assert FlextUtilities.safe_parse_int("0").unwrap() == 0

        # Invalid inputs
        invalid_result = FlextUtilities.safe_parse_int("invalid")
        assert invalid_result.is_failure
        assert "cannot parse" in invalid_result.error.lower()

        float_result = FlextUtilities.safe_parse_int("3.14")
        assert float_result.is_failure

        empty_result = FlextUtilities.safe_parse_int("")
        assert empty_result.is_failure

    def test_safe_parse_float_validation(self) -> None:
        """Test safe_parse_float() with various inputs."""
        # Valid floats
        assert FlextUtilities.safe_parse_float("3.14").unwrap() == 3.14
        assert FlextUtilities.safe_parse_float("-2.5").unwrap() == -2.5
        assert FlextUtilities.safe_parse_float("42").unwrap() == 42.0
        assert FlextUtilities.safe_parse_float("0.0").unwrap() == 0.0

        # Invalid inputs
        invalid_result = FlextUtilities.safe_parse_float("invalid")
        assert invalid_result.is_failure
        assert "cannot parse" in invalid_result.error.lower()

        empty_result = FlextUtilities.safe_parse_float("")
        assert empty_result.is_failure

    def test_validate_entity_complete_orchestration(self) -> None:
        """Test validate_entity_complete() orchestration with inherited methods."""
        # Valid entity
        valid_entity_data = {"name": "John", "version": 1, "active": True}
        result = FlextUtilities.validate_entity_complete("user_123", valid_entity_data)

        assert result.is_success
        validated_data = result.data
        assert validated_data["id"] == "user_123"
        assert validated_data["name"] == "John"
        assert validated_data["version"] == 1
        assert "_validated_at" in validated_data
        assert "_validation_id" in validated_data

        # Invalid entity ID
        empty_id_result = FlextUtilities.validate_entity_complete("", valid_entity_data)
        assert empty_id_result.is_failure
        assert "entity id cannot be empty" in empty_id_result.error.lower()

        # Invalid entity data type
        invalid_data_result = FlextUtilities.validate_entity_complete(
            "user_123",
            "not_a_dict",
        )
        assert invalid_data_result.is_failure
        assert "must be a dictionary" in invalid_data_result.error.lower()

        # Missing version
        no_version_data = {"name": "John", "active": True}
        no_version_result = FlextUtilities.validate_entity_complete(
            "user_123",
            no_version_data,
        )
        assert no_version_result.is_failure
        assert "version" in no_version_result.error.lower()

        # Invalid version
        invalid_version_data = {"name": "John", "version": 0, "active": True}
        invalid_version_result = FlextUtilities.validate_entity_complete(
            "user_123",
            invalid_version_data,
        )
        assert invalid_version_result.is_failure
        assert "version must be integer >= 1" in invalid_version_result.error.lower()

        # Skip version requirement
        no_version_required = FlextUtilities.validate_entity_complete(
            "user_123",
            no_version_data,
            require_version=False,
        )
        assert no_version_required.is_success

    def test_generate_entity_metadata_complete_inheritance(self) -> None:
        """Test generate_entity_metadata_complete() using inherited generators."""
        # Basic metadata
        metadata = FlextUtilities.generate_entity_metadata_complete("User")

        assert metadata["type"] == "User"
        assert metadata["version"] == 1
        assert isinstance(metadata["id"], str)
        assert metadata["id"].startswith("FLEXT_")
        assert isinstance(metadata["created_at"], (int, float))
        assert isinstance(metadata["timestamp_iso"], str)
        assert "T" in metadata["timestamp_iso"]  # ISO format

        # Without correlation
        no_corr_metadata = FlextUtilities.generate_entity_metadata_complete(
            "Product",
            include_correlation=False,
        )
        assert "correlation_id" not in no_corr_metadata
        assert "session_id" not in no_corr_metadata

        # With correlation
        with_corr_metadata = FlextUtilities.generate_entity_metadata_complete(
            "Order",
            include_correlation=True,
        )
        assert "correlation_id" in with_corr_metadata
        assert "session_id" in with_corr_metadata
        assert with_corr_metadata["session_id"].startswith("sess_")
        assert "formatted_reference" in with_corr_metadata
        assert "Order(" in with_corr_metadata["formatted_reference"]

    def test_get_system_info_complete_inheritance(self) -> None:
        """Test get_system_info_complete() system information collection."""
        system_info = FlextUtilities.get_system_info_complete()

        # Required fields
        assert "python_version" in system_info
        assert "platform" in system_info
        assert "architecture" in system_info
        assert "processor" in system_info
        assert "flext_version" in system_info
        assert "timestamp" in system_info
        assert "correlation_id" in system_info

        # Value validation
        assert isinstance(system_info["python_version"], str)
        assert "3." in system_info["python_version"]  # Python 3.x
        assert isinstance(system_info["platform"], str)
        assert isinstance(system_info["timestamp"], (int, float))
        assert isinstance(system_info["correlation_id"], str)

    def test_safe_increment_overflow_protection(self) -> None:
        """Test safe_increment() overflow protection."""
        # Normal increment
        result = FlextUtilities.safe_increment(5)
        assert result.is_success
        assert result.data == 6

        # Custom max value
        custom_max_result = FlextUtilities.safe_increment(10, max_value=15)
        assert custom_max_result.is_success
        assert custom_max_result.data == 11

        # Overflow protection
        max_int = 2**31 - 1
        overflow_result = FlextUtilities.safe_increment(max_int)
        assert overflow_result.is_failure
        assert "overflow" in overflow_result.error.lower()

        # Custom overflow - should succeed since 99 + 1 = 100 <= max_value
        custom_result = FlextUtilities.safe_increment(99, max_value=100)
        assert custom_result.is_success
        assert custom_result.data == 100

        # Actual overflow at max_value
        actual_overflow = FlextUtilities.safe_increment(100, max_value=100)
        assert actual_overflow.is_failure
        assert "overflow" in actual_overflow.error.lower()

    def test_safe_get_attr_inherited_methods(self) -> None:
        """Test safe_get_attr() using inherited type checking."""

        # Object with attributes
        class TestObject:
            def __init__(self) -> None:
                self.test_attr = "test_value"
                self.number = 42

        obj = TestObject()

        # Successful attribute access
        attr_result = FlextUtilities.safe_get_attr(obj, "test_attr")
        assert attr_result.is_success
        assert attr_result.data == "test_value"

        number_result = FlextUtilities.safe_get_attr(obj, "number")
        assert number_result.is_success
        assert number_result.data == 42

        # Missing attribute
        missing_result = FlextUtilities.safe_get_attr(obj, "nonexistent")
        assert missing_result.is_failure
        assert "has no attribute" in missing_result.error.lower()

    def test_format_entity_complete_validation(self) -> None:
        """Test format_entity_complete() with validation + formatting."""
        # Valid entity
        formatted = FlextUtilities.format_entity_complete("User", "123", 1)
        assert formatted == "User(id=123, version=1)"

        # Invalid entity type (empty)
        invalid_type = FlextUtilities.format_entity_complete("", "123", 1)
        assert invalid_type == "INVALID_ENTITY"

        # Invalid entity ID (empty)
        invalid_id = FlextUtilities.format_entity_complete("User", "", 1)
        assert invalid_id == "INVALID_ENTITY"

        # Valid complex formatting
        complex_formatted = FlextUtilities.format_entity_complete("Order", "ORD-456", 2)
        assert complex_formatted == "Order(id=ORD-456, version=2)"


class TestPerformanceTracking:
    """Test performance tracking functionality."""

    def setup_method(self) -> None:
        """Clear metrics before each test."""
        clear_performance_metrics()

    def test_record_performance_metrics_storage(self) -> None:
        """Test record_performance() metrics storage."""
        # Record successful operation
        record_performance("database", "get_user", 0.1, success=True)

        metrics = get_performance_metrics()
        assert "database" in metrics

        db_metrics = metrics["database"]
        assert db_metrics["total_calls"] == 1
        assert db_metrics["total_time"] == 0.1
        assert db_metrics["successful_calls"] == 1
        assert db_metrics["failed_calls"] == 0

        # Check function-level metrics
        functions = db_metrics["functions"]
        assert "get_user" in functions

        func_metrics = functions["get_user"]
        assert func_metrics["calls"] == 1
        assert func_metrics["total_time"] == 0.1
        assert func_metrics["avg_time"] == 0.1
        assert func_metrics["min_time"] == 0.1
        assert func_metrics["max_time"] == 0.1

    def test_record_performance_failure_tracking(self) -> None:
        """Test record_performance() failure tracking."""
        record_performance("api", "call_service", 0.5, success=False)

        metrics = get_performance_metrics()
        api_metrics = metrics["api"]

        assert api_metrics["total_calls"] == 1
        assert api_metrics["successful_calls"] == 0
        assert api_metrics["failed_calls"] == 1
        assert api_metrics["total_time"] == 0.5

    def test_track_performance_decorator(self) -> None:
        """Test track_performance() decorator functionality."""

        @track_performance("computation")
        def test_function(x: int) -> int:
            time.sleep(0.001)  # Small delay
            return x * 2

        # Execute function
        result = test_function(5)
        assert result == 10

        # Check metrics
        metrics = get_performance_metrics()
        assert "computation" in metrics

        comp_metrics = metrics["computation"]
        assert comp_metrics["total_calls"] == 1
        assert comp_metrics["successful_calls"] == 1
        assert comp_metrics["failed_calls"] == 0
        assert comp_metrics["total_time"] > 0

        # Check function metrics
        functions = comp_metrics["functions"]
        assert "test_function" in functions

    def test_track_performance_decorator_exception_handling(self) -> None:
        """Test track_performance() decorator with exceptions."""

        @track_performance("validation")
        def failing_function() -> str:
            msg = "Test error"
            raise ValueError(msg)

        # Execute function (should raise)
        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Check metrics recorded failure
        metrics = get_performance_metrics()
        validation_metrics = metrics["validation"]

        assert validation_metrics["total_calls"] == 1
        assert validation_metrics["successful_calls"] == 0
        assert validation_metrics["failed_calls"] == 1

    def test_clear_performance_metrics(self) -> None:
        """Test clear_performance_metrics() functionality."""
        # Add some metrics
        record_performance("test", "function", 0.1, success=True)

        metrics_before = get_performance_metrics()
        assert len(metrics_before) > 0

        # Clear metrics
        clear_performance_metrics()

        metrics_after = get_performance_metrics()
        assert len(metrics_after) == 0


class TestDelegationMixin:
    """Test DelegationMixin automatic delegation patterns."""

    def test_create_delegated_method_functionality(self) -> None:
        """Test create_delegated_method() delegation."""

        def original_method(x: int, y: int) -> int:
            return x + y

        # Create delegated method
        delegated = DelegationMixin.create_delegated_method(
            original_method,
            "add_numbers",
        )

        # Test delegation
        result = delegated(5, 3)
        assert result == 8

    def test_delegate_all_static_methods_batch(self) -> None:
        """Test delegate_all_static_methods() batch delegation."""

        class BaseClass:
            @staticmethod
            def method_one() -> str:
                return "one"

            @staticmethod
            def method_two() -> str:
                return "two"

            def _private_method(self) -> str:
                return "private"

        # Delegate all static methods
        delegated_methods = DelegationMixin.delegate_all_static_methods(BaseClass)

        # Should include public static methods
        assert "method_one" in delegated_methods
        assert "method_two" in delegated_methods

        # Should not include private methods
        assert "_private_method" not in delegated_methods

        # Test delegated functionality
        result_one = delegated_methods["method_one"]()
        assert result_one == "one"

        result_two = delegated_methods["method_two"]()
        assert result_two == "two"


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility function wrappers."""

    def test_safe_call_compatibility(self) -> None:
        """Test safe_call() backward compatibility wrapper."""
        from flext_core.utilities import safe_call

        # Should work like FlextUtilities.safe_call
        result = safe_call(lambda: 42 * 2)
        assert result.is_success
        assert result.data == 84

        # Should handle errors
        error_result = safe_call(lambda: int("invalid"))
        assert error_result.is_failure

    def test_generate_id_compatibility(self) -> None:
        """Test generate_id() backward compatibility wrapper."""
        from flext_core.utilities import generate_id

        # Should work like FlextUtilities.generate_id
        id1 = generate_id()
        id2 = generate_id()

        # UUID format
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        assert uuid_pattern.match(id1)
        assert uuid_pattern.match(id2)
        assert id1 != id2

    def test_generate_correlation_id_compatibility(self) -> None:
        """Test generate_correlation_id() backward compatibility wrapper."""
        from flext_core.utilities import generate_correlation_id

        corr_id = generate_correlation_id()
        correlation_pattern = re.compile(r"^\d{13,}-[a-zA-Z0-9]{6}$")
        assert correlation_pattern.match(corr_id)

    def test_is_not_none_compatibility(self) -> None:
        """Test is_not_none() backward compatibility wrapper."""
        from flext_core.utilities import is_not_none

        assert not is_not_none(None)
        assert is_not_none("test")
        assert is_not_none(42)
        assert is_not_none([])

    def test_truncate_compatibility(self) -> None:
        """Test truncate() backward compatibility wrapper."""
        from flext_core.utilities import truncate

        long_text = "a" * 200
        truncated = truncate(long_text, 50)
        assert len(truncated) == 50
        assert truncated.endswith("...")
