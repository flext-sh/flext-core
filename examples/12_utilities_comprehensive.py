#!/usr/bin/env python3
"""12 - FlextUtilities: Comprehensive Utility Functions.

This example demonstrates the COMPLETE FlextUtilities API providing
validation, transformation, processing, caching, generation, and conversion
utilities for the FLEXT ecosystem.

Key Concepts Demonstrated:
- Validation: Email, URL, phone, data validation
- Transformation: Data transformation and manipulation
- Processing: Batch processing, retry logic
- Cache: In-memory caching utilities
- Generators: ID and token generation
- TextProcessor: Text manipulation and formatting
- Conversions: Type conversions and parsing
- Reliability: Retry, circuit breaker, fallback patterns
- TypeGuards: Type checking utilities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from typing import cast

from flext_core import (
    FlextConstants,
    FlextDomainService,
    FlextLogger,
    FlextResult,
    FlextUtilities,
)

# ========== UTILITIES SERVICE ==========


class UtilitiesComprehensiveService(FlextDomainService[dict[str, object]]):
    """Service demonstrating ALL FlextUtilities patterns."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._cache: dict[str, object] = {}

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute method required by FlextDomainService."""
        self._logger.info("Executing utilities demo")
        return FlextResult[dict[str, object]].ok({
            "status": "completed",
            "utilities_executed": True,
        })

    # ========== VALIDATION UTILITIES ==========

    def demonstrate_validation(self) -> None:
        """Show validation utilities."""
        print("\n=== Validation Utilities ===")

        # Email validation
        print("\n1. Email Validation:")
        emails = [
            "valid@example.com",
            "user.name+tag@company.co.uk",
            "invalid@",
            "@invalid.com",
            "no-at-sign.com",
        ]

        for email in emails:
            result = FlextUtilities.Validation.validate_email(email)
            status = "✅" if result.is_success else "❌"
            print(
                f"  {status} {email}: {result.unwrap() if result.is_success else result.error}"
            )

        # URL validation
        print("\n2. URL Validation:")
        urls = [
            "https://www.example.com",
            "http://localhost:8080/path",
            "ftp://files.server.com",
            "not-a-url",
            "//missing-protocol.com",
        ]

        for url in urls:
            result = FlextUtilities.Validation.validate_url(url)
            status = "✅" if result.is_success else "❌"
            print(
                f"  {status} {url}: Valid"
                if result.is_success
                else f"  {status} {url}: {result.error}"
            )

        # Host validation
        print("\n3. Host Validation:")
        hosts = [
            "www.example.com",  # Valid domain
            "192.168.1.1",  # Valid IP
            "localhost",  # Valid localhost
            "invalid..host",  # Invalid format
            "",  # Empty host
        ]

        for host in hosts:
            result = FlextUtilities.Validation.validate_host(host)
            status = "✅" if result.is_success else "❌"
            print(
                f"  {status} {host}: {result.unwrap() if result.is_success else result.error}"
            )

        # Data validation
        print("\n4. Data Validation:")

        # Required fields (manual validation since validate_required_fields doesn't exist)
        data = {"name": "John", "age": 30}
        required = ["name", "age", "email"]
        missing_fields = [field for field in required if field not in data]
        if missing_fields:
            required_result = FlextResult[str].fail(
                f"Missing required fields: {missing_fields}"
            )
        else:
            required_result = FlextResult[str].ok("All required fields present")
        print(
            f"  Required fields: {'✅' if required_result.is_success else '❌'} {required_result.error if required_result.is_failure else required_result.unwrap()}"
        )

        # Data types (manual validation since validate_data_types doesn't exist)
        type_rules = {
            "name": str,
            "age": int,
            "active": bool,
        }
        test_data = {"name": "Alice", "age": 25, "active": True}
        type_errors: list[str] = []
        for field, expected_type in type_rules.items():
            if field in test_data:
                field_value = test_data[field]
                # Check specific types individually to avoid union type issues
                if expected_type is str and not isinstance(field_value, str):
                    type_errors.append(f"{field} should be str")
                elif expected_type is int and not isinstance(field_value, int):
                    type_errors.append(f"{field} should be int")
                elif expected_type is bool and not isinstance(field_value, bool):
                    type_errors.append(f"{field} should be bool")
        if type_errors:
            result = FlextResult[str].fail(f"Type errors: {type_errors}")
        else:
            result = FlextResult[str].ok("All data types correct")
        print(
            f"  Data types: {'✅' if result.is_success else '❌'} {result.error if result.is_failure else result.unwrap()}"
        )

        # Range validation (using positive integer validation)
        value = 50
        int_result = FlextUtilities.Validation.validate_positive_integer(value)
        print(
            f"  Positive integer validation: {'✅' if int_result.is_success else '❌'} Value {value}"
        )

    # ========== TRANSFORMATION UTILITIES ==========

    def demonstrate_transformation(self) -> None:
        """Show transformation utilities."""
        print("\n=== Transformation Utilities ===")

        # String normalization
        print("\n1. String Transformations:")
        test_string = "  Hello World  "
        normalized = FlextUtilities.Transformation.normalize_string(test_string)
        if normalized.is_success:
            print(f"  normalize_string: '{test_string}' → '{normalized.unwrap()}'")
        else:
            print(f"  normalize_string failed: {normalized.error}")

        # Filename sanitization
        dirty_filename = "file<>name?.txt"
        sanitized = FlextUtilities.Transformation.sanitize_filename(dirty_filename)
        if sanitized.is_success:
            print(f"  sanitize_filename: '{dirty_filename}' → '{sanitized.unwrap()}'")
        else:
            print(f"  sanitize_filename failed: {sanitized.error}")

        # Comma-separated parsing
        print("\n2. Data Parsing:")
        csv_string = "apple, banana, cherry, date"
        parsed = FlextUtilities.Transformation.parse_comma_separated(csv_string)
        if parsed.is_success:
            print(f"  parse_comma_separated: '{csv_string}' → {parsed.unwrap()}")
        else:
            print(f"  parse_comma_separated failed: {parsed.error}")

        # Error message formatting
        print("\n3. Error Formatting:")
        error_msg = "Invalid input"
        formatted = FlextUtilities.Transformation.format_error_message(
            error_msg, "Validation"
        )
        if formatted.is_success:
            print(f"  format_error_message: '{error_msg}' → '{formatted.unwrap()}'")
        else:
            print(f"  format_error_message failed: {formatted.error}")

    # ========== PROCESSING UTILITIES ==========

    def demonstrate_processing(self) -> None:
        """Show processing utilities."""
        print("\n=== Processing Utilities ===")

        # Retry operation
        print("\n1. Retry Operation:")
        attempt_count = 0

        def flaky_operation() -> FlextResult[str]:
            """Operation that fails first 2 times."""
            nonlocal attempt_count
            attempt_count += 1
            print(f"    Attempt {attempt_count}")
            if attempt_count < 3:
                return FlextResult[str].fail(f"Failed attempt {attempt_count}")
            return FlextResult[str].ok("Success!")

        result = FlextUtilities.Processing.retry_operation(
            flaky_operation, max_retries=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS, delay_seconds=0.1
        )
        print(
            f"  Retry result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

        # Timeout operation
        print("\n2. Timeout Operation:")

        def slow_operation() -> FlextResult[str]:
            """Operation that takes time."""
            time.sleep(0.05)  # Simulate work
            return FlextResult[str].ok("Completed")

        result = FlextUtilities.Processing.timeout_operation(
            slow_operation, timeout_seconds=0.1
        )
        print(
            f"  Timeout result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

        # Circuit breaker
        print("\n3. Circuit Breaker:")

        def failing_operation() -> FlextResult[str]:
            """Operation that always fails."""
            return FlextResult[str].fail("Service unavailable")

        result = FlextUtilities.Processing.circuit_breaker(
            failing_operation, failure_threshold=2, recovery_timeout=1
        )
        print(
            f"  Circuit breaker result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

    # ========== CACHE UTILITIES ==========

    def demonstrate_cache(self) -> None:
        """Show cache utilities."""
        print("\n=== Cache Utilities ===")

        # Object cache management
        print("\n1. Object Cache Management:")

        # Create a test object with cache attributes
        class TestObject:
            def __init__(self) -> None:
                self._cache = {"key1": "value1", "key2": "value2"}
                self._memoized = "cached_result"

            @property
            def cache(self) -> dict[str, str]:
                """Public access to cache for demonstration purposes."""
                return self._cache

        test_obj = TestObject()
        print(
            f"  Object has cache attributes: {FlextUtilities.Cache.has_cache_attributes(test_obj)}"
        )
        print(f"  Cache before clear: {test_obj.cache}")

        # Clear object cache
        clear_result = FlextUtilities.Cache.clear_object_cache(test_obj)
        if clear_result.is_success:
            print(f"  Cache cleared successfully: {test_obj.cache}")
        else:
            print(f"  Cache clear failed: {clear_result.error}")

        # Deep get from nested data (manual implementation)
        print("\n2. Deep Data Access:")
        nested_data = {
            "user": {"profile": {"name": "John", "age": 30}},
            "settings": {"theme": "dark"},
        }

        # Manual deep get implementation
        def deep_get(
            data: dict[str, object],
            path: str,
            *,
            default: str | int | bool | None = None,
        ) -> str | int | bool | dict[str, object] | None:
            keys = path.split(".")
            current: object = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]  # type: ignore[misc]
                else:
                    return default
            # Type narrowing for return
            if isinstance(current, (str, int, bool)) or current is None:
                return current
            if isinstance(current, dict):
                # Ensure the dict has the correct type annotation
                return cast("dict[str, object]", current)
            return default

        name = deep_get(cast("dict[str, object]", nested_data), "user.profile.name")
        print(f"  Deep get 'user.profile.name': {name}")

        missing = deep_get(
            cast("dict[str, object]", nested_data),
            "missing.key",
            default="default_value",
        )
        print(f"  Deep get with default: {missing}")

    # ========== GENERATOR UTILITIES ==========

    def demonstrate_generators(self) -> None:
        """Show generator utilities."""
        print("\n=== Generator Utilities ===")

        # Generate IDs
        print("\n1. ID Generation:")

        # UUID
        uuid_id = FlextUtilities.Generators.generate_id()
        print(f"  UUID: {uuid_id}")

        # Entity ID
        entity_id = FlextUtilities.Generators.generate_entity_id()
        print(f"  Entity ID: {entity_id}")

        # Short ID
        short_id = FlextUtilities.Generators.generate_short_id(8)
        print(f"  Short ID (8): {short_id}")

        # Generate timestamps
        print("\n2. Timestamp Generation:")

        # ISO timestamp
        timestamp = FlextUtilities.Generators.generate_timestamp()
        print(f"  ISO timestamp: {timestamp}")

        # Correlation ID
        correlation_id = FlextUtilities.Generators.generate_correlation_id()
        print(f"  Correlation ID: {correlation_id}")

        # Generate short IDs
        print("\n3. Short ID Generation:")

        short_id_12 = FlextUtilities.Generators.generate_short_id(12)
        print(f"  Short ID (12): {short_id_12}")

        short_id_6 = FlextUtilities.Generators.generate_short_id(6)
        print(f"  Short ID (6): {short_id_6}")

    # ========== TEXT PROCESSOR ==========

    def demonstrate_text_processor(self) -> None:
        """Show text processing utilities."""
        print("\n=== Text Processing ===")

        # Text cleaning
        print("\n1. Text Cleaning:")
        dirty_text = "  Hello   World!   \n\t  This is    a   test.  "
        clean_result = FlextUtilities.TextProcessor.clean_text(dirty_text)
        if clean_result.is_success:
            print(f"  Dirty: '{dirty_text}'")
            print(f"  Clean: '{clean_result.unwrap()}'")
        else:
            print(f"  Text cleaning failed: {clean_result.error}")

        # Truncation
        print("\n2. Text Truncation:")
        long_text = (
            "This is a very long text that needs to be truncated for display purposes"
        )
        truncated_result = FlextUtilities.TextProcessor.truncate_text(
            long_text, max_length=30
        )
        if truncated_result.is_success:
            print(f"  Original: '{long_text}'")
            print(f"  Truncated (30): '{truncated_result.unwrap()}'")
        else:
            print(f"  Text truncation failed: {truncated_result.error}")

        # Safe string
        print("\n3. Safe String:")
        unsafe_text = "  Valid text  "
        safe_text = FlextUtilities.TextProcessor.safe_string(unsafe_text)
        print(f"  Unsafe: '{unsafe_text}'")
        print(f"  Safe: '{safe_text}'")

        # Safe string with None (manual implementation)
        def safe_string_with_none(text: str | None, default: str = "") -> str:
            if text is None:
                return default
            return text.strip()

        safe_none = safe_string_with_none(None, "default")
        print(f"  Safe None: '{safe_none}'")

    # ========== CONVERSION UTILITIES ==========

    def demonstrate_conversions(self) -> None:
        """Show conversion utilities."""
        print("\n=== Conversion Utilities ===")

        # String to bool
        print("\n1. String to Boolean:")
        test_values = ["true", "True", "1", "yes", "false", "0", "no", "maybe"]
        for value in test_values:
            result = FlextUtilities.Conversions.to_bool(value=value)
            if result.is_success:
                print(f"  '{value}' → {result.unwrap()}")
            else:
                print(f"  '{value}' → Error: {result.error}")

        # String to number
        print("\n2. String to Number:")
        number_strings = ["123", "45.67", "-100", "1.23e4", "invalid"]
        for num_str in number_strings:
            int_result = FlextUtilities.Conversions.to_int(num_str)
            if int_result.is_success:
                print(f"  '{num_str}' → {int_result.unwrap()} (int)")
            else:
                print(f"  '{num_str}' → Error: {int_result.error}")

        # Boolean conversion
        print("\n3. Boolean Conversion:")
        bool_values: list[bool | int | str | None] = [
            True,
            False,
            1,
            0,
            "true",
            "false",
            None,
        ]
        for bool_value in bool_values:
            bool_result = FlextUtilities.Conversions.to_bool(value=bool_value)
            if bool_result.is_success:
                print(f"  {bool_value!r} → {bool_result.unwrap()}")
            else:
                print(f"  {bool_value!r} → Error: {bool_result.error}")

    # ========== RELIABILITY UTILITIES ==========

    def demonstrate_reliability(self) -> None:
        """Show reliability utilities."""
        print("\n=== Reliability Utilities ===")

        # Retry with backoff
        print("\n1. Retry with Backoff:")
        attempt_count = 0

        def flaky_operation() -> FlextResult[str]:
            """Operation that fails first 2 times."""
            nonlocal attempt_count
            attempt_count += 1
            print(f"    Attempt {attempt_count}")
            if attempt_count < 3:
                return FlextResult[str].fail(f"Failed attempt {attempt_count}")
            return FlextResult[str].ok("Success!")

        result = FlextUtilities.Reliability.retry_with_backoff(
            flaky_operation, max_retries=FlextConstants.Reliability.MAX_RETRY_ATTEMPTS, backoff_factor=0.1
        )
        print(
            f"  Final result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

        # Timeout operation
        print("\n2. Timeout Operation:")

        def slow_operation() -> FlextResult[str]:
            """Operation that takes time."""
            time.sleep(0.05)  # Simulate work
            return FlextResult[str].ok("Completed")

        result = FlextUtilities.Reliability.with_timeout(
            slow_operation, timeout_seconds=0.1
        )
        print(
            f"  Timeout result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

        # Fallback pattern (manual implementation)
        print("\n3. Fallback Pattern:")

        def primary_operation() -> FlextResult[str]:
            """Primary operation that fails."""
            return FlextResult[str].fail("Primary failed")

        def fallback_operation() -> FlextResult[str]:
            """Fallback operation."""
            return FlextResult[str].ok("Fallback value")

        # Manual fallback implementation
        primary_result = primary_operation()
        result = primary_result if primary_result.is_success else fallback_operation()

        print(f"  Result with fallback: {result.unwrap()}")

    # ========== TYPE GUARDS ==========

    def demonstrate_type_guards(self) -> None:
        """Show type guard utilities."""
        print("\n=== Type Guards ===")

        test_values: list[object] = [
            None,
            "",
            "hello",
            123,
            0,
            [],
            [1, 2, 3],
            {},
            {"key": "value"},
            True,
            False,
        ]

        print("\n1. Type Checking:")
        for value in test_values:
            checks: list[str] = []

            # Manual type checking since the methods don't exist
            if value is None:
                checks.append("None")
            elif isinstance(value, str):
                if FlextUtilities.TypeGuards.is_string_non_empty(value):
                    checks.append("Non-empty String")
                else:
                    checks.append("Empty String")
            elif isinstance(value, int):
                checks.append("Number")
            elif isinstance(value, list):
                # Cast to list[object] to avoid unknown type issues
                list_value = cast("list[object]", value)
                if len(list_value) > 0:
                    checks.append("Non-empty List")
                else:
                    checks.append("Empty List")
            elif isinstance(value, dict):
                # Type check is necessary here for validation
                # Cast to dict[str, object] to avoid unknown type issues
                dict_value = cast("dict[str, object]", value)
                if len(dict_value) > 0:
                    checks.append("Non-empty Dict")
                else:
                    checks.append("Empty Dict")
            elif isinstance(value, bool):
                # Type check is necessary here for validation
                checks.append("Bool")

            checks_str = ", ".join(checks) if checks else "Unknown"
            print(f"  {value!r:20} → {checks_str}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated utility patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Manual validation (DEPRECATED)
        warnings.warn(
            "Manual validation is DEPRECATED! Use FlextUtilities.Validation.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (manual):")
        print("if '@' not in email:")
        print("    raise ValueError('Invalid email')")

        print("\n✅ CORRECT WAY (FlextUtilities):")
        print("result = FlextUtilities.Validation.validate_email(email)")
        print("if result.is_failure:")
        print("    return FlextResult.fail(result.error)")

        # OLD: Manual retry (DEPRECATED)
        warnings.warn(
            "Manual retry loops are DEPRECATED! Use FlextUtilities.Reliability.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (manual retry):")
        print("for i in range(3):")
        print("    try:")
        print("        result = operation()")
        print("        break")
        print("    except: pass")

        print("\n✅ CORRECT WAY (FlextUtilities):")
        print("result = FlextUtilities.Reliability.retry(")
        print("    operation,")
        print("    max_attempts=3")
        print(")")

        # OLD: String manipulation (DEPRECATED)
        warnings.warn(
            "Manual string manipulation is DEPRECATED! Use FlextUtilities.TextProcessor.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (manual):")
        print("text = text.strip().replace('  ', ' ')")

        print("\n✅ CORRECT WAY (FlextUtilities):")
        print("text = FlextUtilities.TextProcessor.clean_text(text)")


def main() -> None:
    """Main entry point demonstrating all FlextUtilities capabilities."""
    service = UtilitiesComprehensiveService()

    print("=" * 60)
    print("FLEXTUTILITIES COMPLETE API DEMONSTRATION")
    print("Comprehensive Utility Functions")
    print("=" * 60)

    # Core utilities
    service.demonstrate_validation()
    service.demonstrate_transformation()

    # Processing utilities
    service.demonstrate_processing()
    service.demonstrate_cache()

    # Generation utilities
    service.demonstrate_generators()
    service.demonstrate_text_processor()

    # Advanced utilities
    service.demonstrate_conversions()
    service.demonstrate_reliability()

    # Type guards
    service.demonstrate_type_guards()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextUtilities methods demonstrated!")
    print("🎯 Next: See 13_exceptions_handling.py for FlextExceptions")
    print("=" * 60)


if __name__ == "__main__":
    main()
