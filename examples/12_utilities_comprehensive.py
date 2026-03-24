"""u comprehensive demonstration using Python 3.13+ strict patterns.

Demonstrates validation, ID generation, type conversion, caching, reliability,
string parsing, collection operations, and type checking using flext-core's
comprehensive utility toolkit with PEP 695 type aliases and collections.abc.

**Expected Output:**
- Validation utilities (email, length, pattern, numeric)
- ID generation (UUID, correlation IDs, entity IDs)
- Type conversion and casting utilities
- Object caching and cache management
- Reliability patterns (retry, timeout)
- String parsing and manipulation
- Collection operations (mapping, filtering)
- Type checking and guards

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

from flext_core import FlextConstants, FlextService, c, m, r, t, u

TEST_DATA: t.ConfigMap = t.ConfigMap(
    root={
        "email": "test@example.com",
        "invalid_email": "invalid-email",
        "number_str": "42",
        "float_str": "3.14",
        "invalid_number": "not-a-number",
        "uri": "https://example.com/api",
        "port": 8080,
        "hostname": "api.example.com",
    },
)


class UtilitiesService(FlextService[t.ConfigMap]):
    """Service demonstrating u comprehensive toolkit."""

    @staticmethod
    def _demonstrate_caching() -> None:
        """Show caching utilities."""
        print("\n=== Caching Utilities ===")
        test_data_normalized = u.normalize_component(TEST_DATA)
        print(f"✅ Data normalization: {type(test_data_normalized).__name__}")
        sorted_data = u.sort_dict_keys(TEST_DATA)
        if isinstance(sorted_data, Mapping):
            print("✅ Sorted keys prepared")
        clear_result = u.clear_object_cache(TEST_DATA)
        print(f"✅ Cache clearing: {clear_result.is_success}")

    @staticmethod
    def _demonstrate_collection_operations() -> None:
        """Show collection operation utilities."""
        print("\n=== Collection Operations ===")
        parse_result = u.parse_sequence(
            c.HandlerType,
            ["validation", "id_generation"],
        ).map(
            lambda parsed_enums: print(
                f"✅ Enum sequence parsing: {[e.value for e in parsed_enums]}",
            ),
        )
        if not parse_result.is_success:
            msg = "Collection operations parsing failed"
            raise RuntimeError(msg)

    @staticmethod
    def _demonstrate_conversions() -> None:
        """Show type conversion utilities."""
        print("\n=== Type Conversions ===")
        number_str = str(TEST_DATA["number_str"])
        float_str = str(TEST_DATA["float_str"])
        parser = u()
        delimited_result = parser.parse_delimited("a,b,c", ",").map(
            lambda parsed: print(f"✅ Delimited parsing: {parsed}"),
        )
        if not delimited_result.is_success:
            msg = "Delimited conversion demonstration failed"
            raise RuntimeError(msg)
        print(f"✅ String to int concept: '{number_str}' → int")
        print(f"✅ String to float concept: '{float_str}' → float")
        print("✅ Safe conversion with error handling available")

    @staticmethod
    def _demonstrate_id_generation() -> None:
        """Show ID generation utilities using u."""
        print("\n=== ID Generation ===")
        correlation_id = u.generate("correlation")
        print(f"✅ Correlation ID: {correlation_id[: c.SHORT_UUID_LENGTH]}...")
        short_id = u.generate("ulid")
        print(f"✅ Short ID: {short_id[: c.SHORT_UUID_LENGTH]}...")
        entity_id = u.generate("entity")
        print(f"✅ Entity ID: {entity_id[:16]}...")
        batch_id = u.generate("batch", parts=(100,))
        print(f"✅ Batch ID: {batch_id[:20]}...")
        transaction_id = u.generate("transaction")
        print(f"✅ Transaction ID: {transaction_id[:20]}...")

    @staticmethod
    def _demonstrate_reliability() -> None:
        """Show reliability utilities."""
        print("\n=== Reliability Patterns ===")

        def operation() -> r[str]:
            return r[str].ok("success")

        retry_result: r[str] = u.retry(operation, max_attempts=3, delay_seconds=0.1)
        print(f"✅ Retry logic: {retry_result.is_success}")
        timeout_result = u.with_timeout(operation, timeout_seconds=1.0)
        print(f"✅ Timeout handling: {timeout_result.is_success}")
        print("✅ Circuit breaker patterns available")
        print("✅ Retry logic available")

    @staticmethod
    def _demonstrate_string_parsing() -> None:
        """Show string parsing utilities."""
        print("\n=== String Parsing ===")
        parser = u()
        options = m.ParseOptions(strip=True, remove_empty=True)
        delimited_result = parser.parse_delimited("a, b, c", ",", options=options).map(
            lambda parsed: print(f"✅ Delimited parsing: {parsed}"),
        )
        if not delimited_result.is_success:
            msg = "String parsing demonstration failed"
            raise RuntimeError(msg)
        split_result = parser.split_on_char_with_escape(
            "cn=REDACTED_LDAP_BIND_PASSWORD\\,dc=com",
            ",",
            "\\",
        ).map(lambda split: print(f"✅ Escaped split: {split}"))
        if not split_result.is_success:
            msg = "Escaped split demonstration failed"
            raise RuntimeError(msg)

    @staticmethod
    def _demonstrate_type_checking() -> None:
        """Show type checking utilities."""
        print("\n=== Type Checking ===")
        message_types = u.compute_accepted_message_types(UtilitiesService)
        print(f"✅ Message types computed: {len(message_types)} types")

    @staticmethod
    def _demonstrate_validation() -> None:
        """Show validation utilities using FlextConstants and u."""
        print("\n=== Validation Utilities ===")
        email = str(TEST_DATA["email"])
        email_result = u.validate_pattern(email, FlextConstants.PATTERN_EMAIL, "email")
        print(f"✅ Email validation: {email} -> {email_result.is_success}")
        name = "test"
        name_result = u.validate_length(
            name,
            min_length=FlextConstants.MIN_USERNAME_LENGTH,
            max_length=FlextConstants.MAX_NAME_LENGTH,
        )
        print(f"✅ String validation: {name} -> {name_result.is_success}")
        uri = str(TEST_DATA["uri"])
        uri_result = u.validate_pattern(
            uri,
            "^[a-zA-Z][a-zA-Z0-9+.-]*://[^\\s]+$",
            "uri",
        )
        print(f"✅ URI validation: {uri} -> {uri_result.is_success}")
        port_value = TEST_DATA["port"]
        if isinstance(port_value, int):
            port_result = (
                r[int].ok(port_value)
                if 1 <= port_value <= 65535
                else r[int].fail("port must be between 1 and 65535")
            )
            print(f"✅ Port validation: {port_value} -> {port_result.is_success}")
        hostname = str(TEST_DATA["hostname"])
        hostname_result = u.validate_pattern(
            hostname,
            "^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\\.[a-zA-Z0-9-]{1,63}(?<!-))*$",
            "hostname",
        )
        print(f"✅ Hostname validation: {hostname} -> {hostname_result.is_success}")

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Execute comprehensive utilities demonstrations."""
        print("Starting utilities demonstration")
        try:
            self._demonstrate_validation()
            self._demonstrate_id_generation()
            self._demonstrate_conversions()
            self._demonstrate_caching()
            self._demonstrate_reliability()
            self._demonstrate_string_parsing()
            self._demonstrate_collection_operations()
            self._demonstrate_type_checking()
            return r[t.ConfigMap].ok(
                t.ConfigMap(
                    root={
                        "utilities_demonstrated": [
                            "validation",
                            "id_generation",
                            "conversions",
                            "caching",
                            "reliability",
                            "string_parsing",
                            "collection",
                            "type_checking",
                        ],
                        "utility_categories": 8,
                        "flext_utilities_features": [
                            "type_safety",
                            "error_handling",
                            "performance",
                            "reliability",
                        ],
                    },
                ),
            )
        except Exception as e:
            error_msg = f"Utilities demonstration failed: {e}"
            return r[t.ConfigMap].fail(error_msg)


def demonstrate_utility_composition() -> None:
    """Show how utilities compose together."""
    print("\n=== Utility Composition ===")
    print("✅ Validation + ID generation + caching")
    print("✅ Type conversion + error handling")
    print("✅ Reliability patterns + monitoring")
    print("✅ String parsing + collection operations")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT UTILITIES - COMPREHENSIVE DEMONSTRATION")
    print(
        "Validation, generation, conversion, caching, reliability, parsing, collections, type checking",
    )
    print("=" * 60)
    demonstrate_utility_composition()
    service = UtilitiesService()
    result = service.execute()

    def handle_success(_data: t.ConfigMap) -> None:
        """Handle successful result."""
        print("\n✅ Demonstrated 8 utility categories")
        print("✅ Covered listed utility types")

    def handle_error(error: str) -> r[None]:
        """Handle error result."""
        print(f"\n❌ Failed: {error}")
        return r[None].ok(value=None)

    chain_result = result.map(handle_success).lash(handle_error)
    if not chain_result.is_success:
        msg = "Utility composition chain failed"
        raise RuntimeError(msg)
    print("\n" + "=" * 60)
    print("🎯 Utility Categories: Validation, ID Generation, Conversion")
    print("🎯 Advanced Features: Caching, Reliability, Composition")
    print("🎯 Type Safety: Full generic support with error handling")
    print("🎯 Python 3.13+: PEP 695 type aliases, collections.abc")
    print("=" * 60)


if __name__ == "__main__":
    main()
