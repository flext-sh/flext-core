"""u comprehensive demonstration using Python 3.13+ strict patterns.

Demonstrates validation, ID generation, type conversion, caching, reliability,
string parsing, collection operations, and type checking using flext-core's
comprehensive utility toolkit with PEP 695 type aliases and collections.abc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
    FlextService,
    t,
    u,
)

# ═══════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════

TEST_DATA: t.Types.ConfigurationMapping = {
    "email": "test@example.com",
    "invalid_email": "invalid-email",
    "number_str": "42",
    "float_str": "3.14",
    "invalid_number": "not-a-number",
    "uri": "https://example.com/api",
    "port": 8080,
    "hostname": "api.example.com",
}


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class UtilitiesService(FlextService[t.Types.ServiceMetadataMapping]):
    """Service demonstrating u comprehensive toolkit."""

    def execute(
        self,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
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

            return FlextResult[t.Types.ServiceMetadataMapping].ok({
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
            })

        except Exception as e:
            error_msg = f"Utilities demonstration failed: {e}"
            return FlextResult[t.Types.ServiceMetadataMapping].fail(error_msg)

    @staticmethod
    def _demonstrate_validation() -> None:
        """Show validation utilities using FlextConstants and u."""
        print("\n=== Validation Utilities ===")

        # Email validation using u
        email = str(TEST_DATA["email"])
        email_result = u.Validation.validate_pattern(
            email,
            FlextConstants.Platform.PATTERN_EMAIL,
            "email",
        )
        print(f"✅ Email validation: {email} -> {email_result.is_success}")

        # String validation using FlextConstants limits
        name = "test"
        name_result = u.Validation.validate_length(
            name,
            min_length=FlextConstants.Validation.MIN_USERNAME_LENGTH,
            max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
        )
        print(f"✅ String validation: {name} -> {name_result.is_success}")

        # URI validation
        uri = str(TEST_DATA["uri"])
        uri_result = u.Validation.Network.validate_uri(uri)
        print(f"✅ URI validation: {uri} -> {uri_result.is_success}")

        # Port validation
        port_value = TEST_DATA["port"]
        if isinstance(port_value, int):
            port_result = u.Validation.Network.validate_port_number(port_value)
            print(f"✅ Port validation: {port_value} -> {port_result.is_success}")

        # Hostname validation
        hostname = str(TEST_DATA["hostname"])
        hostname_result = u.Validation.Network.validate_hostname(hostname)
        print(f"✅ Hostname validation: {hostname} -> {hostname_result.is_success}")

    @staticmethod
    def _demonstrate_id_generation() -> None:
        """Show ID generation utilities using u."""
        print("\n=== ID Generation ===")

        # Correlation ID using u
        correlation_id = u.Generators.generate_correlation_id()
        print(
            f"✅ Correlation ID: {correlation_id[: FlextConstants.Utilities.SHORT_UUID_LENGTH]}...",
        )

        # Short ID using u
        short_id = u.Generators.Random.generate_short_id()
        print(
            f"✅ Short ID: {short_id[: FlextConstants.Utilities.SHORT_UUID_LENGTH]}...",
        )

        # Entity ID
        entity_id = u.Generators.generate_entity_id()
        print(f"✅ Entity ID: {entity_id[:16]}...")

        # Batch ID
        batch_id = u.Generators.generate_batch_id(100)
        print(f"✅ Batch ID: {batch_id[:20]}...")

        # Transaction ID
        transaction_id = u.Generators.generate_transaction_id()
        print(f"✅ Transaction ID: {transaction_id[:20]}...")

    @staticmethod
    def _demonstrate_conversions() -> None:
        """Show type conversion utilities."""
        print("\n=== Type Conversions ===")

        # String parsing utilities
        number_str = str(TEST_DATA["number_str"])
        float_str = str(TEST_DATA["float_str"])

        # Parse delimited strings using railway pattern (DRY)
        parser = u.Parser()
        parser.parse_delimited("a,b,c", ",").map(
            lambda parsed: print(f"✅ Delimited parsing: {parsed}"),
        )

        # String to number conversion concepts
        print(f"✅ String to int concept: '{number_str}' → int")
        print(f"✅ String to float concept: '{float_str}' → float")
        print("✅ Safe conversion with error handling available")

    @staticmethod
    def _demonstrate_caching() -> None:
        """Show caching utilities."""
        print("\n=== Caching Utilities ===")

        # Cache key generation using normalization
        test_data_normalized = u.Cache.normalize_component(TEST_DATA)
        print(f"✅ Data normalization: {type(test_data_normalized).__name__}")

        # Sort dictionary keys for consistent cache keys (DRY)
        sorted_data = u.Cache.sort_dict_keys(TEST_DATA)
        if isinstance(sorted_data, Mapping):
            print(f"✅ Sorted keys: {list(sorted_data.keys())}")

        # Clear object cache
        clear_result = u.Cache.clear_object_cache(TEST_DATA)
        print(f"✅ Cache clearing: {clear_result.is_success}")

    @staticmethod
    def _demonstrate_reliability() -> None:
        """Show reliability utilities."""
        print("\n=== Reliability Patterns ===")

        # Retry logic
        def operation() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        retry_result: FlextResult[str] = u.Reliability.retry(
            operation,
            max_attempts=3,
            delay_seconds=0.1,
        )
        print(f"✅ Retry logic: {retry_result.is_success}")

        # Timeout handling
        timeout_result = u.Reliability.with_timeout(
            operation,
            timeout_seconds=1.0,
        )
        print(f"✅ Timeout handling: {timeout_result.is_success}")

        print("✅ Circuit breaker patterns available")
        print("✅ Retry logic available")

    @staticmethod
    def _demonstrate_string_parsing() -> None:
        """Show string parsing utilities."""
        print("\n=== String Parsing ===")

        # Parse delimited strings using railway pattern (DRY)
        parser = u.Parser()
        # Use FlextModels.Collections.ParseOptions instead of private import
        options = FlextModels.Collections.ParseOptions(strip=True, remove_empty=True)
        parser.parse_delimited("a, b, c", ",", options=options).map(
            lambda parsed: print(f"✅ Delimited parsing: {parsed}"),
        )

        # Split with escape handling using railway pattern (DRY)
        parser.split_on_char_with_escape("cn=admin\\,dc=com", ",", "\\").map(
            lambda split: print(f"✅ Escaped split: {split}"),
        )

    @staticmethod
    def _demonstrate_collection_operations() -> None:
        """Show collection operation utilities."""
        print("\n=== Collection Operations ===")

        # Parse sequence of StrEnum values using railway pattern (DRY)
        u.Collection.parse_sequence(
            FlextConstants.Example.UtilityType,
            ["validation", "id_generation"],
        ).map(
            lambda parsed_enums: print(
                f"✅ Enum sequence parsing: {[e.value for e in parsed_enums]}",
            ),
        )

    @staticmethod
    def _demonstrate_type_checking() -> None:
        """Show type checking utilities."""
        print("\n=== Type Checking ===")

        # Compute accepted message types for a handler class
        message_types = u.Checker.compute_accepted_message_types(UtilitiesService)
        print(f"✅ Message types computed: {len(message_types)} types")


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
        "Validation, generation, conversion, caching, reliability, "
        "parsing, collections, type checking",
    )
    print("=" * 60)

    # Demonstrate composition
    demonstrate_utility_composition()

    # Use service pattern
    service = UtilitiesService()
    result = service.execute()

    # Railway pattern for result handling (DRY)
    def handle_success(data: t.Types.ServiceMetadataMapping) -> None:
        """Handle successful result."""
        categories = data.get("utility_categories", 0)
        utilities = data.get("utilities_demonstrated", [])
        utilities_count = len(utilities) if isinstance(utilities, Sequence) else 0
        print(f"\n✅ Demonstrated {categories} utility categories")
        print(f"✅ Covered {utilities_count} utility types")

    def handle_error(error: str) -> FlextResult[None]:
        """Handle error result."""
        print(f"\n❌ Failed: {error}")
        return FlextResult[None].ok(None)

    result.map(handle_success).lash(handle_error)

    print("\n" + "=" * 60)
    print("🎯 Utility Categories: Validation, ID Generation, Conversion")
    print("🎯 Advanced Features: Caching, Reliability, Composition")
    print("🎯 Type Safety: Full generic support with error handling")
    print("🎯 Python 3.13+: PEP 695 type aliases, collections.abc")
    print("=" * 60)


if __name__ == "__main__":
    main()
