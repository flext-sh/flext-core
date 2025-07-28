#!/usr/bin/env python3
"""FLEXT Utilities - Generation, Formatting, and Validation Example.

Demonstrates comprehensive utility functions using FlextUtilities with enterprise-grade
ID generation, data formatting, validation, and type checking capabilities.

Features demonstrated:
- Entity ID generation with different strategies
- UUID generation for correlation tracking
- Timestamp formatting and ISO string generation
- Data validation and type checking
- Safe operations with error handling
- Hash generation and data integrity
- String manipulation and formatting
- Configuration value parsing
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from flext_core.utilities import FlextUtilities

# =============================================================================
# DOMAIN MODELS - Real-world data structures
# =============================================================================


class User:
    """User domain model for demonstration."""

    def __init__(self, name: str, email: str, age: int) -> None:
        self.id = FlextUtilities.generate_entity_id()
        self.name = name
        self.email = email
        self.age = age
        self.created_at = FlextUtilities.generate_iso_timestamp()
        self.correlation_id = FlextUtilities.generate_correlation_id()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "created_at": self.created_at,
            "correlation_id": self.correlation_id,
        }


class Order:
    """Order domain model for demonstration."""

    def __init__(self, customer_id: str, items: list[dict[str, Any]]) -> None:
        self.id = FlextUtilities.generate_uuid()
        self.order_number = FlextUtilities.generate_prefixed_id("ORDER", 8)
        self.customer_id = customer_id
        self.items = items
        self.total = sum(
            item.get("price", 0) * item.get("quantity", 1) for item in items
        )
        self.created_at = FlextUtilities.generate_iso_timestamp()
        self.status = "pending"
        self.hash = FlextUtilities.generate_hash_id(
            f"{self.id}_{self.customer_id}_{self.total}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "order_number": self.order_number,
            "customer_id": self.customer_id,
            "items": self.items,
            "total": self.total,
            "created_at": self.created_at,
            "status": self.status,
            "hash": self.hash,
        }


# =============================================================================
# DATA GENERATION PATTERNS - ID and timestamp generation
# =============================================================================


def demonstrate_id_generation() -> None:
    """Demonstrate various ID generation strategies."""
    print("\n🔢 ID Generation Demonstration")
    print("=" * 50)

    # Entity IDs
    print("📋 Entity ID Generation:")
    for i in range(5):
        entity_id = FlextUtilities.generate_entity_id()
        print(f"  🔹 Entity ID {i + 1}: {entity_id}")

    # UUIDs
    print("\n📋 UUID Generation:")
    for i in range(3):
        uuid = FlextUtilities.generate_uuid()
        print(f"  🔹 UUID {i + 1}: {uuid}")

    # Correlation IDs
    print("\n📋 Correlation ID Generation:")
    for i in range(3):
        correlation_id = FlextUtilities.generate_correlation_id()
        print(f"  🔹 Correlation {i + 1}: {correlation_id}")

    # Prefixed IDs (simulating order numbers)
    print("\n📋 Order Number Generation:")
    for i in range(3):
        order_number = FlextUtilities.generate_prefixed_id("ORDER", 8)
        print(f"  🔹 Order Number {i + 1}: {order_number}")

    # Session IDs
    print("\n📋 Session ID Generation:")
    for i in range(3):
        session_id = FlextUtilities.generate_session_id()
        print(f"  🔹 Session ID {i + 1}: {session_id}")


def demonstrate_timestamp_generation() -> None:
    """Demonstrate timestamp and date generation."""
    print("\n🕒 Timestamp Generation Demonstration")
    print("=" * 50)

    # ISO timestamps
    print("📋 ISO Timestamp Generation:")
    for i in range(3):
        timestamp = FlextUtilities.generate_iso_timestamp()
        print(f"  🔹 ISO Timestamp {i + 1}: {timestamp}")

    # Unix timestamps
    print("\n📋 Unix Timestamps:")
    for i in range(3):
        timestamp = FlextUtilities.generate_timestamp()
        print(f"  🔹 Unix Timestamp {i + 1}: {timestamp}")

    # Short IDs
    print("\n📋 Short ID Generation:")
    for i in range(3):
        short_id = FlextUtilities.generate_short_id(length=10)
        print(f"  🔹 Short ID {i + 1}: {short_id}")


def demonstrate_hash_generation() -> None:
    """Demonstrate hash generation for data integrity."""
    print("\n🔐 Hash Generation Demonstration")
    print("=" * 50)

    test_data = [
        "Hello, World!",
        "user@example.com",
        '{"id": "123", "name": "John Doe"}',
        "FLEXT_Core_2025",
        "sensitive_data_12345",
    ]

    print("📋 Hash ID Generation:")
    for data in test_data:
        hash_value = FlextUtilities.generate_hash_id(data)
        print(f"  🔹 '{data}' -> {hash_value}")

    print("\n📋 Additional Hash Examples:")
    for data in test_data[:3]:  # Show fewer examples
        hash_value = FlextUtilities.generate_hash_id(data)
        print(f"  🔹 '{data}' -> {hash_value}")


# =============================================================================
# VALIDATION PATTERNS - Type checking and data validation
# =============================================================================


def demonstrate_type_checking() -> None:
    """Demonstrate type checking utilities."""
    print("\n🔍 Type Checking Demonstration")
    print("=" * 50)

    # Test data with various types
    test_values = [
        ("string", "hello"),
        ("integer", 42),
        ("float", math.pi),
        ("boolean", True),
        ("list", [1, 2, 3]),
        ("dict", {"key": "value"}),
        ("none", None),
        ("class", User("John", "john@example.com", 30)),
    ]

    print("📋 Type Checking Results:")
    for name, value in test_values:
        is_string = FlextUtilities.is_string(value)
        is_int = FlextUtilities.is_int(value)
        is_list = FlextUtilities.is_list(value)
        is_dict = FlextUtilities.is_dict(value)
        is_not_none = FlextUtilities.is_not_none(value)

        print(f"  🔹 {name} ({type(value).__name__}):")
        print(
            f"     String: {is_string}, Int: {is_int}, List: {is_list}, Dict: {is_dict}, Not None: {is_not_none}",
        )


def demonstrate_data_validation() -> None:
    """Demonstrate data validation utilities."""
    print("\n✅ Data Validation Demonstration")
    print("=" * 50)

    # Email validation
    print("📋 Email Validation:")
    emails = [
        "valid@example.com",
        "user.name@domain.co.uk",
        "invalid.email",
        "@invalid.com",
        "user@",
        "",
    ]

    for email in emails:
        is_valid = FlextUtilities.is_email(email)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  🔹 '{email}' -> {status}")

    # URL validation
    print("\n📋 URL Validation:")
    urls = [
        "https://www.example.com",
        "http://localhost:8080",
        "ftp://files.example.com",
        "not_a_url",
        "",
        "https://",
    ]

    for url in urls:
        is_valid = FlextUtilities.is_url(url)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  🔹 '{url}' -> {status}")

    # String validation
    print("\n📋 String Validation:")
    test_strings = [
        '{"name": "John", "age": 30}',
        '[1, 2, 3, "four"]',
        '{"valid": true}',
        "valid string",
        "",
        "   ",
    ]

    for test_str in test_strings:
        is_valid = FlextUtilities.is_non_empty_string(test_str)
        status = "✅ Valid" if is_valid else "❌ Empty"
        print(f"  🔹 '{test_str}' -> {status}")


# =============================================================================
# SAFE OPERATIONS - Error-safe function calls
# =============================================================================


def demonstrate_safe_operations() -> None:
    """Demonstrate safe operation utilities."""
    print("\n🛡️ Safe Operations Demonstration")
    print("=" * 50)

    # Safe JSON parsing using safe_call
    print("📋 Safe JSON Operations:")
    json_data = ['{"name": "John"}', '{"age": 30}', "{invalid}", "null", "[]"]

    for json_str in json_data:
        result = FlextUtilities.safe_call(lambda: json.loads(json_str))
        if result.is_success:
            print(f"  ✅ '{json_str}' -> {result.data}")
        else:
            print(f"  ❌ '{json_str}' -> {result.error}")

    # Safe function calls
    print("\n📋 Safe Function Calls:")

    def risky_division(a: float, b: float) -> float:
        """Functions that might raise ZeroDivisionError."""
        return a / b

    def risky_list_access(items: list[Any], index: int) -> Any:
        """Functions that might raise IndexError."""
        return items[index]

    test_cases = [
        (lambda: risky_division(10, 2), "10 / 2"),
        (lambda: risky_division(10, 0), "10 / 0 (will fail)"),
        (lambda: risky_list_access([1, 2, 3], 1), "list[1]"),
        (lambda: risky_list_access([1, 2, 3], 10), "list[10] (will fail)"),
        (lambda: json.loads('{"valid": true}'), "JSON parse valid"),
        (lambda: json.loads("{invalid}"), "JSON parse invalid"),
    ]

    for func, description in test_cases:
        result = FlextUtilities.safe_call(func)
        if result.is_success:
            print(f"  ✅ {description} -> {result.data}")
        else:
            print(f"  ❌ {description} -> {result.error}")


# =============================================================================
# STRING MANIPULATION - Advanced string utilities
# =============================================================================


def demonstrate_string_utilities() -> None:
    """Demonstrate string manipulation utilities."""
    print("\n📝 String Utilities Demonstration")
    print("=" * 50)

    # String validation using available methods
    print("📋 String Validation:")
    strings = ["", "   ", "hello", "123", "hello123", "HELLO_WORLD"]

    for string in strings:
        is_string = FlextUtilities.is_string(string)
        is_non_empty = FlextUtilities.is_non_empty_string(string)

        # Basic string operations
        cleaned = string.strip() if isinstance(string, str) else str(string)

        print(f"  🔹 '{string}':")
        print(f"     Is string: {is_string}, Non-empty: {is_non_empty}")
        print(f"     Cleaned: '{cleaned}', Length: {len(string)}")
        print()


# =============================================================================
# CONFIGURATION UTILITIES - Environment and config parsing
# =============================================================================


def demonstrate_config_utilities() -> None:
    """Demonstrate configuration parsing utilities."""
    print("\n⚙️ Configuration Utilities Demonstration")
    print("=" * 50)

    # Environment variable simulation
    print("📋 Configuration Value Parsing:")
    config_values = {
        "DEBUG": "true",
        "PORT": "8080",
        "TIMEOUT": "30.5",
        "DATABASE_URL": "postgresql://localhost:5432/mydb",
        "FEATURES": "feature1,feature2,feature3",
        "EMPTY_VALUE": "",
        "NULL_VALUE": "null",
    }

    for key, value in config_values.items():
        # Parse using available safe methods
        as_int = FlextUtilities.safe_parse_int(value)
        as_float = FlextUtilities.safe_parse_float(value)

        # Simple bool parsing
        as_bool = FlextUtilities.safe_call(
            lambda: value.lower() in {"true", "1", "yes", "on"}
            if isinstance(value, str)
            else bool(value),
        )

        # Simple list parsing
        as_list = FlextUtilities.safe_call(
            lambda: value.split(",")
            if isinstance(value, str) and "," in value
            else [value],
        )

        print(f"  🔹 {key} = '{value}':")
        print(f"     Bool: {as_bool.data if as_bool.is_success else 'N/A'}")
        print(f"     Int: {as_int.data if as_int.is_success else 'N/A'}")
        print(f"     Float: {as_float.data if as_float.is_success else 'N/A'}")
        print(f"     List: {as_list.data if as_list.is_success else 'N/A'}")
        print()


# =============================================================================
# DATA FORMATTING - Advanced formatting utilities
# =============================================================================


def demonstrate_formatting_utilities() -> None:
    """Demonstrate data formatting utilities."""
    print("\n🎨 Data Formatting Demonstration")
    print("=" * 50)

    # Basic formatting using available methods
    print("📋 Basic Formatting:")
    numbers = [1234.56, 1000000, 0.123456, 42, -123.45]

    for number in numbers:
        # Simple formatting
        currency = f"${number:,.2f}"
        percentage = f"{(number / 100):.2%}"
        decimal = f"{number:.2f}"

        print(f"  🔹 {number}:")
        print(f"     Currency: {currency}")
        print(f"     Percentage: {percentage}")
        print(f"     Decimal: {decimal}")
        print()

    # Size formatting with simple logic
    print("📋 Size Formatting:")
    sizes = [512, 1024, 1048576, 1073741824, 1099511627776]

    for size in sizes:
        # Simple size formatting
        if size < 1024:
            formatted = f"{size} bytes"
        elif size < 1024**2:
            formatted = f"{size / 1024:.1f} KB"
        elif size < 1024**3:
            formatted = f"{size / 1024**2:.1f} MB"
        elif size < 1024**4:
            formatted = f"{size / 1024**3:.1f} GB"
        else:
            formatted = f"{size / 1024**4:.1f} TB"
        print(f"  🔹 {size} bytes -> {formatted}")


# =============================================================================
# ENTERPRISE DATA PROCESSING - Complex real-world scenarios
# =============================================================================


def demonstrate_enterprise_scenarios() -> None:
    """Demonstrate enterprise-grade data processing scenarios."""
    print("\n🏢 Enterprise Scenarios Demonstration")
    print("=" * 50)

    # User management scenario
    print("📋 User Management Scenario:")
    user_data = [
        {"name": "Alice Johnson", "email": "alice@example.com", "age": 28},
        {"name": "Bob Smith", "email": "bob@example.com", "age": 35},
        {"name": "Carol Davis", "email": "carol@invalid", "age": 42},  # Invalid email
    ]

    users = []
    for data in user_data:
        # Validate email before creating user
        if FlextUtilities.is_email(data["email"]):
            user = User(data["name"], data["email"], data["age"])
            users.append(user)
            print(f"  ✅ Created user: {user.name} (ID: {user.id})")
        else:
            print(f"  ❌ Invalid email for user: {data['name']} ({data['email']})")

    # Order processing scenario
    print("\n📋 Order Processing Scenario:")
    orders_data = [
        {
            "customer_id": users[0].id if users else "unknown",
            "items": [
                {
                    "product_id": "prod_1",
                    "name": "Laptop",
                    "price": 999.99,
                    "quantity": 1,
                },
                {
                    "product_id": "prod_2",
                    "name": "Mouse",
                    "price": 29.99,
                    "quantity": 2,
                },
            ],
        },
        {
            "customer_id": users[1].id if len(users) > 1 else "unknown",
            "items": [
                {
                    "product_id": "prod_3",
                    "name": "Keyboard",
                    "price": 79.99,
                    "quantity": 1,
                },
            ],
        },
    ]

    orders = []
    for order_data in orders_data:
        order = Order(order_data["customer_id"], order_data["items"])
        orders.append(order)

        formatted_total = f"${order.total:,.2f}"
        print(f"  📦 Order {order.order_number}:")
        print(f"     Customer: {order.customer_id}")
        print(f"     Items: {len(order.items)}")
        print(f"     Total: {formatted_total}")
        print(f"     Hash: {order.hash}")
        print()

    # Data integrity verification
    print("📋 Data Integrity Verification:")
    for order in orders:
        # Regenerate hash and verify
        expected_hash = FlextUtilities.generate_hash_id(
            f"{order.id}_{order.customer_id}_{order.total}",
        )
        is_valid = order.hash == expected_hash
        status = "✅ Valid" if is_valid else "❌ Corrupted"
        print(f"  🔹 Order {order.order_number}: {status}")

    # Export data scenario
    print("\n📋 Data Export Scenario:")
    export_data = {
        "export_id": FlextUtilities.generate_uuid(),
        "timestamp": FlextUtilities.generate_iso_timestamp(),
        "users": [user.to_dict() for user in users],
        "orders": [order.to_dict() for order in orders],
        "metadata": {
            "user_count": len(users),
            "order_count": len(orders),
            "total_value": sum(order.total for order in orders),
        },
    }

    # Safe JSON export
    json_result = FlextUtilities.safe_call(
        lambda: json.dumps(export_data, indent=2, default=str),
    )

    if json_result.is_success:
        print("  ✅ Data export successful")
        print(f"     Export ID: {export_data['export_id']}")
        print(f"     Users: {export_data['metadata']['user_count']}")
        print(f"     Orders: {export_data['metadata']['order_count']}")
        total_formatted = f"${export_data['metadata']['total_value']:,.2f}"
        print(f"     Total Value: {total_formatted}")

        # Generate checksum for export
        export_hash = FlextUtilities.generate_hash_id(json_result.data)
        print(f"     Export Hash: {export_hash}")
    else:
        print(f"  ❌ Export failed: {json_result.error}")


# =============================================================================
# PERFORMANCE BENCHMARKING - Utility performance testing
# =============================================================================


def demonstrate_performance_benchmarks() -> None:
    """Demonstrate performance characteristics of utilities."""
    print("\n⚡ Performance Benchmarks Demonstration")
    print("=" * 50)

    # ID generation performance
    print("📋 ID Generation Performance:")
    operations = 1000

    start_time = time.time()
    for _ in range(operations):
        FlextUtilities.generate_entity_id()
    entity_time = time.time() - start_time

    start_time = time.time()
    for _ in range(operations):
        FlextUtilities.generate_uuid()
    uuid_time = time.time() - start_time

    print(
        f"  🔹 {operations} Entity IDs: {entity_time:.4f}s ({operations / entity_time:.0f}/s)",
    )
    print(f"  🔹 {operations} UUIDs: {uuid_time:.4f}s ({operations / uuid_time:.0f}/s)")

    # Hash generation performance
    print("\n📋 Hash Generation Performance:")
    test_data = "test_data_for_hashing" * 100  # Larger data

    start_time = time.time()
    for _ in range(operations):
        FlextUtilities.generate_hash_id(test_data)
    hash_time = time.time() - start_time

    start_time = time.time()
    for _ in range(operations):
        FlextUtilities.generate_uuid()
    uuid_hash_time = time.time() - start_time

    print(
        f"  🔹 {operations} Hash IDs: {hash_time:.4f}s ({operations / hash_time:.0f}/s)",
    )
    print(
        f"  🔹 {operations} UUID hashes: {uuid_hash_time:.4f}s ({operations / uuid_hash_time:.0f}/s)",
    )

    # Validation performance
    print("\n📋 Validation Performance:")
    emails = ["test@example.com"] * operations

    start_time = time.time()
    for email in emails:
        FlextUtilities.is_email(email)
    email_time = time.time() - start_time

    print(
        f"  🔹 {operations} Email validations: {email_time:.4f}s ({operations / email_time:.0f}/s)",
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:
    """Run comprehensive FlextUtilities demonstration."""
    print("=" * 80)
    print("🔧 FLEXT UTILITIES - GENERATION, FORMATTING & VALIDATION DEMONSTRATION")
    print("=" * 80)

    # Example 1: ID Generation
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: ID Generation Strategies")
    print("=" * 60)
    demonstrate_id_generation()

    # Example 2: Timestamp Generation
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Timestamp and Date Generation")
    print("=" * 60)
    demonstrate_timestamp_generation()

    # Example 3: Hash Generation
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Hash Generation for Data Integrity")
    print("=" * 60)
    demonstrate_hash_generation()

    # Example 4: Type Checking
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Type Checking and Validation")
    print("=" * 60)
    demonstrate_type_checking()

    # Example 5: Data Validation
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Data Validation Utilities")
    print("=" * 60)
    demonstrate_data_validation()

    # Example 6: Safe Operations
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: Safe Operations and Error Handling")
    print("=" * 60)
    demonstrate_safe_operations()

    # Example 7: String Utilities
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 7: String Manipulation and Cleaning")
    print("=" * 60)
    demonstrate_string_utilities()

    # Example 8: Configuration Utilities
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 8: Configuration and Environment Parsing")
    print("=" * 60)
    demonstrate_config_utilities()

    # Example 9: Data Formatting
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 9: Advanced Data Formatting")
    print("=" * 60)
    demonstrate_formatting_utilities()

    # Example 10: Enterprise Scenarios
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 10: Enterprise Data Processing")
    print("=" * 60)
    demonstrate_enterprise_scenarios()

    # Example 11: Performance Benchmarks
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 11: Performance Benchmarks")
    print("=" * 60)
    demonstrate_performance_benchmarks()

    print("\n" + "=" * 80)
    print("🎉 FLEXT UTILITIES DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
