#!/usr/bin/env python3
"""FLEXT Validation - Advanced Validation System Example.

Demonstrates comprehensive validation using FlextValidators with enterprise-grade
validation patterns, functional predicates, and structured validation results.

Features demonstrated:
- Comprehensive data validation with FlextValidators
- Functional predicates with composition patterns
- Structured validation results with detailed error reporting
- Enterprise validation workflows with complex business rules
- Performance validation patterns and optimization
- Custom validator creation and composition
- Multi-field validation with dependency checks
"""

from __future__ import annotations

import time
from typing import Any

from flext_core import FlextResult
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextPredicates, FlextValidators

# =============================================================================
# DOMAIN MODELS - Enterprise validation examples
# =============================================================================


class Customer:
    """Customer domain model for validation demonstration."""

    def __init__(
        self,
        customer_id: str,
        name: str,
        email: str,
        age: int,
        phone: str | None = None,
        address: dict[str, str] | None = None,
    ) -> None:
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.age = age
        self.phone = phone
        self.address = address or {}
        self.created_at = FlextUtilities.generate_iso_timestamp()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for validation."""
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "phone": self.phone,
            "address": self.address,
            "created_at": self.created_at,
        }


class Product:
    """Product domain model for validation demonstration."""

    def __init__(
        self,
        product_id: str,
        name: str,
        price: float,
        category: str,
        stock: int,
        tags: list[str] | None = None,
    ) -> None:
        self.product_id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.stock = stock
        self.tags = tags or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for validation."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "price": self.price,
            "category": self.category,
            "stock": self.stock,
            "tags": self.tags,
        }


# =============================================================================
# BASIC VALIDATION PATTERNS - Core validation functionality
# =============================================================================


def demonstrate_basic_validations() -> None:
    """Demonstrate basic validation patterns using FlextValidators."""
    print("\n🔍 Basic Validation Demonstration")
    print("=" * 50)

    # Type validation
    print("📋 Type Validation:")
    test_values = [
        ("string_value", "Hello World"),
        ("integer_value", 42),
        ("float_value", 3.14),
        ("boolean_value", True),
        ("none_value", None),
        ("list_value", [1, 2, 3]),
        ("dict_value", {"key": "value"}),
        ("empty_string", ""),
        ("whitespace_string", "   "),
    ]

    for name, value in test_values:
        is_string = FlextValidators.is_string(value)
        is_non_empty_string = FlextValidators.is_non_empty_string(value)
        is_int = FlextValidators.is_int(value)
        is_not_none = FlextValidators.is_not_none(value)
        is_list = FlextValidators.is_list(value)
        is_dict = FlextValidators.is_dict(value)

        print(f"  🔹 {name} ({type(value).__name__}):")
        print(f"     String: {is_string}, Non-empty: {is_non_empty_string}")
        print(f"     Int: {is_int}, Not None: {is_not_none}")
        print(f"     List: {is_list}, Dict: {is_dict}")
        print()


def demonstrate_format_validations() -> None:
    """Demonstrate format validation patterns."""
    print("\n📧 Format Validation Demonstration")
    print("=" * 50)

    # Email validation
    print("📋 Email Validation:")
    emails = [
        "valid.email@example.com",
        "user+tag@domain.co.uk",
        "test.email.with+symbol@example.com",
        "invalid.email@",
        "invalid@domain",
        "not_an_email",
        "",
        "spaces @invalid.com",
    ]

    for email in emails:
        is_valid = FlextValidators.is_email(email)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  🔹 '{email}' -> {status}")

    # URL validation
    print("\n📋 URL Validation:")
    urls = [
        "https://www.example.com",
        "http://localhost:8080/path",
        "https://api.example.com/v1/users",
        "http://subdomain.example.com:3000",
        "ftp://files.example.com",
        "invalid_url",
        "http://",
        "",
    ]

    for url in urls:
        is_valid = FlextValidators.is_url(url)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  🔹 '{url}' -> {status}")

    # UUID validation
    print("\n📋 UUID Validation:")
    uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        FlextUtilities.generate_uuid(),
        "invalid-uuid-format",
        "550e8400-e29b-41d4-a716",
        "",
        "not-a-uuid-at-all",
    ]

    for uuid in uuids:
        is_valid = FlextValidators.is_uuid(uuid)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  🔹 '{uuid}' -> {status}")


# =============================================================================
# FUNCTIONAL PREDICATES - Functional programming patterns
# =============================================================================


def demonstrate_functional_predicates() -> None:
    """Demonstrate functional predicate patterns."""
    print("\n🔧 Functional Predicates Demonstration")
    print("=" * 50)

    # Basic predicates
    print("📋 Basic Predicates:")
    test_data = [
        ("not_none_test", [None, "value", 0, False, []]),
        ("positive_numbers", [-5, 0, 1, 42, 3.14, -1.5]),
        ("min_length_strings", ["", "hi", "hello", "world", "a"]),
    ]

    for test_name, values in test_data:
        print(f"  🔹 {test_name}:")

        if test_name == "not_none_test":
            predicate = FlextPredicates.not_none()
        elif test_name == "positive_numbers":
            predicate = FlextPredicates.positive_number()
        else:  # min_length_strings
            predicate = FlextPredicates.min_length(3)

        for value in values:
            result = predicate(value)
            status = "✅ Pass" if result else "❌ Fail"
            print(f"     {value} -> {status}")
        print()

    # Email and URL predicates
    print("📋 Format Predicates:")
    email_predicate = FlextPredicates.is_email()
    url_predicate = FlextPredicates.is_url()

    test_emails = ["valid@example.com", "invalid.email", "user@domain.com"]
    test_urls = ["https://example.com", "invalid_url", "http://localhost:8080"]

    print("  🔹 Email predicate:")
    for email in test_emails:
        result = email_predicate(email)
        status = "✅ Valid" if result else "❌ Invalid"
        print(f"     '{email}' -> {status}")

    print("  🔹 URL predicate:")
    for url in test_urls:
        result = url_predicate(url)
        status = "✅ Valid" if result else "❌ Invalid"
        print(f"     '{url}' -> {status}")


def demonstrate_predicate_composition() -> None:
    """Demonstrate predicate composition patterns."""
    print("\n🔗 Predicate Composition Demonstration")
    print("=" * 50)

    # Range predicates
    print("📋 Range Predicates:")
    age_predicate = FlextPredicates.in_range(18, 65)
    price_predicate = FlextPredicates.in_range(0.01, 1000.0)

    ages = [16, 18, 25, 65, 70]
    prices = [0.0, 0.01, 50.0, 999.99, 1000.0, 1001.0]

    print("  🔹 Age validation (18-65):")
    for age in ages:
        result = age_predicate(age)
        status = "✅ Valid" if result else "❌ Invalid"
        print(f"     {age} years -> {status}")

    print("  🔹 Price validation ($0.01-$1000.00):")
    for price in prices:
        result = price_predicate(price)
        status = "✅ Valid" if result else "❌ Invalid"
        print(f"     ${price} -> {status}")

    # String predicates
    print("\n📋 String Predicates:")
    starts_with_flext = FlextPredicates.starts_with("FLEXT_")
    ends_with_com = FlextPredicates.ends_with(".com")
    min_length_5 = FlextPredicates.min_length(5)

    test_strings = [
        "FLEXT_EntityID",
        "USER_123",
        "example.com",
        "test@example.com",
        "short",
        "verylongstring",
    ]

    print("  🔹 Multiple string predicates:")
    for string in test_strings:
        starts_flext = starts_with_flext(string)
        ends_com = ends_with_com(string)
        min_len = min_length_5(string)

        print(f"     '{string}':")
        print(f"       Starts FLEXT_: {starts_flext}")
        print(f"       Ends .com: {ends_com}")
        print(f"       Min length 5: {min_len}")
        print()


# =============================================================================
# ENTERPRISE VALIDATION WORKFLOWS - Complex business rules
# =============================================================================


def validate_customer_complete(customer_data: dict[str, Any]) -> FlextResult[Customer]:
    """Comprehensive customer validation with business rules."""
    print(f"🔍 Validating customer: {customer_data.get('name', 'Unknown')}")

    # Required field validation
    required_fields = ["customer_id", "name", "email", "age"]
    for field in required_fields:
        if field not in customer_data or not customer_data[field]:
            return FlextResult.fail(f"Missing required field: {field}")

    # Extract data
    customer_id = customer_data["customer_id"]
    name = customer_data["name"]
    email = customer_data["email"]
    age = customer_data["age"]
    phone = customer_data.get("phone")
    address = customer_data.get("address", {})

    # Validate customer ID format
    if not FlextValidators.is_non_empty_string(customer_id):
        return FlextResult.fail("Customer ID must be a non-empty string")

    if not customer_id.startswith("CUST_"):
        return FlextResult.fail("Customer ID must start with 'CUST_'")

    # Validate name
    if not FlextValidators.is_non_empty_string(name):
        return FlextResult.fail("Customer name is required")

    if len(name) < 2:
        return FlextResult.fail("Customer name must be at least 2 characters")

    if len(name) > 100:
        return FlextResult.fail("Customer name must be less than 100 characters")

    # Validate email
    if not FlextValidators.is_email(email):
        return FlextResult.fail("Invalid email format")

    # Business rule: No personal email domains
    personal_domains = ["gmail.com", "yahoo.com", "hotmail.com"]
    email_domain = email.split("@")[1] if "@" in email else ""
    if email_domain in personal_domains:
        return FlextResult.fail(f"Personal email domains not allowed: {email_domain}")

    # Validate age
    if not FlextValidators.is_int(age):
        return FlextResult.fail("Age must be an integer")

    if not FlextValidators.is_in_range(age, 18, 120):
        return FlextResult.fail("Age must be between 18 and 120")

    # Validate phone (optional)
    if phone and not FlextValidators.is_non_empty_string(phone):
        return FlextResult.fail("Phone must be a non-empty string if provided")

    # Validate address (optional)
    if address and not FlextValidators.is_dict(address):
        return FlextResult.fail("Address must be a dictionary if provided")

    if address:
        required_address_fields = ["street", "city", "zip_code"]
        for field in required_address_fields:
            if field not in address or not FlextValidators.is_non_empty_string(
                address[field],
            ):
                return FlextResult.fail(f"Address missing required field: {field}")

    # Create validated customer
    customer = Customer(customer_id, name, email, age, phone, address)
    print(f"✅ Customer validation successful: {customer.name}")

    return FlextResult.ok(customer)


def validate_product_complete(product_data: dict[str, Any]) -> FlextResult[Product]:
    """Comprehensive product validation with business rules."""
    print(f"🔍 Validating product: {product_data.get('name', 'Unknown')}")

    # Required field validation
    required_fields = ["product_id", "name", "price", "category", "stock"]
    for field in required_fields:
        if field not in product_data:
            return FlextResult.fail(f"Missing required field: {field}")

    # Extract data
    product_id = product_data["product_id"]
    name = product_data["name"]
    price = product_data["price"]
    category = product_data["category"]
    stock = product_data["stock"]
    tags = product_data.get("tags", [])

    # Validate product ID
    if not FlextValidators.is_non_empty_string(product_id):
        return FlextResult.fail("Product ID must be a non-empty string")

    if not product_id.startswith("PROD_"):
        return FlextResult.fail("Product ID must start with 'PROD_'")

    # Validate name
    if not FlextValidators.is_non_empty_string(name):
        return FlextResult.fail("Product name is required")

    if len(name) < 3:
        return FlextResult.fail("Product name must be at least 3 characters")

    # Validate price
    if not isinstance(price, (int, float)):
        return FlextResult.fail("Price must be a number")

    if price <= 0:
        return FlextResult.fail("Price must be positive")

    if price > 10000:
        return FlextResult.fail("Price must be less than $10,000")

    # Validate category
    valid_categories = ["electronics", "clothing", "books", "home", "sports"]
    if category not in valid_categories:
        return FlextResult.fail(
            f"Invalid category: {category}. Must be one of: {valid_categories}",
        )

    # Validate stock
    if not FlextValidators.is_int(stock):
        return FlextResult.fail("Stock must be an integer")

    if stock < 0:
        return FlextResult.fail("Stock cannot be negative")

    # Validate tags (optional)
    if tags and not FlextValidators.is_list(tags):
        return FlextResult.fail("Tags must be a list if provided")

    if tags:
        for i, tag in enumerate(tags):
            if not FlextValidators.is_non_empty_string(tag):
                return FlextResult.fail(f"Tag {i} must be a non-empty string")

    # Create validated product
    product = Product(product_id, name, price, category, stock, tags)
    print(f"✅ Product validation successful: {product.name}")

    return FlextResult.ok(product)


# =============================================================================
# VALIDATION WORKFLOWS - Enterprise validation scenarios
# =============================================================================


def demonstrate_customer_validation() -> None:
    """Demonstrate comprehensive customer validation."""
    print("\n👤 Customer Validation Demonstration")
    print("=" * 50)

    customer_test_data = [
        {
            "customer_id": "CUST_001",
            "name": "Alice Johnson",
            "email": "alice@company.com",
            "age": 28,
            "phone": "+1-555-0123",
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zip_code": "12345",
            },
        },
        {
            "customer_id": "CUST_002",
            "name": "Bob Smith",
            "email": "bob@business.org",
            "age": 35,
        },
        {
            "customer_id": "INVALID_ID",  # Invalid ID format
            "name": "Carol Davis",
            "email": "carol@example.com",
            "age": 42,
        },
        {
            "customer_id": "CUST_003",
            "name": "Dave Wilson",
            "email": "dave@gmail.com",  # Personal email domain
            "age": 30,
        },
        {
            "customer_id": "CUST_004",
            "name": "Eve",  # Too short name
            "email": "eve@company.com",
            "age": 25,
        },
        {
            "customer_id": "CUST_005",
            "name": "Frank Miller",
            "email": "invalid.email",  # Invalid email
            "age": 40,
        },
    ]

    valid_customers = []
    invalid_customers = []

    for customer_data in customer_test_data:
        result = validate_customer_complete(customer_data)
        if result.is_success:
            valid_customers.append(result.data)
        else:
            invalid_customers.append((customer_data, result.error))
            print(f"❌ Validation failed: {result.error}")

    print("\n📊 Customer Validation Results:")
    print(f"  ✅ Valid customers: {len(valid_customers)}")
    print(f"  ❌ Invalid customers: {len(invalid_customers)}")

    # Show valid customers
    if valid_customers:
        print("\n📋 Valid Customers:")
        for customer in valid_customers:
            print(f"  🔹 {customer.name} ({customer.customer_id})")
            print(f"     Email: {customer.email}, Age: {customer.age}")

    # Show validation errors
    if invalid_customers:
        print("\n📋 Validation Errors:")
        for customer_data, error in invalid_customers:
            name = customer_data.get("name", "Unknown")
            print(f"  🔹 {name}: {error}")


def demonstrate_product_validation() -> None:
    """Demonstrate comprehensive product validation."""
    print("\n🛍️ Product Validation Demonstration")
    print("=" * 50)

    product_test_data = [
        {
            "product_id": "PROD_001",
            "name": "Wireless Headphones",
            "price": 199.99,
            "category": "electronics",
            "stock": 50,
            "tags": ["audio", "wireless", "premium"],
        },
        {
            "product_id": "PROD_002",
            "name": "Cotton T-Shirt",
            "price": 29.99,
            "category": "clothing",
            "stock": 100,
            "tags": ["cotton", "casual"],
        },
        {
            "product_id": "INVALID_PROD",  # Invalid ID format
            "name": "Invalid Product",
            "price": 50.0,
            "category": "electronics",
            "stock": 10,
        },
        {
            "product_id": "PROD_003",
            "name": "XY",  # Too short name
            "price": 15.0,
            "category": "books",
            "stock": 25,
        },
        {
            "product_id": "PROD_004",
            "name": "Expensive Item",
            "price": 15000.0,  # Too expensive
            "category": "electronics",
            "stock": 1,
        },
        {
            "product_id": "PROD_005",
            "name": "Invalid Category Item",
            "price": 99.99,
            "category": "invalid_category",  # Invalid category
            "stock": 20,
        },
    ]

    valid_products = []
    invalid_products = []

    for product_data in product_test_data:
        result = validate_product_complete(product_data)
        if result.is_success:
            valid_products.append(result.data)
        else:
            invalid_products.append((product_data, result.error))
            print(f"❌ Validation failed: {result.error}")

    print("\n📊 Product Validation Results:")
    print(f"  ✅ Valid products: {len(valid_products)}")
    print(f"  ❌ Invalid products: {len(invalid_products)}")

    # Show valid products
    if valid_products:
        print("\n📋 Valid Products:")
        for product in valid_products:
            print(f"  🔹 {product.name} ({product.product_id})")
            print(f"     Price: ${product.price}, Stock: {product.stock}")
            if product.tags:
                print(f"     Tags: {', '.join(product.tags)}")

    # Show validation errors
    if invalid_products:
        print("\n📋 Validation Errors:")
        for product_data, error in invalid_products:
            name = product_data.get("name", "Unknown")
            print(f"  🔹 {name}: {error}")


# =============================================================================
# PERFORMANCE VALIDATION - Validation performance testing
# =============================================================================


def demonstrate_validation_performance() -> None:
    """Demonstrate validation performance characteristics."""
    print("\n⚡ Validation Performance Demonstration")
    print("=" * 50)

    operations = 1000

    # Email validation performance
    print("📋 Email Validation Performance:")
    test_emails = ["test@example.com"] * operations

    start_time = time.time()
    for email in test_emails:
        FlextValidators.is_email(email)
    email_time = time.time() - start_time

    print(
        f"  🔹 {operations} Email validations: {email_time:.4f}s ({operations / email_time:.0f}/s)",
    )

    # String validation performance
    print("\n📋 String Validation Performance:")
    test_strings = ["Hello World"] * operations

    start_time = time.time()
    for string in test_strings:
        FlextValidators.is_non_empty_string(string)
    string_time = time.time() - start_time

    print(
        f"  🔹 {operations} String validations: {email_time:.4f}s ({operations / string_time:.0f}/s)",
    )

    # Predicate performance
    print("\n📋 Predicate Performance:")
    positive_predicate = FlextPredicates.positive_number()
    test_numbers = [42] * operations

    start_time = time.time()
    for number in test_numbers:
        positive_predicate(number)
    predicate_time = time.time() - start_time

    print(
        f"  🔹 {operations} Predicate validations: {predicate_time:.4f}s ({operations / predicate_time:.0f}/s)",
    )

    # Complex validation performance
    print("\n📋 Complex Validation Performance:")
    customer_data = {
        "customer_id": "CUST_001",
        "name": "Performance Test User",
        "email": "perf@company.com",
        "age": 30,
    }

    start_time = time.time()
    for _ in range(100):  # Fewer operations for complex validation
        validate_customer_complete(customer_data)
    complex_time = time.time() - start_time

    print(
        f"  🔹 100 Complex validations: {complex_time:.4f}s ({100 / complex_time:.0f}/s)",
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:
    """Run comprehensive FlextValidation demonstration."""
    print("=" * 80)
    print("🔍 FLEXT VALIDATION - ADVANCED VALIDATION SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Example 1: Basic Validations
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Basic Validation Patterns")
    print("=" * 60)
    demonstrate_basic_validations()

    # Example 2: Format Validations
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Format Validation Patterns")
    print("=" * 60)
    demonstrate_format_validations()

    # Example 3: Functional Predicates
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Functional Predicates")
    print("=" * 60)
    demonstrate_functional_predicates()

    # Example 4: Predicate Composition
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Predicate Composition")
    print("=" * 60)
    demonstrate_predicate_composition()

    # Example 5: Customer Validation Workflow
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Customer Validation Workflow")
    print("=" * 60)
    demonstrate_customer_validation()

    # Example 6: Product Validation Workflow
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: Product Validation Workflow")
    print("=" * 60)
    demonstrate_product_validation()

    # Example 7: Validation Performance
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 7: Validation Performance")
    print("=" * 60)
    demonstrate_validation_performance()

    print("\n" + "=" * 80)
    print("🎉 FLEXT VALIDATION DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
