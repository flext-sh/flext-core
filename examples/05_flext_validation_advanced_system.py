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
- Maximum type safety using flext_core.types
"""

from __future__ import annotations

import math
import time
from decimal import Decimal

# Import shared domain models to reduce duplication
from shared_domain import (
    Money,
    Product as SharedProduct,
    SharedDomainFactory,
    User as SharedUser,
)

from flext_core import (
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextResult,
    FlextTypes,
    TAnyObject,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# =============================================================================
# VALIDATION CONSTANTS - Business rule constraints
# =============================================================================

# Customer name validation constants
MIN_CUSTOMER_NAME_LENGTH = 2  # Minimum characters for customer name
MAX_CUSTOMER_NAME_LENGTH = 100  # Maximum characters for customer name

# Product name validation constants
MIN_PRODUCT_NAME_LENGTH = 3  # Minimum characters for product name

# Price validation constants
MAX_PRODUCT_PRICE = 10000  # Maximum allowed product price ($10,000)

# Age validation constants
MAX_AGE = 150  # Maximum allowed age
MIN_GUARDIAN_AGE = 21  # Minimum age before requiring guardian consent

# =============================================================================
# ENHANCED DOMAIN MODELS - Using shared domain and flext-core patterns
# =============================================================================


class ValidationDemoUser(SharedUser, FlextLoggableMixin):
    """Enhanced user with advanced validation capabilities."""

    def get_validation_rules(self) -> dict[str, list[str]]:
        """Get validation rules for comprehensive validation."""
        return {
            "name": ["required", "string", "min:2", "max:100"],
            "email_address": ["required", "email"],
            "age": ["required", "integer", "min:18", "max:120"],
            "phone": ["optional", "phone"],
            "address": ["optional", "dict"],
        }

    def validate_business_rules(self) -> FlextResult[None]:
        """Enhanced business rule validation."""
        # Use inherited domain validation
        base_validation = self.validate_domain_rules()
        if base_validation.is_failure:
            return base_validation

        # Additional business rules
        if (
            self.age.value < MIN_GUARDIAN_AGE
            and hasattr(self, "requires_guardian_consent")
        ):
            return FlextResult.fail("Users under 21 require guardian consent")

        return FlextResult.ok(None)


class ValidationDemoProduct(SharedProduct, FlextComparableMixin):
    """Enhanced product with comprehensive validation."""

    def get_comparison_key(self) -> object:
        """Get comparison key for products (price)."""
        return self.price.amount

    def get_validation_rules(self) -> dict[str, list[str]]:
        """Get validation rules for product validation."""
        return {
            "name": ["required", "string", "min:3", "max:200"],
            "description": ["required", "string", "min:10"],
            "price": ["required", "money", "positive"],
            "category": ["required", "string", "min:2"],
            "in_stock": ["required", "boolean"],
        }

    def validate_inventory_rules(self) -> FlextResult[None]:
        """Validate inventory-specific business rules."""
        base_validation = self.validate_domain_rules()
        if base_validation.is_failure:
            return base_validation

        # Advanced inventory rules
        high_value_threshold = Money(amount=Decimal(1000), currency="USD")
        if not self.in_stock and self.price.amount > high_value_threshold.amount:
            return FlextResult.fail("High-value items must be in stock")

        return FlextResult.ok(None)


# =============================================================================
# VALIDATION UTILITIES - Helper functions for validation
# =============================================================================


def validate_user_business_rules(user: SharedUser) -> FlextResult[None]:
    """Validate user business rules using shared domain model."""
    # Basic domain validation
    domain_validation = user.validate_domain_rules()
    if domain_validation.is_failure:
        return domain_validation

    # Additional business rules specific to this validation system
    if user.age.value < MIN_GUARDIAN_AGE:
        # Check if this is a restricted activity
        return FlextResult.fail(f"Users under {MIN_GUARDIAN_AGE} have restrictions")

    if user.status.value == "suspended":
        return FlextResult.fail("Suspended users cannot perform operations")

    return FlextResult.ok(None)


# =============================================================================
# DEMONSTRATION FUNCTIONS - Core validation demonstrations
# =============================================================================


def demonstrate_basic_validations() -> None:
    """Demonstrate basic validation patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 1: Basic Validations")
    print("=" * 60)

    # Test data for validation
    test_cases: list[tuple[str, TAnyObject]] = [
        ("Valid String", "Hello World"),
        ("Empty String", ""),
        ("Valid Integer", 42),
        ("Valid Float", math.pi),
        ("Valid List", [1, 2, 3]),
        ("Valid Dict", {"key": "value"}),
        ("None Value", None),
    ]

    for test_name, test_value in test_cases:
        log_message = f"🔍 Testing: {test_name} = {test_value}"
        print(log_message)

        # Basic type validations using FlextTypes.TypeGuards
        is_str = FlextTypes.TypeGuards.is_instance_of(test_value, str)
        is_int = FlextTypes.TypeGuards.is_instance_of(test_value, int)
        is_float = FlextTypes.TypeGuards.is_instance_of(test_value, float)
        is_list = FlextTypes.TypeGuards.is_instance_of(test_value, list)
        is_dict = FlextTypes.TypeGuards.is_instance_of(test_value, dict)

        log_message = f"   📝 String: {is_str}"
        print(log_message)
        log_message = f"   🔢 Integer: {is_int}"
        print(log_message)
        log_message = f"   🔢 Float: {is_float}"
        print(log_message)
        log_message = f"   📋 List: {is_list}"
        print(log_message)
        log_message = f"   📚 Dict: {is_dict}"
        print(log_message)

        # Non-empty string validation
        if is_str:
            non_empty = len(test_value.strip()) > 0
            log_message = f"   📝 Non-empty: {non_empty}"
            print(log_message)

        print()

    print("✅ Basic validations demonstration completed")


def demonstrate_format_validations() -> None:
    """Demonstrate format validation patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 2: Format Validations")
    print("=" * 60)

    # Email validation test cases
    email_cases: list[tuple[str, str]] = [
        ("Valid Email", "user@example.com"),
        ("Valid Complex Email", "user.name+tag@domain.co.uk"),
        ("Invalid Email - No @", "invalid-email"),
        ("Invalid Email - No Domain", "user@"),
        ("Invalid Email - Empty", ""),
    ]

    log_message = "📧 Email Validation:"
    print(log_message)
    for test_name, email in email_cases:
        # Simple email validation
        is_valid = "@" in email and "." in email.split("@")[-1]
        status = "✅ Valid" if is_valid else "❌ Invalid"
        log_message = f"   {test_name}: {email} -> {status}"
        print(log_message)

    print()

    # URL validation test cases
    url_cases: list[tuple[str, str]] = [
        ("Valid HTTPS URL", "https://example.com"),
        ("Valid HTTP URL", "http://localhost:8080"),
        ("Invalid URL", "not-a-url"),
        ("Invalid URL - No Protocol", "example.com"),
        ("Empty URL", ""),
    ]

    log_message = "🌐 URL Validation:"
    print(log_message)
    for test_name, url in url_cases:
        # Simple URL validation
        is_valid = url.startswith(("http://", "https://"))
        status = "✅ Valid" if is_valid else "❌ Invalid"
        log_message = f"   {test_name}: {url} -> {status}"
        print(log_message)

    print()

    # Numeric range validation
    numeric_cases: list[tuple[str, float, float, float]] = [
        ("In Range", 50.0, 0.0, 100.0),
        ("At Min", 0.0, 0.0, 100.0),
        ("At Max", 100.0, 0.0, 100.0),
        ("Below Min", -10.0, 0.0, 100.0),
        ("Above Max", 150.0, 0.0, 100.0),
    ]

    log_message = "🔢 Numeric Range Validation:"
    print(log_message)
    for test_name, value, min_val, max_val in numeric_cases:
        is_valid = min_val <= value <= max_val
        status = "✅ Valid" if is_valid else "❌ Invalid"
        log_message = f"   {test_name}: {value} in [{min_val}, {max_val}] -> {status}"
        print(log_message)

    print("✅ Format validations demonstration completed")


def demonstrate_functional_predicates() -> None:
    """Demonstrate functional predicates using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 3: Functional Predicates")
    print("=" * 60)

    # Test data
    test_values: list[TAnyObject] = [
        "Hello World",
        "",
        "   ",
        42,
        -5,
        0,
        100,
        math.pi,
        [1, 2, 3],
        [],
        {"key": "value"},
        {},
        None,
    ]

    # Define predicates using FlextTypes.TypeGuards
    predicates = [
        ("Is String", lambda x: FlextTypes.TypeGuards.is_instance_of(x, str)),
        ("Is Non-Empty String", lambda x: isinstance(x, str) and len(x.strip()) > 0),
        (
            "Is Positive Integer",
            lambda x: FlextTypes.TypeGuards.is_instance_of(x, int) and x > 0,
        ),
        ("Is Non-Negative", lambda x: isinstance(x, (int, float)) and x >= 0),
        ("Is List", lambda x: FlextTypes.TypeGuards.is_instance_of(x, list)),
        (
            "Is Non-Empty List",
            lambda x: FlextTypes.TypeGuards.is_instance_of(x, list) and len(x) > 0,
        ),
        ("Is Dict", lambda x: FlextTypes.TypeGuards.is_instance_of(x, dict)),
        ("Is Not None", lambda x: x is not None),
    ]

    for pred_name, predicate in predicates:
        log_message = f"🔍 Predicate: {pred_name}"
        print(log_message)

        results: list[tuple[TAnyObject, bool]] = []
        for value in test_values:
            try:
                result = predicate(value)
                results.append((value, result))
            except Exception as e:  # noqa: BLE001
                error_message: TErrorMessage = f"Predicate failed: {e}"
                log_message = f"   ❌ {value} -> {error_message}"
                print(log_message)
                results.append((value, False))

        # Show results
        valid_count = sum(1 for _, result in results if result)
        log_message = f"   ✅ Valid: {valid_count}/{len(results)}"
        print(log_message)

        # Show some examples
        valid_examples = [str(value) for value, result in results if result][:3]
        if valid_examples:
            log_message = f"   📝 Examples: {', '.join(valid_examples)}"
            print(log_message)

        print()

    print("✅ Functional predicates demonstration completed")


def demonstrate_predicate_composition() -> None:
    """Demonstrate predicate composition patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 4: Predicate Composition")
    print("=" * 60)

    # Test data
    test_values: list[TAnyObject] = [
        "valid@email.com",
        "invalid-email",
        "user@domain",
        "test@example.co.uk",
        "",
        "not-an-email",
        42,
        None,
    ]

    # Define individual predicates
    def is_string(value: TAnyObject) -> bool:
        """Check if value is string."""
        return FlextTypes.TypeGuards.is_instance_of(value, str)

    def has_at_symbol(value: TAnyObject) -> bool:
        """Check if string has @ symbol."""
        return isinstance(value, str) and "@" in value

    def has_domain_part(value: TAnyObject) -> bool:
        """Check if email has domain part."""
        return isinstance(value, str) and "." in value.split("@")[-1]

    def is_non_empty(value: TAnyObject) -> bool:
        """Check if value is non-empty."""
        return isinstance(value, str) and len(value.strip()) > 0

    # Compose predicates
    def is_valid_email(value: TAnyObject) -> bool:
        """Compose email validation predicates."""
        return (
            is_string(value)
            and is_non_empty(value)
            and has_at_symbol(value)
            and has_domain_part(value)
        )

    log_message = "📧 Email Validation with Predicate Composition:"
    print(log_message)

    for value in test_values:
        log_message = f"🔍 Testing: {value}"
        print(log_message)

        # Individual predicate results
        str_result = is_string(value)
        non_empty_result = is_non_empty(value)
        at_result = has_at_symbol(value)
        domain_result = has_domain_part(value)

        log_message = f"   📝 Is String: {str_result}"
        print(log_message)
        log_message = f"   📝 Non-Empty: {non_empty_result}"
        print(log_message)
        log_message = f"   📝 Has @: {at_result}"
        print(log_message)
        log_message = f"   📝 Has Domain: {domain_result}"
        print(log_message)

        # Composed result
        final_result = is_valid_email(value)
        status = "✅ Valid Email" if final_result else "❌ Invalid Email"
        log_message = f"   🎯 Final Result: {status}"
        print(log_message)

        print()

    print("✅ Predicate composition demonstration completed")


def validate_customer_complete(
    customer_data: TUserData,
) -> FlextResult[SharedUser]:
    """Validate customer data using shared domain models and utility validation."""
    log_message: TLogMessage = (
        f"🔍 Validating customer: {customer_data.get('name', 'Unknown')}"
    )
    print(log_message)

    # Use SharedDomainFactory for robust validation
    user_result = SharedDomainFactory.create_user(
        name=customer_data.get("name", ""),
        email=customer_data.get("email", ""),
        age=customer_data.get("age", 0),
    )

    if user_result.is_failure:
        return FlextResult.fail(f"User creation failed: {user_result.error}")

    user = user_result.data

    # Apply additional business rule validation using utility function
    try:
        business_validation = validate_user_business_rules(user)
        if business_validation.is_failure:
            return FlextResult.fail(business_validation.error)

        # Use shared user directly instead of creating local demo user
        enhanced_user = user

        # Log successful validation (shared domain user doesn't have logger)
        log_message = f"Enhanced customer validation successful: {user.name}"
        print(log_message)

        log_message = (
            f"✅ Enhanced customer validation successful: {enhanced_user.name}"
        )
        print(log_message)
        return FlextResult.ok(user)

    except Exception as e:  # noqa: BLE001
        error_message = f"Failed to validate shared customer: {e}"
        return FlextResult.fail(error_message)


def validate_product_complete(  # noqa: PLR0911, PLR0912
    product_data: TAnyObject,
) -> FlextResult[SharedProduct]:
    """Validate product data completely using flext_core.types."""
    log_message: TLogMessage = (
        f"🔍 Validating product: {product_data.get('name', 'Unknown')}"
    )
    print(log_message)

    # Extract fields with type checking
    product_id = product_data.get("product_id")
    name = product_data.get("name")
    price = product_data.get("price")
    category = product_data.get("category")
    stock = product_data.get("stock")
    tags = product_data.get("tags")

    # Validate product_id
    if not FlextTypes.TypeGuards.is_instance_of(product_id, str):
        error_message: TErrorMessage = "Product ID must be a string"
        return FlextResult.fail(error_message)

    if not product_id or len(product_id.strip()) == 0:
        error_message = "Product ID cannot be empty"
        return FlextResult.fail(error_message)

    # Validate name
    if not FlextTypes.TypeGuards.is_instance_of(name, str):
        error_message = "Name must be a string"
        return FlextResult.fail(error_message)

    if not name or len(name.strip()) < MIN_PRODUCT_NAME_LENGTH:
        error_message = f"Name must be at least {MIN_PRODUCT_NAME_LENGTH} characters"
        return FlextResult.fail(error_message)

    # Validate price
    if not FlextTypes.TypeGuards.is_instance_of(price, (int, float)):
        error_message = "Price must be a number"
        return FlextResult.fail(error_message)

    if price < 0:
        error_message = "Price cannot be negative"
        return FlextResult.fail(error_message)

    if price > MAX_PRODUCT_PRICE:
        error_message = f"Price cannot exceed ${MAX_PRODUCT_PRICE}"
        return FlextResult.fail(error_message)

    # Validate category
    if not FlextTypes.TypeGuards.is_instance_of(category, str):
        error_message = "Category must be a string"
        return FlextResult.fail(error_message)

    if not category or len(category.strip()) == 0:
        error_message = "Category cannot be empty"
        return FlextResult.fail(error_message)

    # Validate stock
    if not FlextTypes.TypeGuards.is_instance_of(stock, int):
        error_message = "Stock must be an integer"
        return FlextResult.fail(error_message)

    if stock < 0:
        error_message = "Stock cannot be negative"
        return FlextResult.fail(error_message)

    # Validate tags (optional)
    if tags is not None and not FlextTypes.TypeGuards.is_instance_of(tags, list):
        error_message = "Tags must be a list if provided"
        return FlextResult.fail(error_message)

    if tags is not None:
        for tag in tags:
            if not FlextTypes.TypeGuards.is_instance_of(tag, str):
                error_message = "All tags must be strings"
                return FlextResult.fail(error_message)

    # Create product using SharedDomainFactory
    try:
        product_result = SharedDomainFactory.create_product(
            name=name,
            description=f"Product {name} in {category} category",
            price_amount=str(price),
            currency="USD",
            category=category,
            in_stock=stock > 0,
            id=product_id,
        )

        if product_result.is_failure:
            return FlextResult.fail(f"Product creation failed: {product_result.error}")

        shared_product = product_result.data

        # Create enhanced validation demo product
        enhanced_product = ValidationDemoProduct(
            id=shared_product.id,
            name=shared_product.name,
            description=shared_product.description,
            price=shared_product.price,
            category=shared_product.category,
            in_stock=shared_product.in_stock,
            version=shared_product.version,
            created_at=shared_product.created_at,
        )

        # Perform inventory validation
        inventory_validation = enhanced_product.validate_inventory_rules()
        if inventory_validation.is_failure:
            return FlextResult.fail(inventory_validation.error)

        log_message = (
            f"✅ Enhanced product validation successful: {enhanced_product.name}"
        )
        print(log_message)
        return FlextResult.ok(enhanced_product)

    except Exception as e:  # noqa: BLE001
        error_message = f"Failed to create enhanced product: {e}"
        return FlextResult.fail(error_message)


def demonstrate_customer_validation() -> None:
    """Demonstrate customer validation workflows using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 5: Customer Validation")
    print("=" * 60)

    # Test customer data
    test_customers: list[tuple[str, TUserData]] = [
        (
            "Valid Customer",
            {
                "customer_id": "CUST_001",
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "phone": "+1-555-123-4567",
                "address": {"street": "123 Main St", "city": "Anytown"},
            },
        ),
        (
            "Invalid Email",
            {
                "customer_id": "CUST_002",
                "name": "Jane Smith",
                "email": "invalid-email",
                "age": 25,
            },
        ),
        (
            "Invalid Age",
            {
                "customer_id": "CUST_003",
                "name": "Bob Johnson",
                "email": "bob@example.com",
                "age": -5,
            },
        ),
        (
            "Empty Name",
            {
                "customer_id": "CUST_004",
                "name": "",
                "email": "test@example.com",
                "age": 40,
            },
        ),
    ]

    for test_name, customer_data in test_customers:
        log_message = f"👤 Testing: {test_name}"
        print(log_message)

        validation_result = validate_customer_complete(customer_data)
        if validation_result.is_success:
            customer = validation_result.data
            log_message = (
                f"✅ Enhanced customer created: {customer.name} (ID: {customer.id})"
            )
            print(log_message)
        else:
            log_message = f"❌ Validation failed: {validation_result.error}"
            print(log_message)

        print()

    print("✅ Customer validation demonstration completed")


def demonstrate_product_validation() -> None:
    """Demonstrate product validation workflows using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 6: Product Validation")
    print("=" * 60)

    # Test product data
    test_products: list[tuple[str, TAnyObject]] = [
        (
            "Valid Product",
            {
                "product_id": "PROD_001",
                "name": "Laptop Computer",
                "price": 999.99,
                "category": "Electronics",
                "stock": 50,
                "tags": ["computer", "laptop", "electronics"],
            },
        ),
        (
            "Invalid Price",
            {
                "product_id": "PROD_002",
                "name": "Expensive Item",
                "price": 15000.00,  # Exceeds max price
                "category": "Luxury",
                "stock": 5,
            },
        ),
        (
            "Invalid Stock",
            {
                "product_id": "PROD_003",
                "name": "Test Product",
                "price": 29.99,
                "category": "Test",
                "stock": -10,  # Negative stock
            },
        ),
        (
            "Empty Category",
            {
                "product_id": "PROD_004",
                "name": "No Category Product",
                "price": 19.99,
                "category": "",
                "stock": 100,
            },
        ),
    ]

    for test_name, product_data in test_products:
        log_message = f"📦 Testing: {test_name}"
        print(log_message)

        validation_result = validate_product_complete(product_data)
        if validation_result.is_success:
            product = validation_result.data
            log_message = (
                f"✅ Enhanced product created: {product.name} (ID: {product.id})"
            )
            print(log_message)
            log_message = (
                f"   💰 Price: ${product.price.amount} {product.price.currency}, "
                f"📦 In Stock: {product.in_stock}"
            )
            print(log_message)
        else:
            log_message = f"❌ Validation failed: {validation_result.error}"
            print(log_message)

        print()

    print("✅ Product validation demonstration completed")


def demonstrate_validation_performance() -> None:
    """Demonstrate validation performance patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 60
    print(log_message)
    print("📋 EXAMPLE 7: Validation Performance")
    print("=" * 60)

    # Benchmark data
    test_data: list[TAnyObject] = [
        "valid@email.com",
        "invalid-email",
        "user@domain.co.uk",
        "test@example.com",
        "not-an-email",
    ] * 200  # 1000 total tests

    # Benchmark email validation
    log_message = "🏃 Benchmarking Email Validation"
    print(log_message)

    def validate_email_simple(email: TAnyObject) -> bool:
        """Validate email format simply."""
        return isinstance(email, str) and "@" in email and "." in email.split("@")[-1]

    start_time = time.time()
    valid_count = 0
    for email in test_data:
        if validate_email_simple(email):
            valid_count += 1
    end_time = time.time()

    validation_time = end_time - start_time
    log_message = f"✅ Validated {len(test_data)} emails in {validation_time:.4f}s"
    print(log_message)
    log_message = f"   📊 Valid emails: {valid_count}/{len(test_data)}"
    print(log_message)

    # Benchmark type checking
    log_message = "🏃 Benchmarking Type Checking"
    print(log_message)

    start_time = time.time()
    type_results = [
        FlextTypes.TypeGuards.is_instance_of(item, str) for item in test_data
    ]
    end_time = time.time()

    type_check_time = end_time - start_time
    string_count = sum(type_results)
    log_message = f"✅ Type checked {len(test_data)} items in {type_check_time:.4f}s"
    print(log_message)
    log_message = f"   📊 String items: {string_count}/{len(test_data)}"
    print(log_message)

    # Performance summary
    log_message = "\n📊 Performance Summary:"
    print(log_message)
    log_message = f"   Email Validation: {validation_time:.4f}s"
    print(log_message)
    log_message = f"   Type Checking: {type_check_time:.4f}s"
    print(log_message)
    log_message = f"   Total Operations: {len(test_data)}"
    print(log_message)

    print("✅ Validation performance demonstration completed")


def main() -> None:
    """Run comprehensive FlextValidators demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT VALIDATION - ADVANCED VALIDATION SYSTEM DEMONSTRATION")
    print("=" * 80)

    # Run all demonstrations
    demonstrate_basic_validations()
    demonstrate_format_validations()
    demonstrate_functional_predicates()
    demonstrate_predicate_composition()
    demonstrate_customer_validation()
    demonstrate_product_validation()
    demonstrate_validation_performance()

    print("\n" + "=" * 80)
    print("🎉 FLEXT VALIDATION DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
