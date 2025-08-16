#!/usr/bin/env python3
"""Advanced validation system using FlextValidators.

Demonstrates comprehensive data validation with functional predicates,
composition patterns, and structured error reporting.
- Enterprise validation workflows with complex business rules
- Performance validation patterns and optimization
- Custom validator creation and composition
- Multi-field validation with dependency checks
- Maximum type safety using flext_core.typings
"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from flext_core import (
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextResult,
    TAnyObject,
    TErrorMessage,
    TUserData,
)

from .shared_domain import (
    Money,
    Product as SharedProduct,
    SharedDemonstrationPattern,
    SharedDomainFactory,
    User as SharedUser,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
# COMPLEXITY REDUCTION HELPERS - SOLID SRP: Validation pattern refactoring
# =============================================================================


class ValidationResult:
    """Helper to accumulate validation errors - eliminates multiple returns."""

    def __init__(self) -> None:
        """Initialize validation result accumulator."""
        self.errors: list[TErrorMessage] = []
        self.warnings: list[str] = []

    def add_error(self, error: TErrorMessage) -> None:
        """Add validation error to accumulator."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add validation warning to accumulator."""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if any validation errors occurred."""
        return len(self.errors) > 0

    def get_combined_error(self) -> TErrorMessage:
        """Get combined error message from all validation errors."""
        if not self.has_errors():
            return ""
        return f"Validation failed: {'; '.join(self.errors)}"

    def to_flext_result(self) -> FlextResult[None]:
        """Convert validation result to FlextResult - eliminates early returns."""
        if self.has_errors():
            return FlextResult.fail(self.get_combined_error())
        return FlextResult.ok(None)


class ProductFieldValidator:
    """Strategy pattern: Specialized product field validation - reduces complexity."""

    def __init__(self, validation_result: ValidationResult) -> None:
        """Initialize with validation result accumulator."""
        self.validation_result = validation_result

    def validate_product_id(self, product_id: object) -> None:
        """Validate product ID field - SOLID SRP."""
        if not isinstance(product_id, str):
            self.validation_result.add_error("Product ID must be a string")
            return

        # Now product_id is narrowed to str type
        if not product_id or len(product_id.strip()) == 0:
            self.validation_result.add_error("Product ID cannot be empty")

    def validate_name(self, name: object) -> None:
        """Validate product name field - SOLID SRP."""
        if not isinstance(name, str):
            self.validation_result.add_error("Name must be a string")
            return

        # Now name is narrowed to str type
        if not name or len(name.strip()) < MIN_PRODUCT_NAME_LENGTH:
            self.validation_result.add_error(
                f"Name must be at least {MIN_PRODUCT_NAME_LENGTH} characters",
            )

    def validate_price(self, price: object) -> None:
        """Validate product price field - SOLID SRP."""
        if not isinstance(price, (int, float)):
            self.validation_result.add_error("Price must be a number")
            return

        # Now price is narrowed to int | float type
        if price < 0:
            self.validation_result.add_error("Price cannot be negative")

        if price > MAX_PRODUCT_PRICE:
            self.validation_result.add_error(
                f"Price cannot exceed ${MAX_PRODUCT_PRICE}",
            )

    def validate_category(self, category: object) -> None:
        """Validate product category field - SOLID SRP."""
        if not isinstance(category, str):
            self.validation_result.add_error("Category must be a string")
            return

        # Now category is narrowed to str type
        if not category or len(category.strip()) == 0:
            self.validation_result.add_error("Category cannot be empty")

    def validate_stock(self, stock: object) -> None:
        """Validate product stock field - SOLID SRP."""
        if not isinstance(stock, int):
            self.validation_result.add_error("Stock must be an integer")
            return

        # Now stock is narrowed to int type
        if stock < 0:
            self.validation_result.add_error("Stock cannot be negative")

    def validate_tags(self, tags: object) -> None:
        """Validate product tags field - SOLID SRP."""
        if tags is None:
            return  # Tags are optional

        if not isinstance(tags, list):
            self.validation_result.add_error("Tags must be a list if provided")
            return

        # Now tags is narrowed to list type
        for tag in tags:
            if not isinstance(tag, str):
                self.validation_result.add_error("All tags must be strings")
                break


class ProductValidationOrchestrator:
    """Orchestrator for complete product validation - eliminates multiple returns."""

    def __init__(self, product_data: TAnyObject) -> None:
        """Initialize with product data to validate."""
        self.product_data = product_data
        self.validation_result = ValidationResult()
        self.field_validator = ProductFieldValidator(self.validation_result)

    def validate_all_fields(self) -> FlextResult[None]:
        """Validate all product fields using accumulator pattern - single return."""
        # Extract all fields
        product_dict = cast("dict[str, object]", self.product_data)
        fields = {
            "product_id": product_dict.get("product_id"),
            "name": product_dict.get("name"),
            "price": product_dict.get("price"),
            "category": product_dict.get("category"),
            "stock": product_dict.get("stock"),
            "tags": product_dict.get("tags"),
        }

        # Validate each field using strategy pattern
        self.field_validator.validate_product_id(fields["product_id"])
        self.field_validator.validate_name(fields["name"])
        self.field_validator.validate_price(fields["price"])
        self.field_validator.validate_category(fields["category"])
        self.field_validator.validate_stock(fields["stock"])
        self.field_validator.validate_tags(fields["tags"])

        # Single return point - Result pattern
        return self.validation_result.to_flext_result()


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
        if self.age.value < MIN_GUARDIAN_AGE and hasattr(
            self,
            "requires_guardian_consent",
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

    if user.status == "suspended":
        return FlextResult.fail("Suspended users cannot perform operations")

    return FlextResult.ok(None)


# =============================================================================
# DEMONSTRATION FUNCTIONS - Core validation demonstrations
# =============================================================================


def demonstrate_basic_validations() -> None:
    """Demonstrate basic validation patterns using flext_core.typings."""
    "\n" + "=" * 60

    # Test data for validation - use object for flexible test values
    test_cases: list[tuple[str, object]] = [
        ("Valid String", "Hello World"),
        ("Empty String", ""),
        ("Valid Integer", 42),
        ("Valid Float", math.pi),
        ("Valid List", [1, 2, 3]),
        ("Valid Dict", {"key": "value"}),
        ("None Value", None),
    ]

    for _test_name, test_value in test_cases:
        # Basic type validations using built-in isinstance (better type narrowing)
        is_str = isinstance(test_value, str)
        isinstance(test_value, int)
        isinstance(test_value, float)
        isinstance(test_value, list)
        isinstance(test_value, dict)

        # Non-empty string validation (demonstration)
        if is_str and isinstance(test_value, str):
            _ = len(test_value.strip()) > 0


def demonstrate_format_validations() -> None:
    """Demonstrate format validation patterns using flext_core.typings."""
    "\n" + "=" * 60

    # Email validation test cases
    email_cases: list[tuple[str, str]] = [
        ("Valid Email", "user@example.com"),
        ("Valid Complex Email", "user.name+tag@domain.co.uk"),
        ("Invalid Email - No @", "invalid-email"),
        ("Invalid Email - No Domain", "user@"),
        ("Invalid Email - Empty", ""),
    ]

    for _test_name, email in email_cases:
        # Simple email validation
        "@" in email and "." in email.split("@")[-1]

    # URL validation test cases
    url_cases: list[tuple[str, str]] = [
        ("Valid HTTPS URL", "https://example.com"),
        ("Valid HTTP URL", "http://localhost:8080"),
        ("Invalid URL", "not-a-url"),
        ("Invalid URL - No Protocol", "example.com"),
        ("Empty URL", ""),
    ]

    for _test_name, url in url_cases:
        # Simple URL validation
        url.startswith(("http://", "https://"))

    # Numeric range validation
    numeric_cases: list[tuple[str, float, float, float]] = [
        ("In Range", 50.0, 0.0, 100.0),
        ("At Min", 0.0, 0.0, 100.0),
        ("At Max", 100.0, 0.0, 100.0),
        ("Below Min", -10.0, 0.0, 100.0),
        ("Above Max", 150.0, 0.0, 100.0),
    ]

    for _test_name, _value, _min_val, _max_val in numeric_cases:
        pass


def demonstrate_functional_predicates() -> None:
    """Demonstrate functional predicates using flext_core.typings."""
    "\n" + "=" * 60

    # Test data - use object for mixed value types
    test_values: list[object] = [
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

    # Define predicates using built-in isinstance (better type narrowing)
    predicates: list[tuple[str, Callable[[object], bool]]] = [
        ("Is String", lambda x: isinstance(x, str)),
        ("Is Non-Empty String", lambda x: isinstance(x, str) and len(x.strip()) > 0),
        (
            "Is Positive Integer",
            lambda x: isinstance(x, int) and x > 0,
        ),
        ("Is Non-Negative", lambda x: isinstance(x, (int, float)) and x >= 0),
        ("Is List", lambda x: isinstance(x, list)),
        (
            "Is Non-Empty List",
            lambda x: isinstance(x, list) and len(x) > 0,
        ),
        ("Is Dict", lambda x: isinstance(x, dict)),
        ("Is Not None", lambda x: x is not None),
    ]

    for _pred_name, predicate in predicates:
        results: list[tuple[object, bool]] = []
        for value in test_values:
            try:
                result = predicate(value)
                results.append((value, result))
            except (RuntimeError, ValueError, TypeError):
                results.append((value, False))

        # Show results
        valid_count = sum(1 for _, result in results if result)
        f"   ✅ Valid: {valid_count}/{len(results)}"

        # Show some examples
        valid_examples = [str(value) for value, result in results if result][:3]
        if valid_examples:
            f"   📝 Examples: {', '.join(valid_examples)}"


def demonstrate_predicate_composition() -> None:
    """Demonstrate predicate composition patterns using flext_core.typings."""
    "\n" + "=" * 60

    # Test data - use object for mixed value types
    test_values: list[object] = [
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
    def is_string(value: object) -> bool:
        """Check if value is string."""
        return isinstance(value, str)

    def has_at_symbol(value: object) -> bool:
        """Check if string has @ symbol."""
        return isinstance(value, str) and "@" in value

    def has_domain_part(value: object) -> bool:
        """Check if email has domain part."""
        return isinstance(value, str) and "." in value.split("@")[-1]

    def is_non_empty(value: object) -> bool:
        """Check if value is non-empty."""
        return isinstance(value, str) and len(value.strip()) > 0

    # Compose predicates
    def is_valid_email(value: object) -> bool:
        """Compose email validation predicates."""
        return (
            is_string(value)
            and is_non_empty(value)
            and has_at_symbol(value)
            and has_domain_part(value)
        )

    for value in test_values:
        # Individual predicate results
        is_string(value)
        is_non_empty(value)
        has_at_symbol(value)
        has_domain_part(value)

        # Composed result
        is_valid_email(value)


def validate_customer_complete(
    customer_data: TUserData,
) -> FlextResult[SharedUser]:
    """Validate customer data using shared domain models and utility validation.

    Refactored to reduce complexity with single responsibility methods.
    """
    (f"🔍 Validating customer: {customer_data.get('name', 'Unknown')}")

    # Extract and validate input data types
    input_result = _extract_and_validate_customer_input(customer_data)
    if input_result.is_failure:
        return FlextResult.fail(input_result.error or "Input validation failed")

    if input_result.data is not None:
        name_value, email_value, age_value = input_result.data
    else:
        return FlextResult.fail("Input validation returned None")

    # Create user with factory
    user_result = _create_user_with_factory(name_value, email_value, age_value)
    if user_result.is_failure:
        return user_result

    user = user_result.data

    # Apply business validation
    if user is not None:
        return _validate_and_finalize_customer(user)
    return FlextResult.fail("User creation returned None")


def _extract_and_validate_customer_input(
    customer_data: TUserData,
) -> FlextResult[tuple[str, str, int]]:
    """Extract and validate customer input data types."""
    name_value = customer_data.get("name", "")
    email_value = customer_data.get("email", "")
    age_value = customer_data.get("age", 0)

    # Type validation
    if not isinstance(name_value, str):
        return FlextResult.fail("Name must be a string")
    if not isinstance(email_value, str):
        return FlextResult.fail("Email must be a string")
    if not isinstance(age_value, int):
        return FlextResult.fail("Age must be an integer")

    return FlextResult.ok((name_value, email_value, age_value))


def _create_user_with_factory(
    name: str,
    email: str,
    age: int,
) -> FlextResult[SharedUser]:
    """Create user using SharedDomainFactory with validation."""
    user_result = SharedDomainFactory.create_user(
        name=name,
        email=email,
        age=age,
    )

    if user_result.is_failure:
        return FlextResult.fail(f"User creation failed: {user_result.error}")

    user = user_result.data
    if user is None:
        return FlextResult.fail("User creation returned None data")
    return FlextResult.ok(user)


def _validate_and_finalize_customer(user: SharedUser) -> FlextResult[SharedUser]:
    """Apply business rules and finalize customer validation."""
    try:
        business_validation = validate_user_business_rules(user)
        if business_validation.is_failure:
            error_msg = business_validation.error or "Business validation failed"
            return FlextResult.fail(error_msg)

        # Log successful validation

        return FlextResult.ok(user)

    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Failed to validate shared customer: {e}"
        return FlextResult.fail(error_message)


def validate_product_complete(
    product_data: TAnyObject,
) -> FlextResult[SharedProduct]:
    """Validate product data completely using validation orchestrator.

    Refactored to reduce complexity with single responsibility methods.
    """
    f"🔍 Validating product: {cast('dict[str, object]', product_data).get('name', 'Unknown')}"

    # Validate fields using orchestrator pattern
    field_result = _validate_product_fields(product_data)
    if field_result.is_failure:
        return FlextResult.fail(field_result.error or "Field validation failed")

    # Extract and validate types
    type_result = _extract_and_validate_product_types(product_data)
    if type_result.is_failure:
        return FlextResult.fail(type_result.error or "Type validation failed")

    validated_data = type_result.data

    # Create and finalize product
    if validated_data is not None:
        return _create_and_finalize_product(validated_data)
    return FlextResult.fail("Type validation returned None")


def _validate_product_fields(product_data: TAnyObject) -> FlextResult[None]:
    """Validate product fields using orchestrator pattern."""
    orchestrator = ProductValidationOrchestrator(product_data)
    field_validation = orchestrator.validate_all_fields()

    if field_validation.is_failure:
        error_msg = field_validation.error or "Field validation failed"
        return FlextResult.fail(error_msg)

    return FlextResult.ok(None)


def _extract_and_validate_product_types(
    product_data: TAnyObject,
) -> FlextResult[dict[str, object]]:
    """Extract and validate product data types."""
    # Extract fields
    product_dict = cast("dict[str, object]", product_data)
    name = product_dict.get("name")
    price = product_dict.get("price")
    category = product_dict.get("category")
    stock = product_dict.get("stock")
    product_id = product_dict.get("product_id")

    # Type validation for required fields
    if not isinstance(name, str):
        return FlextResult.fail("Product name must be a string")
    if not isinstance(price, (int, float)):
        return FlextResult.fail("Product price must be a number")
    if not isinstance(category, str):
        return FlextResult.fail("Product category must be a string")
    if not isinstance(stock, int):
        return FlextResult.fail("Product stock must be an integer")

    return FlextResult.ok(
        {
            "name": name,
            "price": price,
            "category": category,
            "stock": stock,
            "product_id": product_id,
        },
    )


def _create_and_finalize_product(
    validated_data: dict[str, object],
) -> FlextResult[SharedProduct]:
    """Create product and perform final validation."""
    try:
        # Create product using SharedDomainFactory
        product_result = SharedDomainFactory.create_product(
            name=str(validated_data["name"]),
            description=(
                f"Product {validated_data['name']} in "
                f"{validated_data['category']} category"
            ),
            price_amount=str(validated_data["price"]),
            currency="USD",
            category=str(validated_data["category"]),
            in_stock=int(cast("int", validated_data["stock"])) > 0,
            id=validated_data["product_id"],
        )

        if product_result.is_failure:
            return FlextResult.fail(f"Product creation failed: {product_result.error}")

        shared_product = product_result.data
        if shared_product is None:
            return FlextResult.fail("Product creation returned None data")

        # Create enhanced demo product and validate inventory
        return _create_enhanced_demo_product(shared_product)

    except (RuntimeError, ValueError, TypeError) as e:
        return FlextResult.fail(f"Product validation error: {e}")


def _create_enhanced_demo_product(
    shared_product: SharedProduct,
) -> FlextResult[SharedProduct]:
    """Create enhanced demo product and validate inventory."""
    try:
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
            error_msg = inventory_validation.error or "Inventory validation failed"
            return FlextResult.fail(error_msg)

        return FlextResult.ok(enhanced_product)

    except (RuntimeError, ValueError, TypeError) as e:
        return FlextResult.fail(f"Failed to create enhanced product: {e}")


def demonstrate_customer_validation() -> None:
    """Demonstrate customer validation workflows using flext_core.typings."""
    "\n" + "=" * 60

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

    for _test_name, customer_data in test_customers:
        validation_result = validate_customer_complete(customer_data)
        if validation_result.success:
            customer = validation_result.data
            if customer is None:
                continue


def demonstrate_product_validation() -> None:
    """Demonstrate product validation workflows using flext_core.typings."""
    "\n" + "=" * 60

    # Test product data
    test_products: list[tuple[str, dict[str, object]]] = [
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

    for _test_name, product_data in test_products:
        validation_result = validate_product_complete(product_data)
        if validation_result.success:
            product = validation_result.data
            if product is None:
                continue


def demonstrate_validation_performance() -> None:
    """Demonstrate validation performance patterns using flext_core.typings."""
    "\n" + "=" * 60

    # Benchmark data - use object for mixed value types
    test_data: list[object] = [
        "valid@email.com",
        "invalid-email",
        "user@domain.co.uk",
        "test@example.com",
        "not-an-email",
    ] * 200  # 1000 total tests

    # Benchmark email validation

    def validate_email_simple(email: object) -> bool:
        """Validate email format simply."""
        return isinstance(email, str) and "@" in email and "." in email.split("@")[-1]

    start_time = time.time()
    valid_count = 0
    for email in test_data:
        if validate_email_simple(email):
            valid_count += 1
    end_time = time.time()

    validation_time = end_time - start_time
    f"✅ Validated {len(test_data)} emails in {validation_time:.4f}s"
    f"   📊 Valid emails: {valid_count}/{len(test_data)}"

    # Benchmark type checking

    start_time = time.time()
    type_results = [isinstance(item, str) for item in test_data]
    end_time = time.time()

    type_check_time = end_time - start_time
    string_count = sum(type_results)
    f"✅ Type checked {len(test_data)} items in {type_check_time:.4f}s"
    f"   📊 String items: {string_count}/{len(test_data)}"

    # Performance summary
    f"   Total Operations: {len(test_data)}"


def main() -> None:
    """Run comprehensive FlextValidators demonstration using shared pattern."""
    # DRY PRINCIPLE: Use SharedDemonstrationPattern to eliminate duplication
    SharedDemonstrationPattern.run_demonstration(
        "FLEXT VALIDATION - ADVANCED VALIDATION SYSTEM DEMONSTRATION",
        [
            demonstrate_basic_validations,
            demonstrate_format_validations,
            demonstrate_functional_predicates,
            demonstrate_predicate_composition,
            demonstrate_customer_validation,
            demonstrate_product_validation,
            demonstrate_validation_performance,
        ],
    )


if __name__ == "__main__":
    main()
