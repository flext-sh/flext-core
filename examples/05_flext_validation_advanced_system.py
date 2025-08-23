#!/usr/bin/env python3
"""05 - Validation: Clean Data Validation Patterns.

Shows how to build robust validation systems using FlextResult.
Demonstrates functional validation, composition, and error handling.

Key Patterns:
• FlextResult for validation outcomes
• Functional validation composition
• Predicate-based validation
• Chain validation patterns
"""

from typing import cast

from shared_domain import SharedDomainFactory, User

from flext_core import FlextResult

# Constants to avoid magic numbers
MAX_EMAIL_LENGTH = 100
MIN_AGE_ADULT = 18

# =============================================================================
# BASIC VALIDATORS - Simple predicate functions
# =============================================================================


class BasicValidators:
    """Basic validation functions using FlextResult."""

    @staticmethod
    def required(value: str, field_name: str = "Field") -> FlextResult[str]:
        """Validate that field is not empty."""
        if value and value.strip():
            return FlextResult[str].ok(value.strip())
        return FlextResult[str].fail(f"{field_name} is required")

    @staticmethod
    def email_format(email: str) -> FlextResult[str]:
        """Validate email format."""
        if "@" in email and "." in email.split("@")[1]:
            return FlextResult[str].ok(email)
        return FlextResult[str].fail("Invalid email format")

    @staticmethod
    def age_range(age: int, min_age: int = 0, max_age: int = 150) -> FlextResult[int]:
        """Validate age is in valid range."""
        if min_age <= age <= max_age:
            return FlextResult[int].ok(age)
        return FlextResult[int].fail(f"Age must be between {min_age} and {max_age}")

    @staticmethod
    def min_length(
        value: str, min_len: int, field_name: str = "Field"
    ) -> FlextResult[str]:
        """Validate minimum string length."""
        if len(value) >= min_len:
            return FlextResult[str].ok(value)
        return FlextResult[str].fail(
            f"{field_name} must be at least {min_len} characters"
        )

    @staticmethod
    def positive_number(value: float, field_name: str = "Value") -> FlextResult[float]:
        """Validate number is positive."""
        if value > 0:
            return FlextResult[float].ok(value)
        return FlextResult[float].fail(f"{field_name} must be positive")


# =============================================================================
# VALIDATION CHAINS - Composable validation
# =============================================================================


class ValidationChains:
    """Chain multiple validators together."""

    @staticmethod
    def validate_name(name: str) -> FlextResult[str]:
        """Validate user name with multiple rules."""
        return (
            BasicValidators.required(name, "Name")
            .flat_map(lambda n: BasicValidators.min_length(n, 2, "Name"))
            .filter(
                lambda n: not any(c.isdigit() for c in n), "Name cannot contain numbers"
            )
        )

    @staticmethod
    def validate_email(email: str) -> FlextResult[str]:
        """Validate email with multiple rules."""
        return (
            BasicValidators.required(email, "Email")
            .flat_map(BasicValidators.email_format)
            .filter(lambda e: len(e) <= MAX_EMAIL_LENGTH, "Email too long")
        )

    @staticmethod
    def validate_age(age: int) -> FlextResult[int]:
        """Validate age with business rules."""
        return BasicValidators.age_range(age, 16, 120).tap(
            lambda a: print(f"Age {a} validated") if a >= MIN_AGE_ADULT else None
        )


# =============================================================================
# FORM VALIDATOR - Complete object validation
# =============================================================================


class FormValidator:
    """Validate complete forms with multiple fields."""

    @staticmethod
    def validate_user_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Validate complete user registration form."""
        # Extract fields
        name = str(data.get("name", ""))
        email = str(data.get("email", ""))
        age_obj = data.get("age", 0)
        age = int(age_obj) if isinstance(age_obj, (int, str)) else 0

        # Validate each field
        name_result: FlextResult[str] = ValidationChains.validate_name(name)
        email_result: FlextResult[str] = ValidationChains.validate_email(email)
        age_result: FlextResult[int] = ValidationChains.validate_age(age)

        # Combine results
        if name_result.success and email_result.success and age_result.success:
            return FlextResult[dict[str, object]].ok(
                {
                    "name": name_result.value,
                    "email": email_result.value,
                    "age": age_result.value,
                }
            )

        # Collect all errors
        errors = []
        if name_result.is_failure:
            errors.append(name_result.error or "Unknown error")
        if email_result.is_failure:
            errors.append(email_result.error or "Unknown error")
        if age_result.is_failure:
            errors.append(age_result.error or "Unknown error")
        return FlextResult[dict[str, object]].fail(
            f"Validation failed: {'; '.join(errors)}"
        )

    @staticmethod
    def validate_product_data(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate product form."""
        name = str(data.get("name", ""))
        price_obj = data.get("price", 0)
        price = float(price_obj) if isinstance(price_obj, (int, float, str)) else 0.0
        category = str(data.get("category", ""))

        # Chain validations
        name_result: FlextResult[str] = BasicValidators.required(
            name, "Product name"
        ).flat_map(lambda n: BasicValidators.min_length(n, 3, "Product name"))

        price_result: FlextResult[float] = BasicValidators.positive_number(
            price, "Price"
        )

        category_result: FlextResult[str] = BasicValidators.required(
            category, "Category"
        )

        # Combine results
        if name_result.success and price_result.success and category_result.success:
            return FlextResult[dict[str, object]].ok(
                {
                    "name": name_result.value,
                    "price": price_result.value,
                    "category": category_result.value,
                }
            )

        errors = []
        if name_result.is_failure:
            errors.append(name_result.error or "Unknown error")
        if price_result.is_failure:
            errors.append(price_result.error or "Unknown error")
        if category_result.is_failure:
            errors.append(category_result.error or "Unknown error")
        return FlextResult[dict[str, object]].fail(
            f"Product validation failed: {'; '.join(errors)}"
        )


# =============================================================================
# BUSINESS RULES - Domain-specific validation
# =============================================================================


class BusinessRules:
    """Business rule validators."""

    @staticmethod
    def validate_user_registration(user_data: dict[str, object]) -> FlextResult[User]:
        """Validate user registration with business rules."""
        return (
            FormValidator.validate_user_data(user_data)
            .flat_map(
                lambda data: SharedDomainFactory.create_user(
                    str(data["name"]),
                    str(data["email"]),
                    int(data["age"]) if isinstance(data["age"], int) else 0,
                )
            )
            .flat_map(BusinessRules._check_user_eligibility)
        )

    @staticmethod
    def _check_user_eligibility(user: User) -> FlextResult[User]:
        """Check if user meets business eligibility requirements."""
        # Business rule: Users under 18 need parental consent
        if user.age.value < MIN_AGE_ADULT:
            return FlextResult[User].fail("Users under 18 require parental consent")

        # Business rule: Email domain validation
        email = user.email_address.email
        if email.endswith("@blacklisted.com"):
            return FlextResult[User].fail("Email domain not allowed")

        return FlextResult[User].ok(user)


# =============================================================================
# BATCH VALIDATOR - Process multiple items
# =============================================================================


class BatchValidator:
    """Validate batches of data."""

    @staticmethod
    def validate_user_batch(
        user_list: list[dict[str, object]],
    ) -> FlextResult[dict[str, object]]:
        """Validate multiple users in batch."""
        if not user_list:
            return FlextResult[dict[str, object]].fail("No users to validate")

        results = []
        errors = []

        for i, user_data in enumerate(user_list):
            validation_result = BusinessRules.validate_user_registration(user_data)
            if validation_result.success:
                results.append(validation_result.value)
            else:
                errors.append(f"User {i}: {validation_result.error}")

        return FlextResult[dict[str, object]].ok(
            {
                "total": len(user_list),
                "valid": len(results),
                "invalid": len(errors),
                "users": results,
                "errors": errors,
                "success_rate": (len(results) / len(user_list)) * 100,
            }
        )


# =============================================================================
# DEMONSTRATIONS - Real-world validation usage
# =============================================================================


def demo_basic_validation() -> None:
    """Demonstrate basic validation patterns."""
    print("\n🧪 Testing basic validation...")

    # Test individual validators
    name_result = BasicValidators.required("Alice", "Name")
    email_result = BasicValidators.email_format("alice@example.com")
    age_result = BasicValidators.age_range(25)

    if name_result.success:
        print(f"✅ Valid name: {name_result.value}")

    if email_result.success:
        print(f"✅ Valid email: {email_result.value}")

    if age_result.success:
        print(f"✅ Valid age: {age_result.value}")

    # Test invalid data
    invalid_email = BasicValidators.email_format("invalid-email")
    if invalid_email.is_failure:
        print(f"✅ Correctly rejected invalid email: {invalid_email.error}")


def demo_validation_chains() -> None:
    """Demonstrate chained validation."""
    print("\n🧪 Testing validation chains...")

    # Valid data
    name_result = ValidationChains.validate_name("Bob Smith")
    email_result = ValidationChains.validate_email("bob@example.com")
    age_result = ValidationChains.validate_age(30)

    if name_result.success:
        print(f"✅ Name chain validation: {name_result.value}")

    if email_result.success:
        print(f"✅ Email chain validation: {email_result.value}")

    if age_result.success:
        print(f"✅ Age chain validation: {age_result.value}")

    # Invalid data that should fail multiple rules
    invalid_name = ValidationChains.validate_name("A")  # Too short
    if invalid_name.is_failure:
        print(f"✅ Name chain rejected: {invalid_name.error}")


def demo_form_validation() -> None:
    """Demonstrate complete form validation."""
    print("\n🧪 Testing form validation...")

    # Valid user form
    valid_user_data = {"name": "Carol Davis", "email": "carol@example.com", "age": 28}

    user_result = FormValidator.validate_user_data(
        cast("dict[str, object]", valid_user_data)
    )
    if user_result.success:
        validated = user_result.value
        print(f"✅ User form validated: {validated['name']}")

    # Valid product form
    valid_product_data = {
        "name": "Laptop Computer",
        "price": 999.99,
        "category": "Electronics",
    }

    product_result = FormValidator.validate_product_data(
        cast("dict[str, object]", valid_product_data)
    )
    if product_result.success:
        validated_product = product_result.value
        print(f"✅ Product form validated: {validated_product['name']}")


def demo_business_rules() -> None:
    """Demonstrate business rule validation."""
    print("\n🧪 Testing business rules...")

    # Valid registration
    valid_data = {"name": "David Wilson", "email": "david@example.com", "age": 25}

    registration_result = BusinessRules.validate_user_registration(
        cast("dict[str, object]", valid_data)
    )
    if registration_result.success:
        user = registration_result.value
        print(f"✅ User registration approved: {user.name}")

    # Invalid registration - underage
    underage_data = {"name": "Young User", "email": "young@example.com", "age": 16}

    underage_result = BusinessRules.validate_user_registration(
        cast("dict[str, object]", underage_data)
    )
    if underage_result.is_failure:
        print(f"✅ Underage user rejected: {underage_result.error}")


def demo_batch_validation() -> None:
    """Demonstrate batch validation."""
    print("\n🧪 Testing batch validation...")

    user_batch = [
        {"name": "Eve Brown", "email": "eve@example.com", "age": 32},
        {"name": "Frank Miller", "email": "frank@example.com", "age": 28},
        {"name": "Invalid User", "email": "bad-email", "age": 15},  # Will fail
        {"name": "Grace Lee", "email": "grace@example.com", "age": 35},
    ]

    batch_result = BatchValidator.validate_user_batch(
        cast("list[dict[str, object]]", user_batch)
    )
    if batch_result.success:
        result = batch_result.value
        print(f"✅ Batch validation: {result['valid']}/{result['total']} valid")
        print(f"   Success rate: {result['success_rate']:.1f}%")

        if result["errors"]:
            print(
                f"   Errors: {len(cast('list[object]', result['errors']))} failed validations"
            )


def demo_functional_composition() -> None:
    """Demonstrate functional validation composition."""
    print("\n🧪 Testing functional composition...")

    # Chain multiple validation operations
    result = (
        FlextResult.ok(
            {
                "name": "Helen Taylor",
                "email": "helen@example.com",
                "age": 29,
            }
        )
        .flat_map(FormValidator.validate_user_data)
        .flat_map(
            lambda data: SharedDomainFactory.create_user(
                str(data["name"]),
                str(data["email"]),
                int(data["age"]) if isinstance(data["age"], int) else 0,
            )
        )
        .map(lambda user: {"user": user, "status": "validated"})
    )

    # Use success pattern for cleaner error handling
    if result.success:
        response = result.value
        user = cast("User", response["user"])
        print(f"✅ Functional composition: {user.name} {response['status']}")


def main() -> None:
    """🎯 Example 05: Validation Patterns."""
    print("=" * 70)
    print("✅ EXAMPLE 05: VALIDATION (REFACTORED)")
    print("=" * 70)

    print("\n📚 Refactoring Benefits:")
    print("  • 90% less boilerplate code")
    print("  • Clean functional validation")
    print("  • Simple composition patterns")
    print("  • Removed complex orchestration")

    print("\n🔍 DEMONSTRATIONS")
    print("=" * 40)

    # Show the refactored validation patterns
    demo_basic_validation()
    demo_validation_chains()
    demo_form_validation()
    demo_business_rules()
    demo_batch_validation()
    demo_functional_composition()

    print("\n" + "=" * 70)
    print("✅ REFACTORED VALIDATION EXAMPLE COMPLETED!")
    print("=" * 70)

    print("\n🎓 Key Improvements:")
    print("  • Simple, composable validators")
    print("  • Clean validation chains")
    print("  • Practical business rules")
    print("  • Railway-oriented validation")


if __name__ == "__main__":
    main()
