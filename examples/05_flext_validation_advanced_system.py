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

from flext_core import FlextResult

from .shared_domain import SharedDomainFactory, User

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
            return FlextResult.ok(value.strip())
        return FlextResult[str].fail(f"{field_name} is required")

    @staticmethod
    def email_format(email: str) -> FlextResult[str]:
        """Validate email format."""
        if "@" in email and "." in email.split("@")[1]:
            return FlextResult.ok(email)
        return FlextResult[str].fail("Invalid email format")

    @staticmethod
    def age_range(age: int, min_age: int = 0, max_age: int = 150) -> FlextResult[int]:
        """Validate age is in valid range."""
        if min_age <= age <= max_age:
            return FlextResult.ok(age)
        return FlextResult[str].fail(f"Age must be between {min_age} and {max_age}")

    @staticmethod
    def min_length(
        value: str, min_len: int, field_name: str = "Field"
    ) -> FlextResult[str]:
        """Validate minimum string length."""
        if len(value) >= min_len:
            return FlextResult.ok(value)
        return FlextResult[str].fail(
            f"{field_name} must be at least {min_len} characters"
        )

    @staticmethod
    def positive_number(value: float, field_name: str = "Value") -> FlextResult[float]:
        """Validate number is positive."""
        if value > 0:
            return FlextResult.ok(value)
        return FlextResult[str].fail(f"{field_name} must be positive")


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
            .flat_map(lambda e: BasicValidators.email_format(e))
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
        age = int(data.get("age", 0))

        # Validate each field
        name_result = ValidationChains.validate_name(name)
        email_result = ValidationChains.validate_email(email)
        age_result = ValidationChains.validate_age(age)

        # Combine results
        if all(r.success for r in [name_result, email_result, age_result]):
            return FlextResult.ok(
                {
                    "name": name_result.unwrap(),
                    "email": email_result.unwrap(),
                    "age": age_result.unwrap(),
                }
            )

        # Collect all errors
        errors = [
            r.error or "Unknown error"
            for r in [name_result, email_result, age_result]
            if r.failure
        ]
        return FlextResult[str].fail(f"Validation failed: {'; '.join(errors)}")

    @staticmethod
    def validate_product_data(
        data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate product form."""
        name = str(data.get("name", ""))
        price = float(data.get("price", 0))
        category = str(data.get("category", ""))

        # Chain validations
        name_result = BasicValidators.required(name, "Product name").flat_map(
            lambda n: BasicValidators.min_length(n, 3, "Product name")
        )

        price_result = BasicValidators.positive_number(price, "Price")

        category_result = BasicValidators.required(category, "Category")

        # Combine results
        if all(r.success for r in [name_result, price_result, category_result]):
            return FlextResult.ok(
                {
                    "name": name_result.unwrap(),
                    "price": price_result.unwrap(),
                    "category": category_result.unwrap(),
                }
            )

        errors = [
            r.error or "Unknown error"
            for r in [name_result, price_result, category_result]
            if r.failure
        ]
        return FlextResult[str].fail(f"Product validation failed: {'; '.join(errors)}")


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
                    data["name"], data["email"], data["age"]
                )
            )
            .flat_map(lambda user: BusinessRules._check_user_eligibility(user))
        )

    @staticmethod
    def _check_user_eligibility(user: User) -> FlextResult[User]:
        """Check if user meets business eligibility requirements."""
        # Business rule: Users under 18 need parental consent
        if user.age.value < MIN_AGE_ADULT:
            return FlextResult[str].fail("Users under 18 require parental consent")

        # Business rule: Email domain validation
        email = user.email_address.email
        if email.endswith("@blacklisted.com"):
            return FlextResult[str].fail("Email domain not allowed")

        return FlextResult.ok(user)


# =============================================================================
# BATCH VALIDATOR - Process multiple items
# =============================================================================


class BatchValidator:
    """Validate batches of data."""

    @staticmethod
    def validate_user_batch(user_list: list[dict]) -> FlextResult[dict]:
        """Validate multiple users in batch."""
        if not user_list:
            return FlextResult[str].fail("No users to validate")

        results = []
        errors = []

        for i, user_data in enumerate(user_list):
            validation_result = BusinessRules.validate_user_registration(user_data)
            if validation_result.success:
                results.append(validation_result.unwrap())
            else:
                errors.append(f"User {i}: {validation_result.error}")

        return FlextResult.ok(
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
        print(f"✅ Valid name: {name_result.unwrap()}")

    if email_result.success:
        print(f"✅ Valid email: {email_result.unwrap()}")

    if age_result.success:
        print(f"✅ Valid age: {age_result.unwrap()}")

    # Test invalid data
    invalid_email = BasicValidators.email_format("invalid-email")
    if invalid_email.failure:
        print(f"✅ Correctly rejected invalid email: {invalid_email.error}")


def demo_validation_chains() -> None:
    """Demonstrate chained validation."""
    print("\n🧪 Testing validation chains...")

    # Valid data
    name_result = ValidationChains.validate_name("Bob Smith")
    email_result = ValidationChains.validate_email("bob@example.com")
    age_result = ValidationChains.validate_age(30)

    if name_result.success:
        print(f"✅ Name chain validation: {name_result.unwrap()}")

    if email_result.success:
        print(f"✅ Email chain validation: {email_result.unwrap()}")

    if age_result.success:
        print(f"✅ Age chain validation: {age_result.unwrap()}")

    # Invalid data that should fail multiple rules
    invalid_name = ValidationChains.validate_name("A")  # Too short
    if invalid_name.failure:
        print(f"✅ Name chain rejected: {invalid_name.error}")


def demo_form_validation() -> None:
    """Demonstrate complete form validation."""
    print("\n🧪 Testing form validation...")

    # Valid user form
    valid_user_data = {"name": "Carol Davis", "email": "carol@example.com", "age": 28}

    user_result = FormValidator.validate_user_data(valid_user_data)
    if user_result.success:
        validated = user_result.unwrap()
        print(f"✅ User form validated: {validated['name']}")

    # Valid product form
    valid_product_data = {
        "name": "Laptop Computer",
        "price": 999.99,
        "category": "Electronics",
    }

    product_result = FormValidator.validate_product_data(valid_product_data)
    if product_result.success:
        validated_product = product_result.unwrap()
        print(f"✅ Product form validated: {validated_product['name']}")


def demo_business_rules() -> None:
    """Demonstrate business rule validation."""
    print("\n🧪 Testing business rules...")

    # Valid registration
    valid_data = {"name": "David Wilson", "email": "david@example.com", "age": 25}

    registration_result = BusinessRules.validate_user_registration(valid_data)
    if registration_result.success:
        user = registration_result.unwrap()
        print(f"✅ User registration approved: {user.name}")

    # Invalid registration - underage
    underage_data = {"name": "Young User", "email": "young@example.com", "age": 16}

    underage_result = BusinessRules.validate_user_registration(underage_data)
    if underage_result.failure:
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

    batch_result = BatchValidator.validate_user_batch(user_batch)
    if batch_result.success:
        result = batch_result.unwrap()
        print(f"✅ Batch validation: {result['valid']}/{result['total']} valid")
        print(f"   Success rate: {result['success_rate']:.1f}%")

        if result["errors"]:
            print(f"   Errors: {len(result['errors'])} failed validations")


def demo_functional_composition() -> None:
    """Demonstrate functional validation composition."""
    print("\n🧪 Testing functional composition...")

    # Chain multiple validation operations
    result = (
        FlextResult.ok(
            {"name": "Helen Taylor", "email": "helen@example.com", "age": 29}
        )
        .flat_map(lambda data: FormValidator.validate_user_data(data))
        .flat_map(
            lambda data: SharedDomainFactory.create_user(
                data["name"], data["email"], data["age"]
            )
        )
        .map(lambda user: {"user": user, "status": "validated"})
    )

    if result.success:
        response = result.unwrap()
        user = response["user"]
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
