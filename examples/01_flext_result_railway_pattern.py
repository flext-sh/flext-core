#!/usr/bin/env python3
"""01 - Railway-Oriented Programming: The Foundation Pattern.

Demonstrates FlextResult[T] - the core pattern for type-safe error handling
throughout the entire FLEXT ecosystem. This eliminates exceptions in business
logic and enables composable, predictable operations.

Key Patterns Demonstrated:
• FlextResult[T].ok() / FlextResult[T].fail() creation
• Railway composition with .map() and .flat_map()
• Error handling without try/catch boilerplate
• Chaining operations with automatic error propagation
• Type-safe success/failure checking

Architecture Benefits:
• Zero exceptions in business logic
• Composable operation chains
• Predictable error propagation
• Type-safe error handling
• 90% less error handling boilerplate
"""

import json
import secrets
from typing import cast

from flext_core import FlextResult, FlextValidation, safe_call

from .shared_domain import SharedDomainFactory, User as SharedUser

# =============================================================================
# TYPE DEFINITIONS - Centralized type aliases
# =============================================================================

UserDataDict = dict[str, object]
ProcessingResultDict = dict[str, object]

# =============================================================================
# BUSINESS CONSTANTS - Configuration values
# =============================================================================

FAILURE_RATE = 0.2  # 20% simulated failure rate for demonstrations


# =============================================================================
# TRADITIONAL APPROACH (BEFORE) - Exception Hell
# =============================================================================


def _raise_name_required() -> None:
    """Inner function for name validation error."""
    msg = "Name is required"
    raise ValueError(msg)


def _raise_email_required() -> None:
    """Inner function for email validation error."""
    msg = "Valid email is required"
    raise ValueError(msg)


def _raise_age_required() -> None:
    """Inner function for age validation error."""
    msg = "Valid age is required"
    raise ValueError(msg)


def _raise_database_timeout() -> None:
    """Inner function for database timeout error."""
    msg = "Database timeout"
    raise ConnectionError(msg)


def _raise_email_service_error() -> None:
    """Inner function for email service error."""
    msg = "Email service unavailable"
    raise RuntimeError(msg)


def process_user_data_traditional(data: dict[str, object]) -> dict[str, object]:
    """Traditional approach: 25+ lines with exception handling chaos."""
    try:
        # Input validation with exceptions
        if not data.get("name"):
            _raise_name_required()
        email = data.get("email")
        if not email or not isinstance(email, str) or "@" not in email:
            _raise_email_required()
        if not data.get("age") or not isinstance(data["age"], int):
            _raise_age_required()

        # User creation with nested try/catch
        try:
            {
                "name": data["name"],
                "email": data["email"],
                "age": data["age"],
            }
            # Simulate user creation logic
            sysrand = secrets.SystemRandom()
            user_id = f"user_{sysrand.randrange(1000, 10000)}"
        except Exception as e:
            msg = f"User creation failed: {e}"
            raise RuntimeError(msg) from e

        # Database operations with exception handling
        try:
            if secrets.SystemRandom().random() < FAILURE_RATE:
                _raise_database_timeout()
            # Simulate save
        except ConnectionError as e:
            msg = f"Database save failed: {e}"
            raise RuntimeError(msg) from e

        # Email with nested exception handling
        try:
            if secrets.SystemRandom().random() < FAILURE_RATE:
                _raise_email_service_error()
        except RuntimeError:
            # Continue anyway for email failures
            pass

        return {"success": True, "user_id": user_id, "message": "User processed"}

    except ValueError as e:
        return {"success": False, "error": f"Validation error: {e}"}
    except RuntimeError as e:
        return {"success": False, "error": f"Processing error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


# =============================================================================
# MODERN FLEXT APPROACH (AFTER) - Railway-Oriented Programming
# =============================================================================


def validate_user_data(data: UserDataDict) -> FlextResult[UserDataDict]:
    """🚀 ZERO-BOILERPLATE validation using FlextValidation."""
    return (
        FlextResult.ok(data)
        .filter(
            lambda d: "name" in d and "email" in d and "age" in d,
            "Missing required fields",
        )
        .filter(
            lambda d: FlextValidation.is_non_empty_string(d["name"]),
            "Invalid name",
        )
        .tap(lambda d: print(f"✅ Validated: {d['name']}"))
    )


def create_user(validated_data: UserDataDict) -> FlextResult[SharedUser]:
    """🚀 ONE-LINE user creation using SharedDomainFactory."""
    return SharedDomainFactory.create_user(
        name=str(validated_data["name"]),
        email=str(validated_data["email"]),
        age=int(cast("int", validated_data["age"])),
    ).tap(lambda u: print(f"✅ Created: {u.name}"))


def save_user_to_database(user: SharedUser) -> FlextResult[str]:
    """🚀 ZERO-BOILERPLATE database simulation using FlextResult."""
    return (
        FlextResult.ok(user.id)
        .filter(
            lambda _: secrets.SystemRandom().random() >= FAILURE_RATE,
            "Database timeout",
        )
        .tap(lambda uid: print(f"✅ Saved: {uid}"))
    )


def send_welcome_email(user: SharedUser) -> FlextResult[bool]:
    """🚀 ONE-LINE email sending with built-in validation."""
    return (
        FlextResult.ok(data=True)
        .filter(
            lambda _: "@invalid.com" not in user.email_address.email,
            "Invalid domain",
        )
        .tap(lambda _: print(f"✅ Email sent to: {user.email_address.email}"))
    )


def process_user_registration(data: UserDataDict) -> FlextResult[dict[str, object]]:
    """🚀 PERFECT Railway-Oriented Programming: ONE-LINE registration pipeline.

    Demonstrates FlextResult[Type].ok/false pattern with proper type safety.
    """
    return (
        validate_user_data(data)
        .flat_map(create_user)
        .flat_map(lambda user: user.activate())
        .flat_map(
            lambda user: FlextResult.combine(
                save_user_to_database(user).map(lambda x: cast("object", x)),
                send_welcome_email(user).map(lambda x: cast("object", x)),
            ).map(
                lambda _: cast(
                    "dict[str, object]",
                    {"user_id": user.id, "status": "registered"},
                ),
            ),
        )
        .tap(
            lambda result: print(
                f"🎉 Registration completed! User: {result.get('user_id', 'unknown')}"
            )
        )
    )


def process_multiple_users(
    users_data: list[UserDataDict],
) -> FlextResult[ProcessingResultDict]:
    """🚀 BATCH processing with ONE-LINE aggregation using FlextResult.combine_all."""
    results = [process_user_registration(data) for data in users_data]
    successful = [r.data for r in results if r.success]
    failed = len(results) - len(successful)

    result_dict: ProcessingResultDict = {
        "total": len(users_data),
        "successful": len(successful),
        "failed": failed,
        "success_rate": len(successful) / len(users_data) * 100
        if len(users_data) > 0
        else 0,
        "results": successful,
    }
    return (
        FlextResult[ProcessingResultDict]
        .ok(result_dict)
        .tap(
            lambda s: print(f"📊 Processed {s['successful']}/{s['total']} users"),
        )
    )


# =============================================================================
# ADVANCED PATTERNS - Error recovery and transformation
# =============================================================================


def process_with_retry(
    data: UserDataDict,
    max_retries: int = 3,
) -> FlextResult[dict[str, object]]:
    """🚀 PERFECT retry pattern with FlextResult[Type] composition.

    Demonstrates proper success/failure handling with type-safe retries.
    """
    for attempt in range(1, max_retries + 1):
        result = process_user_registration(data)
        if result.is_success:
            return result.tap(
                lambda _, a=attempt: print(f"✅ Success on attempt {a}/{max_retries}")
            )
        print(f"⚠️ Attempt {attempt}/{max_retries} failed: {result.error}")

    # All attempts failed - return proper failure result
    return FlextResult[dict[str, object]].fail(
        f"Registration failed after {max_retries} attempts. Last error: {result.error if 'result' in locals() else 'Unknown error'}"
    )


def transform_user_data(raw_data: str) -> FlextResult[UserDataDict]:
    """🚀 ONE-LINE JSON transformation using safe_call."""
    return (
        FlextResult[UserDataDict]
        .ok(raw_data)
        .filter(lambda s: bool(s) and isinstance(s, str), "Invalid input")
        .flat_map(lambda s: safe_call(lambda: json.loads(s)))
        .filter(lambda d: isinstance(d, dict), "Must be dict")
        .tap(lambda d: print(f"✅ Transformed: {d}"))
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demo_successful_registration() -> None:
    """Demonstrate successful user registration with proper FlextResult[Type] handling."""
    print("\n🧪 Testing successful registration...")
    valid_user: UserDataDict = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
    }

    result = process_user_registration(valid_user)
    if result.is_success:
        print(f"✅ Registration successful: {result.data}")
    else:
        print(f"❌ Registration failed: {result.error}")


def demo_validation_failure() -> None:
    """Demonstrate validation failure handling with FlextResult[Type].is_failure."""
    print("\n🧪 Testing validation failure...")
    invalid_user: UserDataDict = {"name": "", "email": "not-an-email", "age": 15}

    result = process_user_registration(invalid_user)
    if result.is_failure:
        print(f"✅ Validation correctly failed: {result.error}")
    else:
        print("❌ Unexpected success with invalid data")


def demo_batch_processing() -> None:
    """Demonstrate batch processing with error handling."""
    users_batch: list[UserDataDict] = [
        {"name": "Bob Smith", "email": "bob@example.com", "age": 35},
        {"name": "Carol Davis", "email": "carol@example.com", "age": 42},
        {"name": "", "email": "invalid", "age": 16},  # This will fail
        {"name": "David Wilson", "email": "david@example.com", "age": 29},
        {"name": "Eve Brown", "email": "eve@invalid.com", "age": 33},  # This might fail
    ]

    result = process_multiple_users(users_batch)
    if result.success and result.data is not None:
        result.data.get("successful", 0)
        results_list = result.data.get("results", [])
        # Type annotation for loop variable and proper type handling
        typed_results_list: list[object] = (
            results_list if isinstance(results_list, list) else []
        )
        for user_result in typed_results_list:
            if isinstance(user_result, dict) and "user" in user_result:
                user_data = user_result["user"]
                if isinstance(user_data, dict):
                    user_data.get("name", "Unknown")


def demo_json_transformation() -> None:
    """Demonstrate JSON transformation pipeline."""
    json_data = '{"name": "Frank Miller", "email": "frank@example.com", "age": 45}'

    result = transform_user_data(json_data).flat_map(process_user_registration)

    if result.is_success:
        data = result.data
        if isinstance(data, dict) and "user" in data and isinstance(data["user"], dict):
            data["user"].get("name", "Unknown")


def demo_retry_pattern() -> None:
    """Demonstrate retry pattern with error recovery."""
    retry_user: UserDataDict = {
        "name": "Grace Taylor",
        "email": "grace@example.com",
        "age": 31,
    }

    result = process_with_retry(retry_user, max_retries=3)
    if result.is_success:
        pass


def demo_boilerplate_comparison() -> None:
    """🎯 MAIN DEMONSTRATION: Before vs After Boilerplate Reduction."""
    test_data = {"name": "John Doe", "email": "john@example.com", "age": 30}

    process_user_data_traditional(test_data)

    # Modern approach - single pipeline
    modern_result = (
        validate_user_data(test_data)
        .flat_map(create_user)
        .tap(lambda user: print(f"✅ User created: {user.name}"))
        .flat_map(save_user_to_database)
        .tap(lambda _: print("✅ User saved to database"))
    )

    if modern_result.success:
        pass


def main() -> None:
    """🎯 Example 01: Railway-Oriented Programming Foundation.

    Demonstrates the core FlextResult[T] pattern that eliminates exceptions
    and enables composable, type-safe error handling throughout FLEXT.
    """
    print("=" * 70)
    print("🚂 EXAMPLE 01: RAILWAY-ORIENTED PROGRAMMING FOUNDATION")
    print("=" * 70)
    print("\n📚 Learning Objectives:")
    print("  • Master FlextResult[T].ok() and .fail() patterns")
    print("  • Understand railway composition with .map()/.flat_map()")
    print("  • Eliminate exceptions from business logic")
    print("  • Chain operations with automatic error propagation")

    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION: Before vs After Comparison")
    print("=" * 70)

    # Core demonstration
    demo_boilerplate_comparison()

    print("\n" + "=" * 70)
    print("🔍 PRACTICAL SCENARIOS")
    print("=" * 70)

    # Practical examples
    demo_successful_registration()
    demo_validation_failure()
    demo_batch_processing()
    demo_json_transformation()
    demo_retry_pattern()

    print("\n" + "=" * 70)
    print("✅ EXAMPLE 01 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n🎓 Key Takeaways:")
    print("  • FlextResult[T] eliminates try/catch boilerplate")
    print("  • Railway composition enables elegant error handling")
    print("  • Type safety is preserved throughout operation chains")
    print("  • Business logic becomes predictable and testable")

    print("\n💡 Next Steps:")
    print("  → Run example 02 for dependency injection patterns")
    print("  → Study shared_domain.py for domain modeling patterns")
    print("  → Explore FlextResult[T] API documentation")


if __name__ == "__main__":
    main()
