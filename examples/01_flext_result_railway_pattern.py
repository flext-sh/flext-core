#!/usr/bin/env python3
"""FLEXT Result - Railway Pattern Example.

Demonstrates advanced error handling using FlextResult with railway-oriented
programming.
Shows how to chain operations safely without exception handling.

Features demonstrated:
- Railway pattern for safe operation chaining
- Error handling without exceptions
- Data transformation pipelines
- Recovery patterns
- Result combination
"""

from __future__ import annotations

import json
import random
from typing import Any

from flext_core import FlextResult
from flext_core.utilities import FlextUtilities
from flext_core.validation import FlextValidation

# =============================================================================
# DOMAIN MODELS - Real-world user data processing
# =============================================================================


class User:
    """User domain model with validation."""

    def __init__(self, name: str, email: str, age: int) -> None:
        self.name = name
        self.email = email
        self.age = age

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "id": FlextUtilities.generate_entity_id(),
            "created_at": FlextUtilities.generate_iso_timestamp(),
        }


# =============================================================================
# BUSINESS LOGIC - Railway pattern implementation
# =============================================================================


def validate_user_data(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
    """Validate user input data using FLEXT validation."""
    print(f"🔍 Validating user data: {data}")

    # Chain multiple validations using railway pattern
    return (
        FlextResult.ok(data)
        .filter(
            lambda d: "name" in d and "email" in d and "age" in d,
            "Missing required fields: name, email, age",
        )
        .filter(
            lambda d: FlextValidation.is_non_empty_string(d["name"]),
            "Name must be a non-empty string",
        )
        .filter(lambda d: FlextValidation.is_email(d["email"]), "Invalid email format")
        .filter(
            lambda d: isinstance(d["age"], int) and 18 <= d["age"] <= 120,
            "Age must be between 18 and 120",
        )
        .tap(lambda d: print(f"✅ Validation successful for: {d['name']}"))
    )


def create_user(validated_data: dict[str, Any]) -> FlextResult[User]:
    """Create User instance from validated data."""
    print(f"👤 Creating user from: {validated_data}")

    try:
        user = User(
            name=validated_data["name"],
            email=validated_data["email"],
            age=validated_data["age"],
        )
        print(f"✅ User created: {user.name} ({user.email})")
        return FlextResult.ok(user)
    except Exception as e:
        return FlextResult.fail(f"Failed to create user: {e}")


def save_user_to_database(user: User) -> FlextResult[str]:
    """Simulate saving user to database."""
    print(f"💾 Saving user to database: {user.name}")

    # Simulate database save with potential failure
    user_id = FlextUtilities.generate_entity_id()

    # Simulate occasional database failure
    if random.random() < 0.2:  # 20% chance of failure
        return FlextResult.fail("Database connection timeout")

    print(f"✅ User saved with ID: {user_id}")
    return FlextResult.ok(user_id)


def send_welcome_email(user: User) -> FlextResult[bool]:
    """Simulate sending welcome email."""
    print(f"📧 Sending welcome email to: {user.email}")

    # Simulate email service
    if "@invalid.com" in user.email:
        return FlextResult.fail("Email service rejected invalid domain")

    print(f"✅ Welcome email sent to: {user.email}")
    return FlextResult.ok(True)


def process_user_registration(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
    """Complete user registration pipeline using railway pattern."""
    print("\n🚀 Starting user registration pipeline...")

    # Railway pattern: chain operations with automatic error propagation
    result = (
        validate_user_data(data)
        .flat_map(create_user)
        .flat_map(
            lambda user:
            # Combine database save and email sending
            FlextResult.combine(
                save_user_to_database(user),
                send_welcome_email(user),
                lambda user_id, email_sent: {
                    "user": user.to_dict(),
                    "user_id": user_id,
                    "email_sent": email_sent,
                    "status": "registered",
                },
            ),
        )
    )

    if result.is_success:
        print("🎉 Registration completed successfully!")
        return result
    print(f"❌ Registration failed: {result.error}")
    return result


def process_multiple_users(
    users_data: list[dict[str, Any]],
) -> FlextResult[list[dict[str, Any]]]:
    """Process multiple users with error aggregation."""
    print(f"\n📊 Processing {len(users_data)} users...")

    results = []
    successful = 0
    failed = 0

    for i, user_data in enumerate(users_data):
        print(f"\n--- Processing user {i + 1}/{len(users_data)} ---")
        result = process_user_registration(user_data)

        if result.is_success:
            results.append(result.data)
            successful += 1
        else:
            # Recovery pattern: log error and continue
            print(f"⚠️  User {i + 1} failed: {result.error}")
            failed += 1

    summary = {
        "total_processed": len(users_data),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(users_data) * 100,
        "results": results,
    }

    print("\n📈 Batch processing summary:")
    print(f"   Total: {summary['total_processed']}")
    print(f"   Successful: {summary['successful']}")
    print(f"   Failed: {summary['failed']}")
    print(f"   Success rate: {summary['success_rate']:.1f}%")

    return FlextResult.ok(summary)


# =============================================================================
# ADVANCED PATTERNS - Error recovery and transformation
# =============================================================================


def process_with_retry(
    data: dict[str, Any],
    max_retries: int = 3,
) -> FlextResult[dict[str, Any]]:
    """Process with retry logic using FlextResult."""
    print(f"\n🔄 Processing with retry (max {max_retries} attempts)...")

    for attempt in range(1, max_retries + 1):
        print(f"   Attempt {attempt}/{max_retries}")
        result = process_user_registration(data)

        if result.is_success:
            print(f"✅ Succeeded on attempt {attempt}")
            return result

        if attempt < max_retries:
            print(f"⚠️  Attempt {attempt} failed: {result.error}, retrying...")
        else:
            print(f"❌ All {max_retries} attempts failed")
            return result.recover(
                lambda error: FlextResult.ok(
                    {
                        "status": "failed_after_retries",
                        "error": error,
                        "attempts": max_retries,
                    },
                ),
            )

    return FlextResult.fail("Unexpected retry loop exit")


def transform_user_data(raw_data: str) -> FlextResult[dict[str, Any]]:
    """Transform raw JSON string to user data with validation."""
    print(f"🔄 Transforming raw data: {raw_data[:50]}...")

    return (
        FlextResult.ok(raw_data)
        .filter(
            lambda s: isinstance(s, str) and len(s) > 0,
            "Input must be non-empty string",
        )
        .flat_map(lambda s: FlextUtilities.safe_call(lambda: json.loads(s)))
        .filter(lambda d: isinstance(d, dict), "Parsed data must be a dictionary")
        .tap(lambda d: print(f"✅ Data transformed: {d}"))
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:
    """Run comprehensive FlextResult demonstration."""
    print("=" * 80)
    print("🚀 FLEXT RESULT - RAILWAY PATTERN DEMONSTRATION")
    print("=" * 80)

    # Example 1: Successful user registration
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Successful User Registration")
    print("=" * 60)

    valid_user = {"name": "Alice Johnson", "email": "alice@example.com", "age": 28}

    result1 = process_user_registration(valid_user)
    if result1.is_success:
        print(f"✅ Success: {json.dumps(result1.data, indent=2)}")
    else:
        print(f"❌ Failed: {result1.error}")

    # Example 2: Validation failure
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Validation Failure")
    print("=" * 60)

    invalid_user = {"name": "", "email": "not-an-email", "age": 15}

    result2 = process_user_registration(invalid_user)
    if result2.is_success:
        print(f"✅ Success: {result2.data}")
    else:
        print(f"❌ Expected failure: {result2.error}")

    # Example 3: Batch processing
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Batch Processing")
    print("=" * 60)

    users_batch = [
        {"name": "Bob Smith", "email": "bob@example.com", "age": 35},
        {"name": "Carol Davis", "email": "carol@example.com", "age": 42},
        {"name": "", "email": "invalid", "age": 16},  # This will fail
        {"name": "David Wilson", "email": "david@example.com", "age": 29},
        {"name": "Eve Brown", "email": "eve@invalid.com", "age": 33},  # This might fail
    ]

    batch_result = process_multiple_users(users_batch)
    if batch_result.is_success:
        print("✅ Batch processing completed!")

    # Example 4: JSON transformation pipeline
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: JSON Transformation Pipeline")
    print("=" * 60)

    json_data = '{"name": "Frank Miller", "email": "frank@example.com", "age": 45}'

    pipeline_result = transform_user_data(json_data).flat_map(process_user_registration)

    if pipeline_result.is_success:
        print(
            "✅ Pipeline success:"
            f" User {pipeline_result.data['user']['name']} processed",
        )
    else:
        print(f"❌ Pipeline failed: {pipeline_result.error}")

    # Example 5: Retry pattern
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Retry Pattern")
    print("=" * 60)

    retry_user = {"name": "Grace Taylor", "email": "grace@example.com", "age": 31}

    retry_result = process_with_retry(retry_user, max_retries=3)
    if retry_result.is_success:
        print("✅ Retry pattern success!")

    print("\n" + "=" * 80)
    print("🎉 FLEXT RESULT DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
