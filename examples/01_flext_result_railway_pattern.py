#!/usr/bin/env python3
"""FLEXT Result Railway Pattern - Foundation Example 01.

Enterprise-grade railway-oriented programming demonstration using FlextResult
for type-safe error handling across data transformation pipelines.

Module Role in Architecture:
    Examples Layer → Foundation Examples → Railway Pattern Implementation

    This example demonstrates essential patterns that enable:
    - Error-safe data processing pipelines used in 15,000+ function signatures
    - Type-safe transformation chains without exception handling
    - Recovery patterns for enterprise fault tolerance
    - Result composition for transaction-like operations

Railway Pattern Features:
    ✅ Safe Operation Chaining: map() and flat_map() for transformation pipelines
    ✅ Error Propagation: Automatic error handling without try/catch blocks
    ✅ Data Validation: Input validation with comprehensive error reporting
    ✅ Recovery Strategies: Fallback mechanisms for operational resilience
    ✅ Result Combination: Transactional patterns for complex operations
    ✅ Type Safety: Full type annotations with FlextResult[T] patterns

Enterprise Applications:
    - Data ETL pipelines with error handling
    - API request processing with validation
    - Database transaction management
    - File processing with recovery mechanisms
    - Service integration with fault tolerance

Real-World Usage Context:
    This pattern is foundational to all FLEXT ecosystem projects, enabling
    reliable data processing across 32 interconnected services without
    traditional exception handling overhead.

Architecture Benefits:
    - Composable Operations: Chain multiple transformations safely
    - Predictable Error Handling: Always return FlextResult[T] or error
    - Performance Optimization: No exception overhead in happy path
    - Testing Simplification: Testable error paths without exception mocking

See Also:
    - src/flext_core/result.py: FlextResult implementation
    - src/flext_core/core.py: FlextCore pipeline functions
    - examples/02_flext_container_dependency_injection.py: Next foundation example
    - shared_domain.py: Shared domain models for consistent examples

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import random

# Import ALL domain models from shared_domain - NO local domain models
from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

from flext_core import (
    FlextResult,
    FlextValidation,
    TAnyObject,
    TEntityId,
    TErrorMessage,
    TLogMessage,
    TUserData,
    safe_call,
)

# Constants to avoid magic numbers
FAILURE_RATE = 0.2  # 20% chance of failure


def validate_user_data(data: TUserData) -> FlextResult[TUserData]:
    """Validate user input data using railway pattern."""
    log_message: TLogMessage = f"🔍 Validating user data: {data}"
    print(log_message)

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
        .tap(lambda d: print(f"✅ Basic validation successful for: {d['name']}"))
    )


def create_user(validated_data: TUserData) -> FlextResult[SharedUser]:
    """Create User using ONLY SharedDomainFactory - NO local models."""
    log_message: TLogMessage = (
        f"👤 Creating user entity from shared domain: {validated_data}"
    )
    print(log_message)

    # Use ONLY SharedDomainFactory - complete domain model reuse
    user_result = SharedDomainFactory.create_user(
        name=str(validated_data["name"]),
        email=str(validated_data["email"]),
        age=int(validated_data["age"]),
    )

    if user_result.is_failure:
        return FlextResult.fail(f"User creation failed: {user_result.error}")

    user = user_result.data

    log_domain_operation(
        "user_created_railway",
        "SharedUser",
        user.id,
        name=user.name,
        email=user.email_address.email,
        pattern="railway",
    )

    print(
        f"✅ Shared domain user created: {user.name} ({user.email_address.email})",
    )
    return FlextResult.ok(user)


def save_user_to_database(user: SharedUser) -> FlextResult[TEntityId]:
    """Simulate saving user entity to database."""
    log_message: TLogMessage = f"💾 Saving user entity to database: {user.name}"
    print(log_message)

    # Use the entity's existing ID instead of generating new one
    user_id: TEntityId = user.id

    # Simulate occasional database failure
    if random.random() < FAILURE_RATE:  # noqa: S311
        error_message: TErrorMessage = "Database connection timeout"
        return FlextResult.fail(error_message)

    print(f"✅ User entity saved with ID: {user_id} (version: {user.version})")
    return FlextResult.ok(user_id)


def send_welcome_email(user: SharedUser) -> FlextResult[bool]:
    """Simulate sending welcome email using user entity."""
    log_message: TLogMessage = (
        f"📧 Sending welcome email to: {user.email_address.email}"
    )
    print(log_message)

    # Simulate email service with domain validation
    if "@invalid.com" in user.email_address.email:
        error_message: TErrorMessage = "Email service rejected invalid domain"
        return FlextResult.fail(error_message)

    print(f"✅ Welcome email sent to: {user.email_address.email}")
    email_sent = True
    return FlextResult.ok(email_sent)


def process_user_registration(data: TUserData) -> FlextResult[TAnyObject]:
    """Complete user registration pipeline using railway pattern.

    Uses shared domain entities for enhanced functionality.
    """
    log_message: TLogMessage = "\n🚀 Starting user registration pipeline..."
    print(log_message)

    # Railway pattern: chain operations with automatic error propagation
    result = (
        validate_user_data(data)
        .flat_map(create_user)
        .flat_map(
            lambda user:
            # Activate user as part of registration process
            user.activate().flat_map(
                lambda activated_user:
                # Combine database save and email sending with activated user
                FlextResult.combine(
                    save_user_to_database(activated_user),
                    send_welcome_email(activated_user),
                ).map(
                    lambda results: {
                        "user": {
                            "id": activated_user.id,
                            "name": activated_user.name,
                            "email": activated_user.email_address.email,
                            "age": activated_user.age.value,
                            "status": activated_user.status.value,
                            "created_at": (
                                str(activated_user.created_at)
                                if activated_user.created_at
                                else None
                            ),
                        },
                        "user_id": results[0]
                        if isinstance(results, list) and len(results) > 0
                        else None,
                        "email_sent": results[1]
                        if isinstance(results, list) and len(results) > 1
                        else False,
                        "status": "registered",
                        "domain_events": len(activated_user.domain_events),
                    },
                ),
            ),
        )
    )

    if result.is_success:
        print("🎉 Registration completed successfully!")
        return FlextResult.ok(result.data if isinstance(result.data, dict) else {})
    print(f"❌ Registration failed: {result.error}")
    return FlextResult.fail(result.error or "Registration failed")


def process_multiple_users(
    users_data: list[TUserData],
) -> FlextResult[TAnyObject]:
    """Process multiple users with error aggregation using types."""
    log_message: TLogMessage = f"\n📊 Processing {len(users_data)} users..."
    print(log_message)

    results: list[TAnyObject] = []
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
            error_message: TErrorMessage = f"User {i + 1} failed: {result.error}"
            print(f"⚠️  {error_message}")
            failed += 1

    summary: TAnyObject = {
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
    data: TUserData,
    max_retries: int = 3,
) -> FlextResult[TAnyObject]:
    """Process with retry logic using FlextResult and types."""
    log_message: TLogMessage = (
        f"\n🔄 Processing with retry (max {max_retries} attempts)..."
    )
    print(log_message)

    for attempt in range(1, max_retries + 1):
        print(f"   Attempt {attempt}/{max_retries}")
        result = process_user_registration(data)

        if result.is_success:
            print(f"✅ Succeeded on attempt {attempt}")
            return result

        if attempt < max_retries:
            error_message: TErrorMessage = (
                f"Attempt {attempt} failed: {result.error}, retrying..."
            )
            print(f"⚠️  {error_message}")
        else:
            print(f"❌ All {max_retries} attempts failed")
            return result.recover(
                lambda error: {
                    "status": "failed_after_retries",
                    "error": error,
                    "attempts": max_retries,
                },
            )

    return FlextResult.fail("Unexpected retry loop exit")


def transform_user_data(raw_data: str) -> FlextResult[TUserData]:
    """Transform raw JSON string to user data with validation."""
    log_message: TLogMessage = f"🔄 Transforming raw data: {raw_data[:50]}..."
    print(log_message)

    return (
        FlextResult.ok(raw_data)
        .filter(
            lambda s: isinstance(s, str) and len(s) > 0,
            "Input must be non-empty string",
        )
        .flat_map(lambda s: safe_call(lambda: json.loads(s)))
        .filter(
            lambda d: isinstance(d, dict),
            "Parsed data must be a dictionary",
        )
        .tap(lambda d: print(f"✅ Data transformed: {d}"))
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demo_successful_registration() -> None:
    """Demonstrate successful user registration."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Successful User Registration")
    print("=" * 60)

    valid_user: TUserData = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
    }

    result = process_user_registration(valid_user)
    if result.is_success:
        print(f"✅ Success: {json.dumps(result.data, indent=2)}")
    else:
        print(f"❌ Failed: {result.error}")


def demo_validation_failure() -> None:
    """Demonstrate validation failure handling."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Validation Failure")
    print("=" * 60)

    invalid_user: TUserData = {"name": "", "email": "not-an-email", "age": 15}

    result = process_user_registration(invalid_user)
    if result.is_success:
        print(f"✅ Success: {result.data}")
    else:
        print(f"❌ Expected failure: {result.error}")


def demo_batch_processing() -> None:
    """Demonstrate batch processing with error handling."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Batch Processing")
    print("=" * 60)

    users_batch: list[TUserData] = [
        {"name": "Bob Smith", "email": "bob@example.com", "age": 35},
        {"name": "Carol Davis", "email": "carol@example.com", "age": 42},
        {"name": "", "email": "invalid", "age": 16},  # This will fail
        {"name": "David Wilson", "email": "david@example.com", "age": 29},
        {"name": "Eve Brown", "email": "eve@invalid.com", "age": 33},  # This might fail
    ]

    result = process_multiple_users(users_batch)
    if result.is_success:
        print("✅ Batch processing completed!")


def demo_json_transformation() -> None:
    """Demonstrate JSON transformation pipeline."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: JSON Transformation Pipeline")
    print("=" * 60)

    json_data = '{"name": "Frank Miller", "email": "frank@example.com", "age": 45}'

    result = transform_user_data(json_data).flat_map(process_user_registration)

    if result.is_success:
        data = result.data
        if isinstance(data, dict) and "user" in data and isinstance(data["user"], dict):
            user_name = data["user"].get("name", "Unknown")
            print(f"✅ Pipeline success: User {user_name} processed")
        else:
            print("✅ Pipeline success: User processed")
    else:
        print(f"❌ Pipeline failed: {result.error}")


def demo_retry_pattern() -> None:
    """Demonstrate retry pattern with error recovery."""
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Retry Pattern")
    print("=" * 60)

    retry_user: TUserData = {
        "name": "Grace Taylor",
        "email": "grace@example.com",
        "age": 31,
    }

    result = process_with_retry(retry_user, max_retries=3)
    if result.is_success:
        print("✅ Retry pattern success!")


def main() -> None:
    """Run comprehensive FlextResult demonstration with shared domain models."""
    print("=" * 80)
    print("🚀 FLEXT RESULT - RAILWAY PATTERN DEMONSTRATION")
    print("=" * 80)

    demo_successful_registration()
    demo_validation_failure()
    demo_batch_processing()
    demo_json_transformation()
    demo_retry_pattern()

    print("\n" + "=" * 80)
    print("🎉 FLEXT RESULT DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
