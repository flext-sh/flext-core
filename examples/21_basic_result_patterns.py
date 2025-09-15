#!/usr/bin/env python3
"""Basic Result Pattern Composition with Python 3.13 Features.

Demonstrates the CURRENT capabilities of FlextResult:
- Basic success/failure wrapping
- Safe data extraction with unwrap()
- Error handling and propagation
- Type-safe result processing
- Integration with domain models

Note: Advanced monadic operators and category theory patterns
are planned for future versions but not yet implemented.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from flext_core import (
    FlextLogger,
    FlextResult,
    FlextTypes,
)

# Configure logging
logger = FlextLogger(__name__)


# === DOMAIN MODELS FOR DEMONSTRATION ===


class User:
    """User entity for result pattern demonstration."""

    def __init__(self, user_id: str, name: str, email: str, age: int) -> None:
        """Initialize User with basic information."""
        self.user_id = user_id
        self.name = name
        self.email = email
        self.age = age

    def __repr__(self) -> str:
        """Representation of User."""
        return f"User(id={self.user_id}, name={self.name}, email={self.email}, age={self.age})"


class Order:
    """Order entity for result pattern demonstration."""

    def __init__(
        self,
        order_id: str,
        user_id: str,
        amount: Decimal,
        items: FlextTypes.Core.StringList,
    ) -> None:
        """Initialize Order with details."""
        self.order_id = order_id
        self.user_id = user_id
        self.amount = amount
        self.items = items

    def __repr__(self) -> str:
        """Representation of Order."""
        return f"Order(id={self.order_id}, user={self.user_id}, amount={self.amount}, items={len(self.items)})"


class ProcessingResult:
    """Final processing result for demonstration."""

    def __init__(self, user: User, order: Order, processed_at: str) -> None:
        """Initialize ProcessingResult with processed data."""
        self.user = user
        self.order = order
        self.processed_at = processed_at

    def __repr__(self) -> str:
        """Representation of ProcessingResult."""
        return f"ProcessingResult(user={self.user.name}, order={self.order.order_id}, at={self.processed_at})"


# === BASIC RESULT PATTERN DEMONSTRATIONS ===


def validate_user_data(data: FlextTypes.Core.Dict) -> FlextResult[FlextTypes.Core.Dict]:
    """Validate user data with FlextResult pattern."""
    logger.info("🔍 Validating user data")

    required_fields = ["user_id", "name", "email", "age"]
    for field in required_fields:
        if field not in data or not data[field]:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"Missing required field: {field}"
            )

    # Basic email validation
    email = cast("str", data["email"])
    if "@" not in email or "." not in email:
        return FlextResult[FlextTypes.Core.Dict].fail("Invalid email format")

    # Age validation
    try:
        age = int(data["age"])
        if age < 0 or age > 150:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "Age must be between 0 and 150"
            )
    except (ValueError, TypeError):
        return FlextResult[FlextTypes.Core.Dict].fail("Age must be a valid integer")

    logger.info("✅ User data validation passed")
    return FlextResult[FlextTypes.Core.Dict].ok(data)


def create_user(data: FlextTypes.Core.Dict) -> FlextResult[User]:
    """Create User from validated data."""
    logger.info("👤 Creating user from data")

    try:
        user = User(
            user_id=cast("str", data["user_id"]),
            name=cast("str", data["name"]),
            email=cast("str", data["email"]),
            age=int(data["age"]),
        )
        logger.info(f"✅ User created: {user}")
        return FlextResult[User].ok(user)
    except Exception as e:
        return FlextResult[User].fail(f"Failed to create user: {e}")


def validate_order_data(
    data: FlextTypes.Core.Dict,
) -> FlextResult[FlextTypes.Core.Dict]:
    """Validate order data."""
    logger.info("📦 Validating order data")

    if "order_amount" not in data:
        return FlextResult[FlextTypes.Core.Dict].fail("Missing order amount")

    if "items" not in data or not data["items"]:
        return FlextResult[FlextTypes.Core.Dict].fail("Order must have items")

    # Validate amount
    try:
        amount = Decimal(str(data["order_amount"]))
        if amount <= 0:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "Order amount must be positive"
            )
    except Exception:
        return FlextResult[FlextTypes.Core.Dict].fail("Invalid order amount format")

    logger.info("✅ Order data validation passed")
    return FlextResult[FlextTypes.Core.Dict].ok(data)


def create_order(user: User, data: FlextTypes.Core.Dict) -> FlextResult[Order]:
    """Create Order from validated data."""
    logger.info("🛒 Creating order from data")

    try:
        order = Order(
            order_id=f"ord_{user.user_id}_{hash(str(data)) % 10000:04d}",
            user_id=user.user_id,
            amount=Decimal(str(data["order_amount"])),
            items=cast("FlextTypes.Core.StringList", data["items"]),
        )
        logger.info(f"✅ Order created: {order}")
        return FlextResult[Order].ok(order)
    except Exception as e:
        return FlextResult[Order].fail(f"Failed to create order: {e}")


def process_user_order(user: User, order: Order) -> FlextResult[ProcessingResult]:
    """Process user order and create final result."""
    logger.info("⚡ Processing user order")

    try:
        result = ProcessingResult(
            user=user, order=order, processed_at=datetime.now(UTC).isoformat()
        )
        logger.info(f"✅ Processing completed: {result}")
        return FlextResult[ProcessingResult].ok(result)
    except Exception as e:
        return FlextResult[ProcessingResult].fail(f"Failed to process order: {e}")


def demonstrate_basic_result_chaining() -> FlextResult[str]:
    """Demonstrate basic result chaining without advanced operators."""
    logger.info("🔗 Demonstrating basic result chaining")

    # Test data
    test_data = {
        "user_id": "user_001",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "age": 30,
        "order_amount": "150.50",
        "items": ["laptop", "mouse", "keyboard"],
    }

    # Step-by-step processing with explicit error handling
    user_validation_result = validate_user_data(test_data)
    if user_validation_result.is_failure:
        return FlextResult[str].fail(
            f"User validation failed: {user_validation_result.error}"
        )

    user_data = user_validation_result.unwrap()
    user_creation_result = create_user(user_data)
    if user_creation_result.is_failure:
        return FlextResult[str].fail(
            f"User creation failed: {user_creation_result.error}"
        )

    user = user_creation_result.unwrap()
    order_validation_result = validate_order_data(test_data)
    if order_validation_result.is_failure:
        return FlextResult[str].fail(
            f"Order validation failed: {order_validation_result.error}"
        )

    order_data = order_validation_result.unwrap()
    order_creation_result = create_order(user, order_data)
    if order_creation_result.is_failure:
        return FlextResult[str].fail(
            f"Order creation failed: {order_creation_result.error}"
        )

    order = order_creation_result.unwrap()
    processing_result = process_user_order(user, order)
    if processing_result.is_failure:
        return FlextResult[str].fail(f"Processing failed: {processing_result.error}")

    final_result = processing_result.unwrap()
    logger.info(f"🎉 Complete processing successful: {final_result}")
    return FlextResult[str].ok(
        "Basic result chaining demonstration completed successfully"
    )


def demonstrate_error_handling() -> FlextResult[str]:
    """Demonstrate error handling with invalid data."""
    logger.info("❌ Demonstrating error handling with invalid data")

    # Invalid test data
    invalid_data = {
        "user_id": "",  # Invalid: empty user_id
        "name": "Bob",
        "email": "invalid-email",  # Invalid: no @ or .
        "age": -5,  # Invalid: negative age
    }

    result = validate_user_data(invalid_data)
    if result.is_failure:
        error_msg = result.error or "Unknown validation error"
        logger.info(f"✅ Error correctly caught: {error_msg}")
        return FlextResult[str].ok(
            f"Error handling demonstration successful: {error_msg}"
        )

    return FlextResult[str].fail("Expected validation to fail, but it succeeded")


def main() -> None:
    """Run all result pattern demonstrations."""
    logger.info("🚀 Starting Basic Result Pattern Demonstrations")
    print("\\n" + "=" * 80)
    print("FLEXT Basic Result Pattern Examples")
    print("=" * 80)

    # Demonstration 1: Basic chaining
    print("\\n1. Basic Result Chaining:")
    print("-" * 40)
    result1 = demonstrate_basic_result_chaining()
    if result1.is_success:
        print(f"✅ SUCCESS: {result1.unwrap()}")
    else:
        print(f"❌ FAILED: {result1.error}")

    # Demonstration 2: Error handling
    print("\\n2. Error Handling:")
    print("-" * 40)
    result2 = demonstrate_error_handling()
    if result2.is_success:
        print(f"✅ SUCCESS: {result2.unwrap()}")
    else:
        print(f"❌ FAILED: {result2.error}")

    print("\\n" + "=" * 80)
    print("All demonstrations completed!")
    print("\\nNote: Advanced monadic operators (>>, <<, @, etc.) are planned")
    print("for future versions but not yet implemented in FlextResult.")
    print("=" * 80)


if __name__ == "__main__":
    main()
