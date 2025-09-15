#!/usr/bin/env python3
"""Advanced Monadic Composition with Python 3.13 Features.

Demonstrates the new advanced monadic operators implemented in FlextResult:
- >> (rshift) for monadic bind (flat_map)
- << (lshift) for functor map
- @ (matmul) for applicative combination
- / (truediv) for alternative fallback
- % (mod) for conditional filtering
- & (and) for sequential composition
- ^ (xor) for error recovery
- traverse and kleisli_compose for category theory patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
import sys
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from flext_core import (
    FlextLogger,
    FlextResult,
)

# Configure logging
logger = FlextLogger(__name__)


# === DOMAIN MODELS FOR DEMONSTRATION ===


class User:
    """User entity for monadic composition demonstration."""

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
    """Order entity for composition demonstration."""

    def __init__(
        self,
        order_id: str,
        user_id: str,
        amount: Decimal,
        items: list[str],
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


# === MONADIC COMPOSITION DEMONSTRATIONS ===


def demonstrate_rshift_operator() -> FlextResult[str]:
    """Demonstrate >> operator for monadic bind (flat_map)."""
    logger.info("🚀 Demonstrating >> operator for monadic bind")

    # Traditional approach with flat_map
    def traditional_approach(data: dict[str, object]) -> FlextResult[ProcessingResult]:
        return (
            validate_user_data(data)
            .flat_map(create_user)
            .flat_map(
                lambda user: validate_order_data(data).flat_map(
                    lambda order_data: create_order(user, order_data),
                ),
            )
            .flat_map(process_user_order)
        )

    # New approach with >> operator
    def monadic_approach(data: dict[str, object]) -> FlextResult[ProcessingResult]:
        return (
            validate_user_data(data)
            >> create_user
            >> (
                lambda user: validate_order_data(data)
                >> (lambda order_data: create_order(user, order_data))
            )
            >> process_user_order
        )

    # Test data
    test_data = {
        "user_id": "user_001",
        "name": "Alice Smith",
        "email": "alice@example.com",
        "age": 30,
        "order_amount": "150.50",
        "items": ["laptop", "mouse", "keyboard"],
    }

    # Execute both approaches
    traditional_result = traditional_approach(test_data)
    monadic_result = monadic_approach(test_data)

    if traditional_result.is_success and monadic_result.is_success:
        logger.info("✅ Both traditional and monadic approaches succeeded")
        return FlextResult[str].ok(">> operator demonstration completed successfully")
    error = traditional_result.error or monadic_result.error or "Unknown error"
    return FlextResult[str].fail(f"Demonstration failed: {error}")


def demonstrate_lshift_operator() -> FlextResult[str]:
    """Demonstrate << operator for functor map."""
    logger.info("🔄 Demonstrating << operator for functor map")

    # Chain multiple transformations using << operator
    result = (
        FlextResult[int].ok(100)
        << (lambda x: x * 2)  # Transform to 200
        << (lambda x: x + 50)  # Transform to 250
        << (lambda x: f"${x:.2f}")  # Transform to "$250.00"
        << str.upper  # Transform to "$250.00"
    )

    if result.is_success:
        logger.info(f"✅ Transformation chain result: {result.unwrap()}")
        return FlextResult[str].ok("<< operator demonstration completed successfully")
    return FlextResult[str].fail(f"Transformation failed: {result.error}")


def demonstrate_matmul_operator() -> FlextResult[str]:
    """Demonstrate @ operator for applicative combination."""
    logger.info("🔗 Demonstrating @ operator for applicative combination")

    # Parallel validation using @ operator
    name_validation = validate_name("Alice Smith")
    email_validation = validate_email("alice@example.com")
    age_validation = validate_age(30)

    # Combine pairs first, then with third (@ operator works with tuples)
    name_email_result = name_validation @ email_validation

    if name_email_result.is_success:
        (name, email) = name_email_result.unwrap()
        logger.info(f"✅ First two validations passed: name={name}, email={email}")

        # Now validate age separately
        if age_validation.is_success:
            age = age_validation.unwrap()
            logger.info(
                f"✅ All validations passed: name={name}, email={email}, age={age}",
            )
            return FlextResult[str].ok(
                "@ operator demonstration completed successfully",
            )
        return FlextResult[str].fail(f"Age validation failed: {age_validation.error}")
    return FlextResult[str].fail(
        f"Name/Email validation failed: {name_email_result.error}",
    )


def demonstrate_truediv_operator() -> FlextResult[str]:
    """Demonstrate / operator for alternative fallback."""
    logger.info("🔀 Demonstrating / operator for alternative fallback")

    # Try multiple data sources with fallback
    primary_data = FlextResult[str].fail("Primary database unavailable")
    backup_data = FlextResult[str].fail("Backup database unavailable")
    cache_data = FlextResult[str].ok("Cached user data: Alice Smith")
    default_data = FlextResult[str].ok("Default anonymous user")

    # Chain fallbacks using / operator
    final_result = primary_data / backup_data / cache_data / default_data

    if final_result.is_success:
        logger.info(f"✅ Fallback succeeded with: {final_result.unwrap()}")
        return FlextResult[str].ok("/ operator demonstration completed successfully")
    return FlextResult[str].fail("All fallbacks failed")


def demonstrate_mod_operator() -> FlextResult[str]:
    """Demonstrate % operator for conditional filtering."""
    logger.info("🔍 Demonstrating % operator for conditional filtering")

    # Chain validations using % operator
    result = (
        FlextResult[int].ok(42)
        % (lambda x: x > 0)  # Must be positive
        % (lambda x: x < 100)  # Must be less than 100
        % (lambda x: x % 2 == 0)  # Must be even
    )

    if result.is_success:
        logger.info(f"✅ All filters passed for value: {result.unwrap()}")
        return FlextResult[str].ok("% operator demonstration completed successfully")
    return FlextResult[str].fail(f"Filter validation failed: {result.error}")


def demonstrate_traverse_operation() -> FlextResult[str]:
    """Demonstrate traverse operation from Category Theory."""
    logger.info("🔄 Demonstrating traverse operation")

    # Process list of items with traverse
    items = ["apple", "banana", "cherry", "date"]

    def process_item(item: str) -> FlextResult[str]:
        if len(item) > 6:  # Simulate failure for long names
            return FlextResult[str].fail(f"Item name too long: {item}")
        return FlextResult[str].ok(f"processed_{item}")

    # Traverse processes all items, failing at first error
    traverse_result = FlextResult.traverse(items, process_item)

    if traverse_result.is_success:
        processed_items = traverse_result.unwrap()
        logger.info(f"✅ All items processed: {processed_items}")
        return FlextResult[str].ok("traverse operation completed successfully")
    return FlextResult[str].fail(f"Traverse failed: {traverse_result.error}")


def demonstrate_applicative_lift() -> FlextResult[str]:
    """Demonstrate applicative lift operations with proper tuple handling."""
    logger.info("⬆️ Demonstrating applicative lift operations")

    # Binary lift - combine two independent results
    result1 = FlextResult[int].ok(10)
    result2 = FlextResult[int].ok(20)

    binary_result = FlextResult.applicative_lift2(
        operator.mul,  # Multiply function
        result1,
        result2,
    )

    # Ternary lift - combine three independent results
    result3 = FlextResult[int].ok(30)

    ternary_result = FlextResult.applicative_lift3(
        lambda x, y, z: x + y + z,  # Sum function
        result1,
        result2,
        result3,
    )

    # Also demonstrate @ operator for binary combination
    matmul_result = result1 @ result2

    if (
        binary_result.is_success
        and ternary_result.is_success
        and matmul_result.is_success
    ):
        logger.info(f"✅ Binary lift result: {binary_result.unwrap()}")
        logger.info(f"✅ Ternary lift result: {ternary_result.unwrap()}")

        # @ operator creates tuple
        (val1, val2) = matmul_result.unwrap()
        logger.info(f"✅ @ operator result: ({val1}, {val2})")

        return FlextResult[str].ok(
            "applicative lift demonstration completed successfully",
        )
    error = (
        binary_result.error
        or ternary_result.error
        or matmul_result.error
        or "Lift failed"
    )
    return FlextResult[str].fail(f"Applicative lift failed: {error}")


# === HELPER FUNCTIONS ===


def validate_user_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
    """Validate user data dictionary."""
    required_fields = ["user_id", "name", "email", "age"]
    for field in required_fields:
        if field not in data:
            return FlextResult[dict[str, object]].fail(
                f"Missing required field: {field}",
            )
    return FlextResult[dict[str, object]].ok(data)


def create_user(data: dict[str, object]) -> FlextResult[User]:
    """Create user from validated data."""
    try:
        user = User(
            user_id=str(data["user_id"]),
            name=str(data["name"]),
            email=str(data["email"]),
            age=int(cast("int | str", data["age"])),
        )
        return FlextResult[User].ok(user)
    except Exception as e:
        return FlextResult[User].fail(f"User creation failed: {e}")


def validate_order_data(data: dict[str, object]) -> FlextResult[dict[str, object]]:
    """Validate order data dictionary."""
    if "order_amount" not in data or "items" not in data:
        return FlextResult[dict[str, object]].fail("Missing order data")
    return FlextResult[dict[str, object]].ok(data)


def create_order(user: User, order_data: dict[str, object]) -> FlextResult[Order]:
    """Create order for user."""
    try:
        order = Order(
            order_id=f"order_{len(user.user_id)}",
            user_id=user.user_id,
            amount=Decimal(str(order_data["order_amount"])),
            items=list(cast("list[str]", order_data["items"])),
        )
        return FlextResult[Order].ok(order)
    except Exception as e:
        return FlextResult[Order].fail(f"Order creation failed: {e}")


def process_user_order(order: Order) -> FlextResult[ProcessingResult]:
    """Process user order."""
    result = ProcessingResult(
        user=User(order.user_id, "Processed User", "processed@example.com", 0),
        order=order,
        processed_at=datetime.now(UTC).isoformat(),
    )
    return FlextResult[ProcessingResult].ok(result)


def validate_name(name: str) -> FlextResult[str]:
    """Validate name field."""
    if len(name.strip()) < 2:
        return FlextResult[str].fail("Name too short")
    return FlextResult[str].ok(name)


def validate_email(email: str) -> FlextResult[str]:
    """Validate email field."""
    if "@" not in email:
        return FlextResult[str].fail("Invalid email format")
    return FlextResult[str].ok(email)


def validate_age(age: int) -> FlextResult[int]:
    """Validate age field."""
    if age < 18 or age > 120:
        return FlextResult[int].fail("Age out of valid range")
    return FlextResult[int].ok(age)


# === MAIN DEMONSTRATION ===


def main() -> FlextResult[str]:
    """Main demonstration of advanced monadic composition."""
    logger.info("🎯 Starting Advanced Monadic Composition Demonstration")

    # Execute all demonstrations
    demonstrations = [
        (">> Operator (Monadic Bind)", demonstrate_rshift_operator),
        ("<< Operator (Functor Map)", demonstrate_lshift_operator),
        ("@ Operator (Applicative Combination)", demonstrate_matmul_operator),
        ("/ Operator (Alternative Fallback)", demonstrate_truediv_operator),
        ("% Operator (Conditional Filtering)", demonstrate_mod_operator),
        ("Traverse Operation", demonstrate_traverse_operation),
        ("Applicative Lift", demonstrate_applicative_lift),
    ]

    results = []
    for name, demo_func in demonstrations:
        logger.info(f"\n--- {name} ---")
        result = demo_func()
        if result.is_success:
            logger.info(f"✅ {name}: {result.unwrap()}")
            results.append(result.unwrap())
        else:
            logger.error(f"❌ {name}: {result.error}")
            return FlextResult[str].fail(f"{name} failed: {result.error}")

    summary = (
        f"🎉 Advanced Monadic Composition Demonstration completed successfully!\n"
        f"📊 Executed {len(results)} demonstrations with 100% success rate.\n"
        f"🚀 New operators enable mathematical notation for functional composition:\n"
        f"   >> for monadic bind, << for functor map, @ for applicative combination,\n"
        f"   / for fallback, % for filtering, & for sequential, ^ for recovery.\n"
        f"🧮 Category theory operations: traverse, kleisli_compose, applicative_lift."
    )

    logger.info(summary)
    return FlextResult[str].ok(summary)


if __name__ == "__main__":
    result = main()
    if result.is_failure:
        logger.error(f"Program failed: {result.error}")
        sys.exit(1)
