"""Validation and type checking utilities.

Validation functions using FlextTypes for type safety
and comprehensive data validation.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal

from flext_core import FlextResult

# use .shared_domain with dot to access local module
from ..shared_domain import Order as SharedOrder, Product as SharedProduct
from .formatting_helpers import MAX_DISCOUNT_PERCENTAGE

# =============================================================================
# TYPE CHECKING FUNCTIONS - Using FlextTypes.TypeGuards for safety
# =============================================================================


def is_email(value: str) -> bool:
    """Validate email format using enterprise standards."""
    min_email_length = 5  # user@a.b is minimum valid email
    return (
        isinstance(value, str)
        and "@" in value
        and "." in value.rsplit("@", maxsplit=1)[-1]
        and len(value) > min_email_length
    )


def is_url(value: str) -> bool:
    """Validate URL format with secure protocols."""
    return isinstance(value, str) and (value.startswith(("http://", "https://")))


def is_string(value: object) -> bool:
    """Check if value is string."""
    return isinstance(value, str)


def is_int(value: object) -> bool:
    """Check if value is integer."""
    return isinstance(value, int)


def is_list(value: object) -> bool:
    """Check if value is list."""
    return isinstance(value, list)


def is_dict(value: object) -> bool:
    """Check if value is dict."""
    return isinstance(value, dict)


def is_non_empty_string(value: object) -> bool:
    """Check if value is non-empty string with whitespace handling."""
    return isinstance(value, str) and len(value.strip()) > 0


# =============================================================================
# BUSINESS LOGIC FUNCTIONS - Domain-specific validations
# =============================================================================


def calculate_discount_price(
    product: SharedProduct,
    discount_percentage: float,
) -> FlextResult[Decimal]:
    """Calculate discounted price with validation."""
    if discount_percentage < 0 or discount_percentage > MAX_DISCOUNT_PERCENTAGE:
        return FlextResult[Decimal].fail(
            f"Invalid discount: {discount_percentage}%. "
            f"Must be 0-{MAX_DISCOUNT_PERCENTAGE}%",
        )

    try:
        discount_amount = product.price.amount * Decimal(str(discount_percentage / 100))
        final_price = product.price.amount - discount_amount
        return FlextResult[Decimal].ok(final_price)
    except Exception as e:
        return FlextResult[Decimal].fail(f"Price calculation failed: {e}")


def process_shared_order(order: SharedOrder) -> FlextResult[SharedOrder]:
    """Process shared domain order with validation."""
    if not order.items:
        return FlextResult[SharedOrder].fail("Order must have at least one item")

    # Validate total calculation
    calculated_total = sum(
        item.quantity * item.unit_price.amount for item in order.items
    )

    # Order doesn't have a total attribute, so we calculate the total dynamically
    # This is a placeholder validation that could be extended
    if calculated_total <= 0:
        return FlextResult[SharedOrder].fail("Order total must be positive")

    return FlextResult[SharedOrder].ok(order)
