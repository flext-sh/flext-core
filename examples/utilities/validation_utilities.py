"""Validation and Type Checking Utilities.

Enterprise-grade validation functions using FlextTypes for maximum type safety
and comprehensive data validation following SOLID principles.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal

from flext_core import FlextResult, FlextTypes

# Import shared domain to reduce duplication
from shared_domain import Product as SharedProduct, Order as SharedOrder

from .formatting_helpers import MAX_DISCOUNT_PERCENTAGE

# =============================================================================
# TYPE CHECKING FUNCTIONS - Using FlextTypes.TypeGuards for safety
# =============================================================================


def is_email(value: str) -> bool:
    """Validate email format using enterprise standards."""
    return (
        isinstance(value, str)
        and "@" in value
        and "." in value.rsplit("@", maxsplit=1)[-1]
        and len(value) > 5  # Basic minimum length
    )


def is_url(value: str) -> bool:
    """Validate URL format with secure protocols."""
    return isinstance(value, str) and (value.startswith(("http://", "https://")))


def is_string(value: object) -> bool:
    """Check if value is string using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, str)


def is_int(value: object) -> bool:
    """Check if value is integer using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, int)


def is_list(value: object) -> bool:
    """Check if value is list using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, list)


def is_dict(value: object) -> bool:
    """Check if value is dict using FlextTypes.TypeGuards."""
    return FlextTypes.TypeGuards.is_instance_of(value, dict)


def is_non_empty_string(value: object) -> bool:
    """Check if value is non-empty string with whitespace handling."""
    return isinstance(value, str) and len(value.strip()) > 0


# =============================================================================
# BUSINESS LOGIC FUNCTIONS - Domain-specific validations
# =============================================================================


def calculate_discount_price(
    product: SharedProduct,
    discount_percentage: float
) -> FlextResult[Decimal]:
    """Calculate discounted price with validation."""
    if discount_percentage < 0 or discount_percentage > MAX_DISCOUNT_PERCENTAGE:
        return FlextResult.fail(
            f"Invalid discount: {discount_percentage}%. Must be 0-{MAX_DISCOUNT_PERCENTAGE}%"
        )
    
    try:
        discount_amount = product.price.amount * Decimal(str(discount_percentage / 100))
        final_price = product.price.amount - discount_amount
        return FlextResult.ok(data=final_price)
    except Exception as e:
        return FlextResult.fail(f"Price calculation failed: {e}")


def process_shared_order(order: SharedOrder) -> FlextResult[SharedOrder]:
    """Process shared domain order with validation."""
    if not order.items:
        return FlextResult.fail("Order must have at least one item")
    
    # Validate total calculation
    calculated_total = sum(
        item.quantity * item.product.price.amount 
        for item in order.items
    )
    
    if abs(calculated_total - order.total.amount) > Decimal("0.01"):
        return FlextResult.fail("Order total mismatch with items")
    
    return FlextResult.ok(data=order)