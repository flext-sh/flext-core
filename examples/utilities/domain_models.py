"""Enhanced domain models with FLEXT mixins.

Demonstrates shared domain models enhanced with mixins
for caching, serialization, and advanced functionality.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples.shared_domain import (
    Order as SharedOrder,
    Product as SharedProduct,
    User as SharedUser,
)
from flext_core import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextUtilities,
)

from .formatting_helpers import (
    ADULT_AGE_THRESHOLD,
    MIDDLE_AGED_THRESHOLD,
    YOUNG_ADULT_AGE_THRESHOLD,
)

# =============================================================================
# ENHANCED DOMAIN MODELS - Using mixins for advanced functionality
# =============================================================================


class UtilityDemoUser(SharedUser, FlextCacheableMixin, FlextSerializableMixin):
    """Enhanced user with utility mixins for demonstrations."""

    def get_cache_key(self) -> str:
        """Get cache key for user."""
        return f"user:{self.id}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 3600  # 1 hour

    def serialize_key(self) -> str:
        """Get serialization key."""
        return f"user_data:{self.id}"

    def to_serializable(self) -> dict[str, object]:
        """Convert to serializable format with enhanced data."""
        base_data = {
            "id": self.id,
            "name": self.name,
            "email": self.email_address.email,
            "age": self.age.value,
            "status": self.status.value,
            "phone": self.phone.number if self.phone else None,
            "created_at": self.created_at,
        }
        return {
            **base_data,
            "cache_key": self.get_cache_key(),
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "status_display": self.status.value.title(),
            "age_category": self._get_age_category(),
        }

    def _get_age_category(self) -> str:
        """Get age category for analytics."""
        age_value = self.age.value
        if age_value < YOUNG_ADULT_AGE_THRESHOLD:
            return "young_adult"
        if age_value < ADULT_AGE_THRESHOLD:
            return "adult"
        if age_value < MIDDLE_AGED_THRESHOLD:
            return "middle_aged"
        return "senior"


class UtilityDemoProduct(
    SharedProduct,
    FlextCacheableMixin,
    FlextSerializableMixin,
    FlextComparableMixin,
    FlextTimestampMixin,
):
    """Enhanced product with comprehensive mixins."""

    def get_cache_key(self) -> str:
        """Get cache key for product."""
        return f"product:{self.id}"

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return 1800  # 30 minutes for product data

    def serialize_key(self) -> str:
        """Get serialization key."""
        return f"product_data:{self.id}"

    def to_serializable(self) -> dict[str, object]:
        """Convert to serializable format with enhanced data."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": {
                "amount": str(self.price.amount),
                "currency": self.price.currency,
            },
            "in_stock": self.in_stock,
            "category": self.category,
            "cache_key": self.get_cache_key(),
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "is_available": self.in_stock,
            "price_display": f"{self.price.currency} {self.price.amount}",
        }

    def compare_with(self, other: object) -> int:
        """Compare products by price for sorting."""
        if not isinstance(other, UtilityDemoProduct):
            return -1
        if self.price.amount < other.price.amount:
            return -1
        if self.price.amount > other.price.amount:
            return 1
        return 0


class UtilityDemoOrder(
    SharedOrder,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
):
    """Enhanced order with logging and tracking capabilities."""

    def get_log_context(self) -> dict[str, object]:
        """Get logging context for order operations."""
        # Calculate total from items since Order doesn't have a total attribute
        calculated_total = sum(
            item.quantity * item.unit_price.amount for item in self.items
        )

        return {
            "order_id": self.id,
            "customer_id": self.customer_id,
            "total_amount": str(calculated_total),
            "items_count": len(self.items),
            "status": self.status.value,
        }

    def serialize_key(self) -> str:
        """Get serialization key."""
        return f"order_data:{self.id}"

    def to_serializable(self) -> dict[str, object]:
        """Convert to serializable format with enhanced data."""
        # Calculate total since Order doesn't have a total attribute
        calculated_total = sum(
            item.quantity * item.unit_price.amount for item in self.items
        )
        currency = self.items[0].unit_price.currency if self.items else "USD"

        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "items": [
                {
                    "product_id": item.product_id,
                    "product_name": item.product_name,
                    "quantity": item.quantity,
                    "unit_price": str(item.unit_price.amount),
                    "line_total": str(item.quantity * item.unit_price.amount),
                }
                for item in self.items
            ],
            "total": {
                "amount": str(calculated_total),
                "currency": currency,
            },
            "status": self.status.value,
            "created_at": getattr(
                self,
                "created_at",
                FlextUtilities.generate_iso_timestamp(),
            ),
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "summary": {
                "items_count": len(self.items),
                "total_display": f"{currency} {calculated_total}",
                "status_display": self.status.value.title(),
            },
        }
