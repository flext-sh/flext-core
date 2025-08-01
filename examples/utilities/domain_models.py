"""Enhanced Domain Models with FLEXT Mixins.

Demonstrates enterprise patterns using shared domain models enhanced with
FLEXT mixins for caching, serialization, and advanced functionality.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextCacheableMixin,
    FlextComparableMixin,
    FlextLoggableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextUtilities,
    TAnyObject,
)

# Import shared domain models to reduce duplication
from shared_domain import (
    Money,
    Order as SharedOrder,
    OrderStatus,
    Product as SharedProduct,
    User as SharedUser,
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

    def to_serializable(self) -> TAnyObject:
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
        elif age_value < ADULT_AGE_THRESHOLD:
            return "adult"
        elif age_value < MIDDLE_AGED_THRESHOLD:
            return "middle_aged"
        else:
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

    def to_serializable(self) -> TAnyObject:
        """Convert to serializable format with enhanced data."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": {
                "amount": str(self.price.amount),
                "currency": self.price.currency,
            },
            "stock_quantity": self.stock_quantity,
            "category": self.category,
            "sku": self.sku,
            "cache_key": self.get_cache_key(),
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "is_available": self.stock_quantity > 0,
            "price_display": f"{self.price.currency} {self.price.amount}",
        }

    def compare_with(self, other: object) -> int:
        """Compare products by price for sorting."""
        if not isinstance(other, UtilityDemoProduct):
            return -1
        if self.price.amount < other.price.amount:
            return -1
        elif self.price.amount > other.price.amount:
            return 1
        else:
            return 0


class UtilityDemoOrder(
    SharedOrder, FlextLoggableMixin, FlextSerializableMixin, FlextTimestampMixin
):
    """Enhanced order with logging and tracking capabilities."""

    def get_log_context(self) -> dict[str, object]:
        """Get logging context for order operations."""
        return {
            "order_id": self.id,
            "customer_id": self.customer.id,
            "total_amount": str(self.total.amount),
            "items_count": len(self.items),
            "status": self.status.value,
        }

    def serialize_key(self) -> str:
        """Get serialization key."""
        return f"order_data:{self.id}"

    def to_serializable(self) -> TAnyObject:
        """Convert to serializable format with enhanced data."""
        return {
            "id": self.id,
            "customer": {
                "id": self.customer.id,
                "name": self.customer.name,
                "email": self.customer.email_address.email,
            },
            "items": [
                {
                    "product_id": item.product.id,
                    "product_name": item.product.name,
                    "quantity": item.quantity,
                    "unit_price": str(item.product.price.amount),
                    "line_total": str(item.quantity * item.product.price.amount),
                }
                for item in self.items
            ],
            "total": {
                "amount": str(self.total.amount),
                "currency": self.total.currency,
            },
            "status": self.status.value,
            "created_at": self.created_at,
            "serialized_at": FlextUtilities.generate_iso_timestamp(),
            "summary": {
                "items_count": len(self.items),
                "total_display": f"{self.total.currency} {self.total.amount}",
                "status_display": self.status.value.title(),
            },
        }
