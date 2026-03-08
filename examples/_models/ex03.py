"""Example 03 domain models."""

from __future__ import annotations

from decimal import Decimal

from pydantic import EmailStr, Field, computed_field, field_validator

from flext_core import c, m, r, t


class Ex03Email(m.Value):
    """Email value object model."""

    address: EmailStr


class Ex03Money(m.Value):
    """Money value object model."""

    amount: Decimal
    currency: c.Domain.Currency = c.Domain.Currency.USD

    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: t.ContainerValue) -> c.Domain.Currency:
        if isinstance(value, c.Domain.Currency):
            return value
        if isinstance(value, str):
            return c.Domain.Currency(value)
        msg = "invalid currency"
        raise TypeError(msg)

    def add(self, other: Ex03Money) -> r[Ex03Money]:
        """Add money with same currency."""
        if self.currency != other.currency:
            return r[Ex03Money].fail("currency mismatch")
        return r[Ex03Money].ok(
            Ex03Money(amount=self.amount + other.amount, currency=self.currency)
        )


class Ex03User(m.Entity):
    """Domain user entity."""

    name: str
    email: Ex03Email
    age: int


class Ex03OrderItem(m.Value):
    """Order item value object."""

    product_id: str
    name: str
    price: Ex03Money
    quantity: int = Field(ge=1)


def _new_order_items() -> list[m.ConfigMap]:
    """Create empty order-item list with explicit type."""
    return []


class Ex03Order(m.AggregateRoot):
    """Domain order aggregate."""

    customer_id: str
    items: list[m.ConfigMap] = Field(default_factory=_new_order_items)
    status: c.Domain.Status = c.Domain.Status.ACTIVE

    def add_item(self, item: Ex03OrderItem) -> r[Ex03Order]:
        """Append item to order."""
        item_payload = m.ConfigMap(root=item.model_dump())
        return r[Ex03Order].ok(
            self.model_copy(update={"items": [*self.items, item_payload]})
        )

    def confirm(self) -> r[Ex03Order]:
        """Confirm order."""
        if not self.items:
            return r[Ex03Order].fail("order has no items")
        return r[Ex03Order].ok(self)

    @computed_field
    def total_items(self) -> int:
        """Compute number of items."""
        return len(self.items)

    @computed_field
    def total_amount(self) -> float:
        """Compute aggregate order amount."""
        return float(len(self.items))
