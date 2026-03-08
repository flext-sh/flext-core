"""Example 03 domain models."""

from __future__ import annotations

from decimal import Decimal

from pydantic import EmailStr, Field, computed_field

from flext_core import c, m, r


class Ex03Email(m.Value):
    """Email value object model."""

    address: EmailStr


class Ex03Money(m.Value):
    """Money value object model."""

    amount: Decimal
    currency: c.Domain.Currency = c.Domain.Currency.USD

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


class Ex03Order(m.AggregateRoot):
    """Domain order aggregate."""

    customer_id: str
    items: list[Ex03OrderItem] = Field(default_factory=list)
    status: c.Domain.Status = c.Domain.Status.ACTIVE

    def add_item(self, item: Ex03OrderItem) -> r[Ex03Order]:
        """Append item to order."""
        return r[Ex03Order].ok(self.model_copy(update={"items": [*self.items, item]}))

    def confirm(self) -> r[Ex03Order]:
        """Confirm order."""
        if not self.items:
            return r[Ex03Order].fail("order has no items")
        return r[Ex03Order].ok(self)

    @computed_field
    def total_items(self) -> int:
        """Compute number of items."""
        return len(self.items)
