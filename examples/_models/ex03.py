"""Example models for ex03."""

from __future__ import annotations

from typing import Annotated

from flext_core import m


class ExamplesFlextModelsEx03:
    """Examples namespace wrapper for ex03 models."""

    class Email(m.Value):
        value: Annotated[str, m.Field(description="Email address value")]

    class Money(m.Value):
        amount: Annotated[float, m.Field(description="Monetary amount")]
        currency: Annotated[str, m.Field(description="ISO 4217 currency code")] = "USD"

    class OrderItem(m.Value):
        sku: Annotated[str, m.Field(description="Stock-keeping unit identifier")]
        quantity: Annotated[int, m.Field(description="Number of units ordered")] = 1

    class Order(m.Entity):
        status: Annotated[str, m.Field(description="Order lifecycle status")] = "active"

    class EmailUser(m.Entity):
        email: Annotated[str, m.Field(description="User email address")]
