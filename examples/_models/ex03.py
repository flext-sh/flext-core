"""Example models for ex03."""

from __future__ import annotations

from typing import Annotated

from flext_core import m, u


class Ex03Email(m.Value):
    value: Annotated[str, u.Field(description="Email address value")]


class Ex03Money(m.Value):
    amount: Annotated[float, u.Field(description="Monetary amount")]
    currency: Annotated[str, u.Field(description="ISO 4217 currency code")] = "USD"


class Ex03OrderItem(m.Value):
    sku: Annotated[str, u.Field(description="Stock-keeping unit identifier")]
    quantity: Annotated[int, u.Field(description="Number of units ordered")] = 1


class Ex03Order(m.Entity):
    status: Annotated[str, u.Field(description="Order lifecycle status")] = "active"


class Ex03User(m.Entity):
    email: Annotated[str, u.Field(description="User email address")]


class ExamplesFlextCoreModelsEx03(m):
    """Examples namespace wrapper for ex03 models."""
