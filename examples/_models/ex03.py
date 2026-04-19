"""Example models for ex03."""

from __future__ import annotations

from examples import m


class Ex03Email(m.Value):
    value: str


class Ex03Money(m.Value):
    amount: float
    currency: str = "USD"


class Ex03OrderItem(m.Value):
    sku: str
    quantity: int = 1


class Ex03Order(m.Entity):
    status: str = "active"


class Ex03User(m.Entity):
    email: str


class ExamplesFlextCoreModelsEx03(m):
    """Examples namespace wrapper for ex03 models."""
