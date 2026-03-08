"""Example 10 handlers models."""

from __future__ import annotations

from flext_core import m


class Ex10Message(m.Command):
    """Command message payload."""

    text: str


class Ex10DerivedMessage(Ex10Message):
    """Derived message used for covariance checks."""


class Ex10Entity(m.Value):
    """Entity-like value payload for runtime checks."""

    unique_id: str


class Ex10ProcessorGood(m.Value):
    """Good processor for protocol checks."""

    marker: str = "good"

    def process(self) -> str:
        """Process successfully."""
        return "ok"


class Ex10ProcessorBad(m.Value):
    """Bad processor for protocol checks."""

    marker: str = "bad"

    def process(self) -> str:
        """Process successfully despite bad protocol metadata."""
        return "ok"
