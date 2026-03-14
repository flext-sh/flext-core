"""Example 01 models."""

from __future__ import annotations

from flext_core import m


class Ex01User(m.Entity):
    """Result demo user model."""

    name: str
    email: str


class Ex01DemonstrationResult(m.Value):
    """Result demo summary model."""

    demonstrations_completed: int
    patterns_covered: tuple[str, ...]
    completed_at: str


class Ex01RunDemonstrationCommand(m.Command):
    """Result demo command model."""

    operation: str = "demonstration"


class Ex01ValidPersonPayload(m.Value):
    name: str
    age: int


class Ex01InvalidPersonPayload(m.Value):
    name: str
    age: str
