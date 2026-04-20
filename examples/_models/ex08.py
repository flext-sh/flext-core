"""Example models for ex08."""

from __future__ import annotations

from flext_core import m


class Ex08User(m.Entity):
    name: str


class Ex08Order(m.Entity):
    status: str = "active"


class ExamplesFlextCoreModelsEx08(m):
    """Examples namespace wrapper for ex08 models."""
