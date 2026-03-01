"""Rule 0 violation: multiple outer classes (expected 1)."""

from __future__ import annotations


class FlextTestConstants:
    """First outer class."""

    VALUE = 1


class FlextTestModels:
    """Second outer class — VIOLATION."""

    NAME = "test"
