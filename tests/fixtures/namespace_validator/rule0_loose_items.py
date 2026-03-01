"""Rule 0 violation: loose constants, functions, enums outside class."""

from __future__ import annotations

from enum import StrEnum


class FlextTestConstants:
    """Main constants class."""

    VALUE = 1


MAX_RETRIES = 3  # loose constant — NOT in allowlist


def helper() -> None:
    """Loose function — NOT in allowlist."""
    pass


class Status(StrEnum):
    """Loose class — 2nd outer class — VIOLATION."""

    ACTIVE = "active"
