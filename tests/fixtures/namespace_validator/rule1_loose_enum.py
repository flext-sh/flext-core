"""Rule 1 violation: StrEnum class outside Constants class."""

from __future__ import annotations

from enum import StrEnum


class FlextTestModels:
    """Models namespace."""


class Status(StrEnum):
    """StrEnum outside Constants — VIOLATION."""

    ACTIVE = "active"
    INACTIVE = "inactive"
