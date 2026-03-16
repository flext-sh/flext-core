"""Rule 1 violation: StrEnum class outside Constants class."""

from __future__ import annotations

from enum import StrEnum, unique


class FlextTestModels:
    """Models namespace."""


@unique
class Status(StrEnum):
    """StrEnum outside Constants — VIOLATION."""

    ACTIVE = "active"
    INACTIVE = "inactive"
