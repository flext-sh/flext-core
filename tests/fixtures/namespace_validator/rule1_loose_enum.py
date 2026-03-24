from __future__ import annotations

from enum import StrEnum, unique


class FlextTestModels:
    pass


@unique
class Status(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class Rule1LooseEnumFixture:
    """Fixture demonstrating loose enum violation."""
