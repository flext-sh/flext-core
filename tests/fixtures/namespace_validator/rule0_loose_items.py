from __future__ import annotations

from enum import StrEnum, unique


class FlextTestConstants:
    VALUE = 1


MAX_RETRIES = 3


def helper() -> None:
    pass


@unique
class Status(StrEnum):
    ACTIVE = "active"


class Rule0LooseItemsFixture:
    """Fixture demonstrating loose items violation."""

    pass
