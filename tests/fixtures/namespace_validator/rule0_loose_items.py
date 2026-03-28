from __future__ import annotations

from enum import StrEnum, unique


class FlextTestConstants:
    VALUE = 1


MAX_RETRIES = 3


def helper() -> None:
    msg = "Must use unified test helpers per Rule 3.6"
    raise NotImplementedError(msg)


@unique
class Status(StrEnum):
    ACTIVE = "active"


class Rule0LooseItemsFixture:
    """Fixture demonstrating loose items violation."""
