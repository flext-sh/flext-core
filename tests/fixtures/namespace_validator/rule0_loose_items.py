"""Rule 0 violation: loose constants, functions, enums outside class."""

from __future__ import annotations

from enum import StrEnum, unique


class Rule0LooseItemsFixture:
    class FlextTestConstants:
        """Main constants class."""

        VALUE = 1

    MAX_RETRIES = 3

    @staticmethod
    def helper() -> None:
        """Loose function — NOT in allowlist."""

    @unique
    class Status(StrEnum):
        """Loose class — 2nd outer class — VIOLATION."""

        ACTIVE = "active"
