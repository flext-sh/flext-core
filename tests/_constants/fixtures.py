"""Constants mixin for fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, Literal


class TestsFlextConstantsFixtures:
    @unique
    class Status(StrEnum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    VALID_STATUSES: ClassVar[frozenset[Status]] = frozenset(Status)
    DEFAULT_HEADERS: ClassVar[dict[str, str]] = {"content_type": "json"}
    MAX_VALUE: ClassVar[int] = 100
    MAX_RETRIES: ClassVar[int] = 3
    StatusLiteral = Literal["active", "inactive"]
