"""Constants mixin for services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, unique
from types import MappingProxyType
from typing import Final


class TestsFlextConstantsServices:
    @unique
    class ServiceTestType(StrEnum):
        """Service test type enum for test scenarios."""

        GET_USER = "get_user"
        VALIDATE = "validate"
        FAIL = "fail"

    DEFAULT_USER_NAME_PREFIX: Final[str] = "User "
    DEFAULT_EMAIL_DOMAIN: Final[str] = "@example.com"
    DEFAULT_ERROR_MESSAGE: Final[str] = "Test error"
    USER_IDS_SUCCESS: Final[tuple[str, ...]] = ("123", "456", "789")
    USER_IDS_INVALID: Final[frozenset[str]] = frozenset({"invalid", ""})

    OPERATION_RESULT_KEY: Final[str] = "result"
    OPERATION_NAME_KEY: Final[str] = "operation"
    OPERATION_FACTORS: Final[Mapping[str, int]] = MappingProxyType({
        "double": 2,
        "negate": -1,
    })
    UNKNOWN_OPERATION_PREFIX: Final[str] = "Unknown operation:"
