"""Constants mixin for services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core import t


class TestsFlextConstantsServices:
    @unique
    class ServiceTestType(StrEnum):
        """Service test type enum for test scenarios."""

        GET_USER = "get_user"
        VALIDATE = "validate"
        FAIL = "fail"

    type ServiceType = ServiceTestType

    SERVICE_TEST_TYPE_GET_USER: ClassVar[ServiceType] = ServiceTestType.GET_USER
    SERVICE_TEST_TYPE_VALIDATE: ClassVar[ServiceType] = ServiceTestType.VALIDATE
    SERVICE_TEST_TYPE_FAIL: ClassVar[ServiceType] = ServiceTestType.FAIL

    DEFAULT_USER_NAME_PREFIX: ClassVar[str] = "User "
    DEFAULT_EMAIL_DOMAIN: ClassVar[str] = "@example.com"
    DEFAULT_ERROR_MESSAGE: ClassVar[str] = "Test error"
    USER_IDS_SUCCESS: ClassVar[t.StrSequence] = ("123", "456", "789")
    USER_IDS_INVALID: ClassVar[frozenset[str]] = frozenset({"invalid", ""})

    OPERATION_RESULT_KEY: ClassVar[str] = "result"
    OPERATION_NAME_KEY: ClassVar[str] = "operation"
    OPERATION_FACTORS: ClassVar[Mapping[str, int]] = MappingProxyType({
        "double": 2,
        "negate": -1,
    })
    UNKNOWN_OPERATION_PREFIX: ClassVar[str] = "Unknown operation:"


__all__: list[str] = ["TestsFlextConstantsServices"]
