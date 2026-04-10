"""Constants mixin for services.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final, Literal


class TestsFlextCoreConstantsServices:
    @unique
    class ServiceTestType(StrEnum):
        """Service test type enum for test scenarios."""

        GET_USER = "get_user"
        VALIDATE = "validate"
        FAIL = "fail"

    class Services:
        """Flext-core-specific service-related constants."""

        DEFAULT_USER_NAME_PREFIX: Final[str] = "User "
        DEFAULT_EMAIL_DOMAIN: Final[str] = "@example.com"
        DEFAULT_ERROR_MESSAGE: Final[str] = "Test error"

    class HTTP:
        """Flext-core-specific HTTP-related constants for testing."""

        @unique
        class Method(StrEnum):
            """HTTP methods for testing."""

            GET = "GET"
            POST = "POST"

        type StatusLiteral = Literal[200, 404, 400]
        type MethodLiteral = Literal["GET", "POST"]
        STATUS_OK: Final[int] = 200
        STATUS_NOT_FOUND: Final[int] = 404
        STATUS_BAD_REQUEST: Final[int] = 400
        CONTENT_TYPE_JSON: Final[str] = "application/json"
        METHOD_GET: Final[str] = Method.GET
        METHOD_POST: Final[str] = Method.POST
