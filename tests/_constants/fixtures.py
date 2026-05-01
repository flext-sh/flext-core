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

    MAX_VALUE: ClassVar[int] = 100
    MAX_RETRIES: ClassVar[int] = 3
    StatusLiteral = Literal["active", "inactive"]

    SAMPLE_PROJECT_NAME: ClassVar[str] = "flext-ldif"
    SAMPLE_PROJECT_VERSION: ClassVar[str] = "0.12.0-dev"
    SAMPLE_PROJECT_LICENSE: ClassVar[str] = "MIT"
    SAMPLE_PROJECT_CLASS_STEM: ClassVar[str] = "FlextLdif"
    SAMPLE_ALIAS_PARENT_SOURCE: ClassVar[str] = "flext_cli"
    SAMPLE_PROJECT_CLASS_PLATFORM: ClassVar[str] = "platform"
    SAMPLE_AUTHOR_ALICE: ClassVar[str] = "Alice Example"
    SAMPLE_AUTHOR_BOB: ClassVar[str] = "Bob Example"
    SAMPLE_PROJECT_NAME_MIGRATION: ClassVar[str] = "algar-oud-mig"
