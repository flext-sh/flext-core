"""Constants mixin for domain.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class TestsFlextConstantsDomain:
    """Flat domain/path constants for flext-core tests."""

    REPO_ROOT_PARENT_DEPTH: Final[int] = 2
    SRC_DIR: Final[str] = "src"
    CORE_PACKAGE_DIR: Final[str] = "flext_core"
    EXAMPLES_DIR: Final[str] = "examples"
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"

    @unique
    class StatusEnum(StrEnum):
        """Reusable test status enum for test fixtures.

        Standard three-state status enum used across multiple test modules.
        """

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    STATUS_ENUM: Final[type[StatusEnum]] = StatusEnum
    STATUS_ACTIVE: Final[StatusEnum] = StatusEnum.ACTIVE
    STATUS_PENDING: Final[StatusEnum] = StatusEnum.PENDING
    STATUS_INACTIVE: Final[StatusEnum] = StatusEnum.INACTIVE
