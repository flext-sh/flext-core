"""Constants mixin for domain.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from enum import StrEnum, unique
from typing import ClassVar

from flext_core import c


class TestsFlextConstantsDomain:
    """Flat domain/path constants for flext-core tests."""

    REPO_ROOT_PARENT_DEPTH: ClassVar[int] = 2
    SRC_DIR: ClassVar[str] = "src"
    CORE_PACKAGE_DIR: ClassVar[str] = "flext_core"
    EXAMPLES_DIR: ClassVar[str] = "examples"
    PYPROJECT_FILENAME: ClassVar[str] = c.PYPROJECT_FILENAME

    @unique
    class StatusEnum(StrEnum):
        """Reusable test status enum for test fixtures.

        Standard three-state status enum used across multiple test modules.
        """

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    STATUS_ENUM: ClassVar[type[StatusEnum]] = StatusEnum
    STATUS_ACTIVE: ClassVar[StatusEnum] = StatusEnum.ACTIVE
    STATUS_PENDING: ClassVar[StatusEnum] = StatusEnum.PENDING
    STATUS_INACTIVE: ClassVar[StatusEnum] = StatusEnum.INACTIVE


__all__: list[str] = ["TestsFlextConstantsDomain"]
