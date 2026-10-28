"""Constants mixin for test enum declarations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
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


class TestsFlextConstantsEnums:
    """String enumerations owned by the tests constants family."""

    @unique
    class Color(StrEnum):
        """Color discriminator for adapter contract scenarios."""

        RED = "red"


__all__: list[str] = ["TestsFlextConstantsEnums"]
