"""Internal enum utilities - DO NOT IMPORT DIRECTLY.

This module provides enum utility functions following the generalized function pattern.
All functionality should be accessed via the u facade in flext_core

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
)
from enum import StrEnum
from typing import ClassVar

from flext_core import FlextTypes as t


class FlextUtilitiesEnum:
    """Utilities for working with StrEnum in a type-safe way."""

    _values_cache: ClassVar[MutableMapping[type[StrEnum], frozenset[str]]] = {}

    @staticmethod
    def enum_values[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of values (cached for performance)."""
        if enum_cls in FlextUtilitiesEnum._values_cache:
            return FlextUtilitiesEnum._values_cache[enum_cls]
        members_dict: Mapping[str, E] = enum_cls.__members__
        result = frozenset(m.value for m in members_dict.values())
        FlextUtilitiesEnum._values_cache[enum_cls] = result
        return result


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesEnum"]
