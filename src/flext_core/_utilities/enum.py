"""Internal enum utilities - DO NOT IMPORT DIRECTLY.

This module provides enum utility functions following the generalized function pattern.
All functionality should be accessed via the u facade in flext_core

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from enum import StrEnum
from typing import ClassVar


class FlextUtilitiesEnum:
    """Utilities for working with StrEnum in a type-safe way."""

    _values_cache: ClassVar[MutableMapping[type[StrEnum], frozenset[str]]] = {}

    @staticmethod
    def _parse[E: StrEnum](enum_cls: type[E], value: str | E) -> E:
        if isinstance(value, enum_cls):
            return value
        if value in enum_cls.__members__:
            return enum_cls.__members__[value]
        return enum_cls(value)

    @staticmethod
    def parse_or_default[E: StrEnum](
        enum_cls: type[E],
        value: str | E | None,
        default: E,
    ) -> E:
        """Convert with fallback to default (never fails)."""
        if value is None:
            return default
        try:
            return FlextUtilitiesEnum._parse(enum_cls, value)
        except ValueError:
            return default

    @staticmethod
    def enum_values[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of values (cached for performance)."""
        if enum_cls in FlextUtilitiesEnum._values_cache:
            return FlextUtilitiesEnum._values_cache[enum_cls]
        members_dict: Mapping[str, E] = enum_cls.__members__
        result = frozenset(m.value for m in members_dict.values())
        FlextUtilitiesEnum._values_cache[enum_cls] = result
        return result


__all__: list[str] = ["FlextUtilitiesEnum"]
