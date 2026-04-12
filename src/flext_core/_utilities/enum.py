"""Internal enum utilities - DO NOT IMPORT DIRECTLY.

This module provides enum utility functions following the generalized function pattern.
All functionality should be accessed via the u facade in flext_core.utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from enum import StrEnum
from typing import ClassVar

from flext_core import EnumT, c, r, t


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
    def _is_member_by_name[E: StrEnum](enum_cls: type[E], value: str) -> bool:
        return value in enum_cls.__members__

    @staticmethod
    def _is_member_by_value[E: StrEnum](enum_cls: type[E], value: str | E) -> bool:
        if isinstance(value, enum_cls):
            return True
        return value in FlextUtilitiesEnum.enum_values(enum_cls)

    @staticmethod
    def coerce_validator[E: StrEnum](enum_cls: type[E]) -> Callable[[t.Scalar | E], E]:
        """Create BeforeValidator for automatic coercion in Pydantic."""

        def _coerce(value: t.Scalar | E) -> E:
            if isinstance(value, (enum_cls, str)):
                try:
                    return FlextUtilitiesEnum._parse(enum_cls, value)
                except ValueError:
                    pass
            raise ValueError(
                c.ERR_ENUM_INVALID_VALUE.format(
                    enum_name=enum_cls.__name__,
                    value=value,
                ),
            )

        return _coerce

    @staticmethod
    def parse_enum(enum_cls: type[EnumT], value: str | EnumT) -> r[EnumT]:
        """Convert string to StrEnum with p.Result."""
        try:
            return r[EnumT].ok(FlextUtilitiesEnum._parse(enum_cls, value))
        except ValueError:
            members_dict = enum_cls.__members__
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = enum_cls.__name__
            return r[EnumT].fail(
                f"Cannot parse {enum_name}: '{value}'. Valid: {valid}",
            )

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

    @staticmethod
    def is_member[E: StrEnum](enum_cls: type[E], value: object) -> bool:
        """Check if a value matches an enum member by name or value."""
        if isinstance(value, enum_cls):
            return True
        if not isinstance(value, str):
            return False
        return FlextUtilitiesEnum._is_member_by_name(
            enum_cls,
            value,
        ) or FlextUtilitiesEnum._is_member_by_value(enum_cls, value)


__all__: list[str] = ["FlextUtilitiesEnum"]
