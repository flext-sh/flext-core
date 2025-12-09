"""Internal enum utilities - DO NOT IMPORT DIRECTLY.

This module provides enum utility functions following the generalized function pattern.
All functionality should be accessed via the u facade in flext_core.utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar, Literal, TypeGuard, TypeIs, cast, overload

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.result import r
from flext_core.typings import t

# Approved modules that can import directly (for testing, internal use)
_APPROVED_MODULES = frozenset({
    "flext_core.utilities",
    "flext_core._utilities",
    "tests.",
})


def _check_direct_access() -> None:
    """Warn if accessed from non-approved module."""
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        caller_module = frame.f_back.f_back.f_globals.get("__name__", "")
        if not any(
            caller_module.startswith(approved) for approved in _APPROVED_MODULES
        ):
            warnings.warn(
                "Direct import from _utilities.enum is deprecated. "
                "Use 'from flext_core import u; u.enum(...)' instead.",
                DeprecationWarning,
                stacklevel=4,
            )


class FlextUtilitiesEnum:
    """Utilities for working with StrEnum in a type-safe way.

    PHILOSOPHY:
    ──────────
    - TypeGuard for narrowing that works in if/else
    - Generic methods that accept ANY StrEnum
    - Caching for performance in frequent validations
    - Direct integration with Pydantic BeforeValidator
    """

    # Cache for metadata methods (manual cache since types aren't hashable for lru_cache)
    _values_cache: ClassVar[dict[type[StrEnum], frozenset[str]]] = {}
    _names_cache: ClassVar[dict[type[StrEnum], frozenset[str]]] = {}
    _members_cache: ClassVar[dict[type[StrEnum], frozenset[StrEnum]]] = {}

    # ─────────────────────────────────────────────────────────────
    # TYPEIS FACTORIES: Generate TypeGuard functions for any StrEnum
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def is_member[E: StrEnum](
        enum_cls: type[E],
        value: t.GeneralValueType,
    ) -> TypeGuard[E]:
        """Generic TypeGuard for any StrEnum.

        Example:
             if FlextUtilitiesEnum.is_member(Status, value):
                 # value: Status (narrowed)
                 process_status(value)

        """
        if isinstance(value, enum_cls):
            return True
        if FlextUtilitiesGuards.is_type(value, str):
            # Check if value is in enum's value-to-member mapping
            value_map = getattr(enum_cls, "_value2member_map_", {})
            return value in value_map
        return False

    @staticmethod
    def is_subset[E: StrEnum](
        enum_cls: type[E],
        valid_members: frozenset[E],
        value: t.GeneralValueType,
    ) -> TypeGuard[E]:
        """TypeGuard for subset of a StrEnum.

        Example:
             ACTIVE_STATES = frozenset({Status.ACTIVE, Status.PENDING})

             if FlextUtilitiesEnum.is_subset(Status, ACTIVE_STATES, value):
                 # value: Status (narrowed to subset)
                 process_active(value)

        """
        if isinstance(value, enum_cls) and value in valid_members:
            return True
        if isinstance(value, str):
            try:
                member = enum_cls(value)
                return member in valid_members
            except ValueError:
                return False
        return False

    # ─────────────────────────────────────────────────────────────
    # CONVERSION: String → StrEnum (type-safe)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse[E: StrEnum](enum_cls: type[E], value: str | E) -> r[E]:
        """Convert string to StrEnum with r.

        Example:
             result = FlextUtilitiesEnum.parse(Status, "active")
             if result.is_success:
                 status: Status = result.value

        """
        if isinstance(value, enum_cls):
            return r[E].ok(value)
        try:
            return r[E].ok(enum_cls(value))
        except ValueError:
            # enum_cls is a StrEnum type, access members via __members__
            # Access enum members via __members__ attribute
            members_dict = getattr(enum_cls, "__members__", {})
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[E].fail(f"Invalid {enum_name}: '{value}'. Valid: {valid}")

    @staticmethod
    def parse_or_default[E: StrEnum](
        enum_cls: type[E],
        value: str | E | None,
        default: E,
    ) -> E:
        """Convert with fallback to default (never fails).

        Example:
             status = FlextUtilitiesEnum.parse_or_default(
                 Status, user_input, Status.PENDING
             )

        """
        if value is None:
            return default
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except ValueError:
            return default

    # ─────────────────────────────────────────────────────────────
    # PYDANTIC VALIDATORS: BeforeValidator factories
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def coerce_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.FlexibleValue], E]:
        """Create BeforeValidator for automatic coercion in Pydantic.

        RECOMMENDED PATTERN for Pydantic fields:

        Example:
             from pydantic import BaseModel
             from typing import Annotated

             # Create the annotated type once
             CoercedStatus = Annotated[
                 Status,
                 BeforeValidator(FlextUtilitiesEnum.coerce_validator(Status))
             ]

             class MyModel(BaseModel):
                 status: CoercedStatus  # Accepts "active" or Status.ACTIVE

        """

        def _coerce(value: t.FlexibleValue) -> E:
            if isinstance(value, enum_cls):
                return value
            if isinstance(value, str):
                try:
                    return enum_cls(value)
                except ValueError:
                    pass
            enum_name = getattr(enum_cls, "__name__", "Enum")
            msg = f"Invalid {enum_name}: {value!r}"
            raise ValueError(msg)

        return _coerce

    @staticmethod
    def coerce_by_name_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.FlexibleValue], E]:
        """BeforeValidator that accepts name OR value of enum.

        Accepts:
             - "ACTIVE" (member name)
             - "active" (member value)
             - Status.ACTIVE (direct member)

        Example:
             StatusByName = Annotated[
                 Status,
                 BeforeValidator(FlextUtilitiesEnum.coerce_by_name_validator(Status))
             ]

        """

        def _coerce(value: t.FlexibleValue) -> E:
            if isinstance(value, enum_cls):
                return value
            if isinstance(value, str):
                # Try by name first
                members_dict = getattr(enum_cls, "__members__", {})
                if value in members_dict:
                    member = members_dict[value]
                    if isinstance(member, enum_cls):
                        return member
                # Then by value
                try:
                    return enum_cls(value)
                except ValueError:
                    pass
            enum_name = getattr(enum_cls, "__name__", "Enum")
            msg = f"Invalid {enum_name}: {value!r}"
            raise ValueError(msg)

        return _coerce

    # ─────────────────────────────────────────────────────────────
    # METADATA: Information about StrEnums
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def values[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of values (cached for performance).

        Business Rule: Returns immutable frozenset of all enum values.
        Cache ensures same object returned for same enum class (identity preserved).
        Manual cache dictionary handles enum classes efficiently.

        Audit Implication: Cached results ensure consistent identity across calls,
        important for set operations and identity checks in audit trails.
        """
        # Check cache first
        if enum_cls in FlextUtilitiesEnum._values_cache:
            return FlextUtilitiesEnum._values_cache[enum_cls]

        # Type hint: enum_cls is type[E] where E is StrEnum, so __members__ exists
        # Use getattr for runtime safety, but type checker knows StrEnum has __members__
        members_dict: dict[str, E] = getattr(enum_cls, "__members__", {})
        result = frozenset(m.value for m in members_dict.values())

        # Cache result
        FlextUtilitiesEnum._values_cache[enum_cls] = result
        return result

    @staticmethod
    def names[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of member names (cached for performance).

        Business Rule: Returns immutable frozenset of all enum member names.
        Cache ensures same object returned for same enum class (identity preserved).
        Manual cache dictionary handles enum classes efficiently.

        Audit Implication: Cached results ensure consistent identity across calls.
        """
        # Check cache first
        if enum_cls in FlextUtilitiesEnum._names_cache:
            return FlextUtilitiesEnum._names_cache[enum_cls]

        # Type hint: enum_cls is type[E] where E is StrEnum, so __members__ exists
        members_dict: dict[str, E] = getattr(enum_cls, "__members__", {})
        result = frozenset(members_dict.keys())

        # Cache result
        FlextUtilitiesEnum._names_cache[enum_cls] = result
        return result

    @staticmethod
    def members[E: StrEnum](enum_cls: type[E]) -> frozenset[E]:
        """Return frozenset of members (cached for performance).

        Business Rule: Returns immutable frozenset of all enum members.
        Cache ensures same object returned for same enum class (identity preserved).
        Manual cache dictionary handles enum classes efficiently.

        Audit Implication: Cached results ensure consistent identity across calls.
        """
        # Check cache first
        if enum_cls in FlextUtilitiesEnum._members_cache:
            # Cache stores frozenset[StrEnum] but we need frozenset[E]
            # Cast is safe because cache is per enum class type
            cached = FlextUtilitiesEnum._members_cache[enum_cls]
            return cast("frozenset[E]", cached)

        # Type hint: enum_cls is type[E] where E is StrEnum, so __members__ exists
        members_dict: dict[str, E] = getattr(enum_cls, "__members__", {})
        result = frozenset(members_dict.values())

        # Cache result - store as frozenset[StrEnum] for compatibility
        FlextUtilitiesEnum._members_cache[enum_cls] = cast("frozenset[StrEnum]", result)
        return result


# ─────────────────────────────────────────────────────────────
# GENERALIZED PUBLIC FUNCTION PATTERN
# ─────────────────────────────────────────────────────────────


class _EnumUtilities:
    """Internal implementation - all methods here are internal to this module."""

    @staticmethod
    def is_member_by_value[E: StrEnum](value: object, enum_cls: type[E]) -> TypeIs[E]:
        """Check membership by value."""
        if isinstance(value, enum_cls):
            return True
        if FlextUtilitiesGuards.is_type(value, str):
            value_map = getattr(enum_cls, "_value2member_map_", {})
            return value in value_map
        return False

    @staticmethod
    def is_member_by_name[E: StrEnum](name: str, enum_cls: type[E]) -> TypeIs[E]:
        """Check membership by name."""
        return name in getattr(enum_cls, "__members__", {})

    @staticmethod
    def parse[E: StrEnum](enum_cls: type[E], value: str | E) -> r[E]:
        """Parse string to enum."""
        if isinstance(value, enum_cls):
            return r[E].ok(value)
        try:
            return r[E].ok(enum_cls(value))
        except ValueError:
            members_dict = getattr(enum_cls, "__members__", {})
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return r[E].fail(f"Invalid {enum_name}: '{value}'. Valid: {valid}")

    @staticmethod
    def coerce[E: StrEnum](enum_cls: type[E], value: str | E) -> E:
        """Coerce value to enum (for Pydantic validators)."""
        if isinstance(value, enum_cls):
            return value
        return enum_cls(value)


# PUBLIC GENERALIZED METHOD - Single entry point with routing
@overload
def enum[E: StrEnum](
    value: object,
    enum_cls: type[E],
    *,
    mode: Literal["is_member"] = "is_member",
    by_name: bool = False,
) -> TypeIs[E]: ...


@overload
def enum[E: StrEnum](
    value: object,
    enum_cls: type[E],
    *,
    mode: Literal["is_name"],
    by_name: bool = False,
) -> TypeIs[E]: ...


@overload
def enum[E: StrEnum](
    value: object,
    enum_cls: type[E],
    *,
    mode: Literal["parse"],
    by_name: bool = False,
) -> r[E]: ...


@overload
def enum[E: StrEnum](
    value: object,
    enum_cls: type[E],
    *,
    mode: Literal["coerce"],
    by_name: bool = False,
) -> E: ...


def enum[E: StrEnum](
    value: object,
    enum_cls: type[E],
    *,
    mode: str = "is_member",
    by_name: bool = False,
) -> bool | r[E] | E:
    """Generalized enum utility function.

    Args:
        value: Value to check/parse/coerce
        enum_cls: The StrEnum class
        mode: Operation mode
            - "is_member": Check if value is member (returns bool, acts as TypeIs[E])
            - "is_name": Check if value is member by name (returns bool, acts as TypeIs[E])
            - "parse": Parse value to enum (returns r[E])
            - "coerce": Coerce value to enum (returns E, raises on failure)
        by_name: For is_member, check by name instead of value

    Returns:
        Depends on mode - bool (for is_member/is_name), r[E] (for parse), or E (for coerce)

    """
    _check_direct_access()

    if mode == "is_member":
        if by_name and isinstance(value, str):
            is_member_result: bool = _EnumUtilities.is_member_by_name(value, enum_cls)
            return is_member_result
        result_bool: bool = _EnumUtilities.is_member_by_value(value, enum_cls)
        return result_bool
    if mode == "is_name":
        return _EnumUtilities.is_member_by_name(str(value), enum_cls)
    if mode == "parse":
        # Type narrowing: value is object, but parse accepts str | E
        # We handle this by checking if it's already an enum instance
        if isinstance(value, enum_cls):
            return _EnumUtilities.parse(enum_cls, value)
        if isinstance(value, str):
            return _EnumUtilities.parse(enum_cls, value)
        # For other types, convert to string
        return _EnumUtilities.parse(enum_cls, str(value))
    if mode == "coerce":
        # Type narrowing: value is object, but coerce accepts str | E
        # coerce always returns E (raises on failure)
        # Explicit type narrowing for type checker
        if isinstance(value, enum_cls):
            # Type narrowing: isinstance check ensures value is E
            # Direct return after isinstance narrowing
            return value
        if isinstance(value, str):
            # Type narrowing: isinstance check ensures value is str
            coerced: E = _EnumUtilities.coerce(enum_cls, value)
            return coerced
        # For other types, convert to string
        # Type narrowing: str(value) is str, which is valid for coerce
        value_str: str = str(value)
        coerced_str: E = _EnumUtilities.coerce(enum_cls, value_str)
        return coerced_str
    error_msg = f"Unknown mode: {mode}"
    raise ValueError(error_msg)


__all__ = [
    "FlextUtilitiesEnum",
    "enum",
]
