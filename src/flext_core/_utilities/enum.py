"""Utilities module - FlextUtilitiesEnum.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar, TypeGuard

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


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
        enum_cls: type[E], value: FlextTypes.GeneralValueType
    ) -> TypeGuard[E]:
        """Generic TypeGuard for any StrEnum.

        Example:
             if FlextUtilitiesEnum.is_member(Status, value):
                 # value: Status (narrowed)
                 process_status(value)

        """
        if isinstance(value, enum_cls):
            return True
        if isinstance(value, str):
            # Check if value is in enum's value-to-member mapping
            value_map = getattr(enum_cls, "_value2member_map_", {})
            return value in value_map
        return False

    @staticmethod
    def is_subset[E: StrEnum](
        enum_cls: type[E],
        valid_members: frozenset[E],
        value: FlextTypes.GeneralValueType,
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
    def parse[E: StrEnum](enum_cls: type[E], value: str | E) -> FlextResult[E]:
        """Convert string to StrEnum with FlextResult.

        Example:
             result = FlextUtilitiesEnum.parse(Status, "active")
             if result.is_success:
                 status: Status = result.value

        """
        if isinstance(value, enum_cls):
            return FlextResult.ok(value)
        try:
            return FlextResult.ok(enum_cls(value))
        except ValueError:
            # enum_cls is a StrEnum type, access members via __members__
            # Access enum members via __members__ attribute
            members_dict = getattr(enum_cls, "__members__", {})
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return FlextResult.fail(f"Invalid {enum_name}: '{value}'. Valid: {valid}")

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
    ) -> Callable[[FlextTypes.FlexibleValue], E]:
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

        def _coerce(value: FlextTypes.FlexibleValue) -> E:
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
    ) -> Callable[[FlextTypes.FlexibleValue], E]:
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

        def _coerce(value: FlextTypes.FlexibleValue) -> E:
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
            return FlextUtilitiesEnum._members_cache[enum_cls]  # type: ignore[return-value]

        # Type hint: enum_cls is type[E] where E is StrEnum, so __members__ exists
        members_dict: dict[str, E] = getattr(enum_cls, "__members__", {})
        result = frozenset(members_dict.values())

        # Cache result
        FlextUtilitiesEnum._members_cache[enum_cls] = result  # type: ignore[assignment]
        return result
