"""Utilities module - FlextUtilitiesEnum.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import cache
from typing import TypeGuard

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

    # ─────────────────────────────────────────────────────────────
    # TYPEIS FACTORIES: Generate TypeGuard functions for any StrEnum
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def is_member[E: StrEnum](enum_cls: type[E], value: object) -> TypeGuard[E]:
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
        value: object,
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
    def parse[E: StrEnum](enum_cls: type[E], value: str | E) -> "FlextResult[E]":
        """Convert string to StrEnum with FlextResult.

        Example:
             result = FlextUtilitiesEnum.parse(Status, "active")
             if result.is_success:
                 status: Status = result.value

        """
        from flext_core.result import FlextResult

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
    @cache
    def values[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of values (cached for performance)."""
        members_dict = getattr(enum_cls, "__members__", {})
        return frozenset(m.value for m in members_dict.values())

    @staticmethod
    @cache
    def names[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of member names (cached)."""
        members_dict = getattr(enum_cls, "__members__", {})
        return frozenset(members_dict.keys())

    @staticmethod
    @cache
    def members[E: StrEnum](enum_cls: type[E]) -> frozenset[E]:
        """Return frozenset of members (cached)."""
        members_dict = getattr(enum_cls, "__members__", {})
        return frozenset(members_dict.values())
