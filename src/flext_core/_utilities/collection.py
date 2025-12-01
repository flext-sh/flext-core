"""Utilities module - FlextUtilitiesCollection.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextUtilitiesCollection:
    """Utilities for collection conversion with StrEnums.

    PATTERNS collections.abc:
    ────────────────────────
    - Sequence[E] for immutable lists
    - Mapping[str, E] for immutable dicts
    - Iterable[E] for any iterable
    """

    # ─────────────────────────────────────────────────────────────
    # LIST CONVERSIONS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_sequence[E: StrEnum](
        enum_cls: type[E],
        values: Iterable[str | E],
    ) -> FlextResult[tuple[E, ...]]:
        """Convert sequence of strings to tuple of StrEnum.

        Example:
             result = FlextUtilitiesCollection.parse_sequence(
                 Status, ["active", "pending"]
             )
             if result.is_success:
                 statuses: tuple[Status, ...] = result.value

        """
        parsed: list[E] = []
        errors: list[str] = []

        for idx, val in enumerate(values):
            if isinstance(val, enum_cls):
                parsed.append(val)
            else:
                try:
                    parsed.append(enum_cls(val))
                except ValueError:
                    errors.append(f"[{idx}]: '{val}'")

        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return FlextResult.fail(f"Invalid {enum_name} values: {', '.join(errors)}")
        return FlextResult.ok(tuple(parsed))

    @staticmethod
    def coerce_list_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[FlextTypes.FlexibleValue], list[E]]:
        """BeforeValidator for list of StrEnums.

        Example:
             StatusList = Annotated[
                 list[Status],
                 BeforeValidator(FlextUtilitiesCollection.coerce_list_validator(Status))
             ]

             class MyModel(BaseModel):
                 statuses: StatusList  # Accepts ["active", "pending"]

        """

        def _coerce(value: FlextTypes.FlexibleValue) -> list[E]:
            if not isinstance(value, (list, tuple, set, frozenset)):
                msg = f"Expected sequence, got {type(value).__name__}"
                raise TypeError(msg)

            result: list[E] = []
            for idx, item in enumerate(value):
                if isinstance(item, enum_cls):
                    result.append(item)
                elif isinstance(item, str):
                    try:
                        result.append(enum_cls(item))
                    except ValueError as err:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} at [{idx}]: {item!r}"
                        raise ValueError(msg) from err
                else:
                    msg = f"Expected str at [{idx}], got {type(item).__name__}"
                    raise TypeError(msg)
            return result

        return _coerce

    # ─────────────────────────────────────────────────────────────
    # DICT CONVERSIONS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_mapping[E: StrEnum](
        enum_cls: type[E],
        mapping: Mapping[str, str | E],
    ) -> FlextResult[dict[str, E]]:
        """Convert Mapping with string values to dict with StrEnum.

        Example:
             result = FlextUtilitiesCollection.parse_mapping(
                 Status, {"user1": "active", "user2": "pending"}
             )

        """
        parsed: dict[str, E] = {}
        errors: list[str] = []

        for key, val in mapping.items():
            if isinstance(val, enum_cls):
                parsed[key] = val
            else:
                try:
                    parsed[key] = enum_cls(val)
                except ValueError:
                    errors.append(f"'{key}': '{val}'")

        if errors:
            enum_name = getattr(enum_cls, "__name__", "Enum")
            return FlextResult.fail(f"Invalid {enum_name} values: {', '.join(errors)}")
        return FlextResult.ok(parsed)

    @staticmethod
    def coerce_dict_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[FlextTypes.FlexibleValue], dict[str, E]]:
        """BeforeValidator for dict with StrEnum values.

        Example:
             StatusDict = Annotated[
                 dict[str, Status],
                 BeforeValidator(FlextUtilitiesCollection.coerce_dict_validator(Status))
             ]

        """

        def _coerce(value: FlextTypes.FlexibleValue) -> dict[str, E]:
            if not isinstance(value, dict):
                msg = f"Expected dict, got {type(value).__name__}"
                raise TypeError(msg)

            result: dict[str, E] = {}
            for key, val in value.items():
                if isinstance(val, enum_cls):
                    result[key] = val
                elif isinstance(val, str):
                    try:
                        result[key] = enum_cls(val)
                    except ValueError as err:
                        enum_name = getattr(enum_cls, "__name__", "Enum")
                        msg = f"Invalid {enum_name} at '{key}': {val!r}"
                        raise ValueError(msg) from err
                else:
                    msg = f"Expected str at '{key}', got {type(val).__name__}"
                    raise TypeError(msg)
            return result

        return _coerce
