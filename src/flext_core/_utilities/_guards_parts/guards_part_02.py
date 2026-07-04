"""Type guards and ensure-style validation helpers for runtime data.

FlextUtilitiesGuards is the FLAT contributor class to the FlextUtilities facade,
composing type/model/protocol predicates (via MRO) with high-level chk/guard
validation engines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import (
    FlextProtocolsResult as p,
    r,
    t,
)

from .guards_part_01 import (
    FlextUtilitiesGuards as FlextUtilitiesGuardsPart01,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextUtilitiesGuards(FlextUtilitiesGuardsPart01):
    @staticmethod
    def _to_container_or_str(value: t.JsonPayload) -> t.JsonValue:
        """Normalize a value to Container: pass through if already, else str()."""
        return value if FlextUtilitiesGuards.container(value) else str(value)

    @staticmethod
    def _check_validator(
        value: t.JsonValue,
        validator: Callable[[t.JsonValue], bool] | type | tuple[type, ...] | None,
    ) -> bool:
        """Evaluate validator against value. Returns True if guard passes."""
        if isinstance(validator, type):
            return isinstance(value, validator)
        if isinstance(validator, tuple):
            tuple_types: tuple[type, ...] = tuple(
                item for item in validator if isinstance(item, type)
            )
            return len(tuple_types) == len(validator) and isinstance(
                value,
                tuple_types,
            )
        if callable(validator):
            return validator(value)
        return bool(value)

    @staticmethod
    def guard(
        value: t.JsonValue,
        validator: Callable[[t.JsonValue], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.Scalar | t.JsonList | t.JsonMapping | None = None,
        return_value: bool = False,
    ) -> t.JsonValue | bool | p.Result[t.JsonValue]:
        fail_msg = "Guard validation failed"
        try:
            validation_passed = FlextUtilitiesGuards._check_validator(value, validator)
        except (TypeError, ValueError, AttributeError):
            fail_msg = "Guard validation raised an exception"
            validation_passed = False
        if validation_passed:
            return (
                FlextUtilitiesGuards._to_container_or_str(value)
                if return_value
                else True
            )
        match default:
            case None:
                return r[t.JsonValue].fail(fail_msg) if return_value else False
            case default_value:
                return FlextUtilitiesGuards._to_container_or_str(default_value)


__all__: list[str] = ["FlextUtilitiesGuards"]
