"""Type guards and ensure-style validation helpers for runtime data.

FlextUtilitiesGuards is the FLAT contributor class to the FlextUtilities facade,
composing type/model/protocol predicates (via MRO) with high-level chk/guard
validation engines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sized
from typing import ClassVar

from flext_core import (
    FlextModelsCollections,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
    r,
    t,
)


class FlextUtilitiesGuards(
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
):
    """Unified guard utilities: type narrowing + chk/guard validation engines."""

    _EQUALITY_OPS: ClassVar[
        Mapping[str, Callable[[t.GuardInput, t.GuardInput], bool]]
    ] = {
        "eq": lambda val, cmp: val == cmp,
        "ne": lambda val, cmp: val != cmp,
    }

    _MEMBERSHIP_OPS: ClassVar[
        Mapping[str, Callable[[t.GuardInput, t.JsonList], bool]]
    ] = {
        "in_": lambda val, cmp: val in cmp,
        "not_in": lambda val, cmp: val not in cmp,
    }

    _NUMERIC_OPS: ClassVar[Mapping[str, Callable[[t.Numeric, float], bool]]] = {
        "gt": lambda val, cmp: val > cmp,
        "gte": lambda val, cmp: val >= cmp,
        "lt": lambda val, cmp: val < cmp,
        "lte": lambda val, cmp: val <= cmp,
    }

    @staticmethod
    def _resolve_numeric(value: t.GuardInput) -> t.Numeric:
        """Extract numeric value (raw for numbers, len for sized types)."""
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            sized_value: Sized = value
            return len(sized_value)
        if hasattr(value, "__len__"):
            try:
                len_method = getattr(value, "__len__", None)
                if callable(len_method):
                    length = len_method()
                    if isinstance(length, int):
                        return length
            except (TypeError, AttributeError):
                return 0
        return 0

    @staticmethod
    def _check_string_ops(
        value: str,
        guard_spec: FlextModelsCollections.GuardCheckSpec,
    ) -> bool:
        """Check string-specific operations (match, starts, ends, contains)."""
        if guard_spec.match is not None and not re.search(guard_spec.match, value):
            return False
        if guard_spec.starts is not None and not value.startswith(guard_spec.starts):
            return False
        if guard_spec.ends is not None and not value.endswith(guard_spec.ends):
            return False
        return not (
            guard_spec.contains is not None
            and isinstance(guard_spec.contains, str)
            and guard_spec.contains not in value
        )

    @staticmethod
    def _check_iterable_contains(
        value: t.GuardInput,
        contains: t.GuardInput,
    ) -> bool:
        """Check if iterable value contains the target."""
        if isinstance(value, str):
            return isinstance(contains, str) and contains in value
        if isinstance(value, bytes):
            return isinstance(contains, bytes) and contains in value
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            return contains in value
        return False

    @staticmethod
    def chk(
        value: t.GuardInput,
        spec: FlextModelsCollections.GuardCheckSpec | None = None,
        **criteria: t.GuardInput,
    ) -> bool:
        guard_spec = (
            spec if spec is not None else FlextModelsCollections.GuardCheckSpec()
        )
        if criteria:
            criteria_spec = FlextModelsCollections.GuardCheckSpec.model_validate(
                criteria,
            )
            guard_spec = guard_spec.model_copy(update=criteria_spec.model_dump())
        if guard_spec.none is True and value is not None:
            return False
        if guard_spec.none is False and value is None:
            return False
        if guard_spec.is_ is not None and not isinstance(value, guard_spec.is_):
            return False
        if guard_spec.not_ is not None and isinstance(value, guard_spec.not_):
            return False
        for op_name, check_fn in FlextUtilitiesGuards._EQUALITY_OPS.items():
            spec_val = getattr(guard_spec, op_name, None)
            if spec_val is not None and not check_fn(value, spec_val):
                return False
        for mem_op, mem_fn in FlextUtilitiesGuards._MEMBERSHIP_OPS.items():
            mem_raw = getattr(guard_spec, mem_op, None)
            if mem_raw is not None and not mem_fn(
                value,
                t.json_list_adapter().validate_python(mem_raw),
            ):
                return False
        check_val = FlextUtilitiesGuards._resolve_numeric(value)
        for op_name, num_fn in FlextUtilitiesGuards._NUMERIC_OPS.items():
            spec_val_num: float | None = getattr(guard_spec, op_name, None)
            if spec_val_num is not None and not num_fn(check_val, spec_val_num):
                return False
        if guard_spec.empty is True and check_val != 0:
            return False
        if guard_spec.empty is False and check_val == 0:
            return False
        if isinstance(value, str):
            if not FlextUtilitiesGuards._check_string_ops(value, guard_spec):
                return False
        elif (
            guard_spec.contains is not None
            and not FlextUtilitiesGuards._check_iterable_contains(
                value,
                guard_spec.contains,
            )
        ):
            return False
        return True

    @staticmethod
    def _to_container_or_str(value: t.JsonValue) -> t.JsonValue:
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
    def _guard_fallback(
        default: t.JsonValue | None,
        *,
        return_value: bool,
        fail_msg: str,
    ) -> t.JsonValue | bool | r[t.JsonValue]:
        """Return default or fail result for guard misses."""
        if default is not None:
            return FlextUtilitiesGuards._to_container_or_str(default)
        return r[t.JsonValue].fail(fail_msg) if return_value else False

    @staticmethod
    def guard(
        value: t.JsonValue,
        validator: Callable[[t.JsonValue], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.JsonValue | None = None,
        return_value: bool = False,
    ) -> t.JsonValue | bool | r[t.JsonValue]:
        try:
            if FlextUtilitiesGuards._check_validator(value, validator):
                if return_value:
                    return FlextUtilitiesGuards._to_container_or_str(value)
                return True
            return FlextUtilitiesGuards._guard_fallback(
                default,
                return_value=return_value,
                fail_msg="Guard validation failed",
            )
        except (TypeError, ValueError, AttributeError):
            return FlextUtilitiesGuards._guard_fallback(
                default,
                return_value=return_value,
                fail_msg="Guard validation raised an exception",
            )


__all__: list[str] = ["FlextUtilitiesGuards"]
