"""Type guards and ensure-style validation helpers for runtime data.

FlextUtilitiesGuards is the FLAT contributor class to the FlextUtilities facade,
composing type/model/protocol predicates (via MRO) with high-level chk/guard
validation engines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sized
from typing import ClassVar

from flext_core import (
    r,
    t,
)
from flext_core._models.collections import FlextModelsCollections
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core._utilities.guards_type_protocol import (
    FlextUtilitiesGuardsTypeProtocol,
)


class FlextUtilitiesGuards(
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
):
    """Unified guard utilities: type narrowing + chk/guard validation engines."""

    @staticmethod
    def _in_list(value: t.GuardInput, container: t.JsonList) -> bool:
        return value in container

    @staticmethod
    def _not_in_list(value: t.GuardInput, container: t.JsonList) -> bool:
        return value not in container

    _EQUALITY_OPS: ClassVar[
        t.MappingKV[str, Callable[[t.GuardInput, t.GuardInput], bool]]
    ] = {
        "eq": operator.eq,
        "ne": operator.ne,
    }
    _MEMBERSHIP_OPS: ClassVar[
        t.MappingKV[str, Callable[[t.GuardInput, t.JsonList], bool]]
    ] = {
        "in_": _in_list,
        "not_in": _not_in_list,
    }
    _NUMERIC_OPS: ClassVar[Mapping[str, Callable[[t.Numeric, float], bool]]] = {
        "gt": lambda v, c: v > c,
        "gte": lambda v, c: v >= c,
        "lt": lambda v, c: v < c,
        "lte": lambda v, c: v <= c,
    }
    _STRING_OPS: ClassVar[Mapping[str, Callable[[str, str], bool]]] = {
        "gt": lambda v, c: v > c,
        "gte": lambda v, c: v >= c,
        "lt": lambda v, c: v < c,
        "lte": lambda v, c: v <= c,
    }

    @staticmethod
    def _resolve_numeric(value: t.GuardInput) -> t.Numeric:
        """Extract numeric value (raw for numbers, len for sized types)."""
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            sized_value: Sized = value
            return len(sized_value)
        return 0

    @staticmethod
    def _check_string_ops(
        value: str,
        guard_spec: FlextModelsCollections.GuardCheckSpec,
    ) -> bool:
        """Check string-specific operations (starts, ends, contains)."""
        if guard_spec.starts is not None and not value.startswith(guard_spec.starts):
            return False
        return not (guard_spec.ends is not None and not value.endswith(guard_spec.ends))

    @staticmethod
    def _check_iterable_contains(
        value: t.GuardInput,
        contains: t.GuardInput,
    ) -> bool:
        """Check if iterable value contains the target (strings handled upstream)."""
        if isinstance(value, str):
            return isinstance(contains, str) and contains in value
        if isinstance(value, bytes):
            return isinstance(contains, bytes) and contains in value
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            return contains in value
        return False

    @staticmethod
    def _check_spec_ops(
        value: t.GuardInput,
        guard_spec: FlextModelsCollections.GuardCheckSpec,
        check_val: t.Numeric,
    ) -> bool:
        """Apply equality/membership/numeric op dicts against guard_spec."""
        result = True
        for op_name, check_fn in FlextUtilitiesGuards._EQUALITY_OPS.items():
            spec_val = getattr(guard_spec, op_name, None)
            if spec_val is not None and not check_fn(value, spec_val):
                result = False
                break
        if result:
            for mem_op, mem_fn in FlextUtilitiesGuards._MEMBERSHIP_OPS.items():
                mem_raw = getattr(guard_spec, mem_op, None)
                if mem_raw is not None and not mem_fn(
                    value,
                    t.json_list_adapter().validate_python(mem_raw),
                ):
                    result = False
                    break
        if result:
            for op_name, num_fn in FlextUtilitiesGuards._NUMERIC_OPS.items():
                spec_val_num: float | str | None = getattr(guard_spec, op_name, None)
                if spec_val_num is None:
                    continue
                if isinstance(spec_val_num, str) and isinstance(value, str):
                    str_fn = FlextUtilitiesGuards._STRING_OPS[op_name]
                    if not str_fn(value, spec_val_num):
                        result = False
                        break
                    continue
                if isinstance(spec_val_num, (int, float)) and not num_fn(
                    check_val, spec_val_num
                ):
                    result = False
                    break
        if result and isinstance(value, str):
            result = FlextUtilitiesGuards._check_string_ops(value, guard_spec)
        if result:
            match guard_spec.contains:
                case None:
                    pass
                case contains_value:
                    result = FlextUtilitiesGuardsTypeCore.container(
                        value
                    ) and FlextUtilitiesGuards._check_iterable_contains(
                        value,
                        contains_value,
                    )
        return result

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
        check_val = FlextUtilitiesGuards._resolve_numeric(value)
        return not (
            (guard_spec.none is True and value is not None)
            or (guard_spec.none is False and value is None)
            or (guard_spec.is_ is not None and not isinstance(value, guard_spec.is_))
            or (guard_spec.not_ is not None and isinstance(value, guard_spec.not_))
            or not FlextUtilitiesGuards._check_spec_ops(value, guard_spec, check_val)
            or (guard_spec.empty is True and check_val != 0)
            or (guard_spec.empty is False and check_val == 0)
        )

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
    ) -> t.JsonValue | bool | r[t.JsonValue]:
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
