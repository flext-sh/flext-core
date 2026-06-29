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
    FlextModelsCollections,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
    t,
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
        if isinstance(value, t.NUMERIC_TYPES):
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
                if isinstance(spec_val_num, t.NUMERIC_TYPES) and not num_fn(
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
            criteria_update: dict[str, t.GuardInput | None] = {
                field_name: getattr(criteria_spec, field_name)
                for field_name in criteria_spec.model_fields_set
            }
            guard_spec = guard_spec.model_copy(update=criteria_update)
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


__all__: list[str] = ["FlextUtilitiesGuards"]
