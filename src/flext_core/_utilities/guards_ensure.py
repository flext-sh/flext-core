"""Type guard helpers with defensive assertions for normalized runtime data."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sized
from typing import ClassVar

from flext_core import r, t
from flext_core._models.collections import FlextModelsCollections
from flext_core._utilities.guards_type import FlextUtilitiesGuardsType


class FlextUtilitiesGuardsEnsure(FlextUtilitiesGuardsType):
    """Ensure-style guard utility methods for data validation and normalization."""

    _EQUALITY_OPS: ClassVar[
        Mapping[str, Callable[[t.RecursiveContainer, t.RecursiveContainer], bool]]
    ] = {
        "eq": lambda val, cmp: val == cmp,
        "ne": lambda val, cmp: val != cmp,
    }

    _MEMBERSHIP_OPS: ClassVar[
        Mapping[str, Callable[[t.RecursiveContainer, t.RecursiveContainerList], bool]]
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
    def _resolve_numeric(value: t.RecursiveContainer) -> t.Numeric:
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
                pass
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
        value: t.RecursiveContainer,
        contains: t.RecursiveContainer,
    ) -> bool:
        """Check if iterable value contains the target."""
        if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
            iterable_value: Iterable[t.RecursiveContainer] = value
            return any(item == contains for item in iterable_value)
        return False

    @staticmethod
    def chk(
        value: t.RecursiveContainer,
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
        # None/type checks (special semantics, not data-driven)
        if guard_spec.none is True and value is not None:
            return False
        if guard_spec.none is False and value is None:
            return False
        if guard_spec.is_ is not None and not isinstance(value, guard_spec.is_):
            return False
        if guard_spec.not_ is not None and isinstance(value, guard_spec.not_):
            return False
        # Equality checks via dispatch
        for op_name, check_fn in FlextUtilitiesGuardsEnsure._EQUALITY_OPS.items():
            spec_val = getattr(guard_spec, op_name, None)
            if spec_val is not None and not check_fn(value, spec_val):
                return False
        # Membership checks via dispatch
        for mem_op, mem_fn in FlextUtilitiesGuardsEnsure._MEMBERSHIP_OPS.items():
            mem_val: t.RecursiveContainerList | None = getattr(guard_spec, mem_op, None)
            if mem_val is not None and not mem_fn(value, mem_val):
                return False
        # Numeric/size checks via dispatch
        check_val = FlextUtilitiesGuardsEnsure._resolve_numeric(value)
        for op_name, num_fn in FlextUtilitiesGuardsEnsure._NUMERIC_OPS.items():
            spec_val_num: float | None = getattr(guard_spec, op_name, None)
            if spec_val_num is not None and not num_fn(check_val, spec_val_num):
                return False
        # Empty checks
        if guard_spec.empty is True and check_val != 0:
            return False
        if guard_spec.empty is False and check_val == 0:
            return False
        # String-specific checks
        if isinstance(value, str):
            if not FlextUtilitiesGuardsEnsure._check_string_ops(value, guard_spec):
                return False
        elif (
            guard_spec.contains is not None
            and not FlextUtilitiesGuardsEnsure._check_iterable_contains(
                value,
                guard_spec.contains,
            )
        ):
            return False
        return True

    @staticmethod
    def _to_container_or_str(value: t.RecursiveContainer) -> t.Container:
        """Normalize a value to Container: pass through if already, else str()."""
        return value if FlextUtilitiesGuardsEnsure.container(value) else str(value)

    @staticmethod
    def _check_validator(
        value: t.RecursiveContainer,
        validator: Callable[[t.RecursiveContainer], bool]
        | type
        | tuple[type, ...]
        | None,
    ) -> bool:
        """Evaluate validator against value. Returns True if guard passes."""
        if isinstance(validator, type):
            return isinstance(value, validator)
        if FlextUtilitiesGuardsEnsure.object_tuple(validator):
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
        default: t.RecursiveContainer | None,
        *,
        return_value: bool,
        fail_msg: str,
    ) -> t.Container | bool | r[t.Container]:
        """Return default or fail result for guard misses."""
        if default is not None:
            return FlextUtilitiesGuardsEnsure._to_container_or_str(default)
        return r[t.Container].fail(fail_msg) if return_value else False

    @staticmethod
    def guard(
        value: t.RecursiveContainer,
        validator: Callable[[t.RecursiveContainer], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.RecursiveContainer | None = None,
        return_value: bool = False,
    ) -> t.Container | bool | r[t.Container]:
        try:
            if FlextUtilitiesGuardsEnsure._check_validator(value, validator):
                if return_value:
                    return FlextUtilitiesGuardsEnsure._to_container_or_str(value)
                return True
            return FlextUtilitiesGuardsEnsure._guard_fallback(
                default,
                return_value=return_value,
                fail_msg="Guard validation failed",
            )
        except (TypeError, ValueError, AttributeError):
            return FlextUtilitiesGuardsEnsure._guard_fallback(
                default,
                return_value=return_value,
                fail_msg="Guard validation raised an exception",
            )


__all__ = ["FlextUtilitiesGuardsEnsure"]
