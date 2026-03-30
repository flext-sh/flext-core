"""Type guard helpers with defensive assertions for normalized runtime data."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from typing import ClassVar

from pydantic import BaseModel, ValidationError

from flext_core import FlextUtilitiesGuardsType, p, r, t
from flext_core._models.collections import FlextModelsCollections


class FlextUtilitiesGuardsEnsure(FlextUtilitiesGuardsType):
    """Ensure-style guard utility methods for data validation and normalization."""

    @staticmethod
    def _ensure_to_dict(
        value: t.NormalizedValue | None,
        default: t.ContainerMapping | None,
    ) -> t.ContainerMapping:
        if value is None:
            empty_dict: t.ContainerMapping = {}
            return default if default is not None else empty_dict
        if isinstance(value, Mapping):
            mapping_value: t.ContainerMapping = value
            normalized: t.MutableContainerMapping = {}
            for key, item_value in mapping_value.items():
                normalized[str(key)] = item_value
            return normalized
        wrapped_dict: t.ContainerMapping = {"value": value}
        return wrapped_dict

    @staticmethod
    def _ensure_to_list(
        value: t.NormalizedValue | t.ContainerList | None,
        default: t.ContainerList | None,
    ) -> t.ContainerList:
        if value is None:
            empty_list: t.ContainerList = []
            return default if default is not None else empty_list
        if isinstance(value, list):
            return list(value)
        single_item_list: t.ContainerList = [value]
        return single_item_list

    @staticmethod
    def _guard_check_condition[T: t.GuardInput](
        value: T,
        condition: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
        | str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if isinstance(condition, type):
            if FlextUtilitiesGuardsEnsure.is_container(value):
                return FlextUtilitiesGuardsEnsure._guard_check_type(
                    value,
                    condition,
                    context_name,
                    error_msg,
                )
            return (
                error_msg
                or f"{context_name} must be {condition.__name__}, got {type(value).__name__}"
            )
        if FlextUtilitiesGuardsEnsure._is_type_tuple(condition):
            if FlextUtilitiesGuardsEnsure.is_container(value):
                return FlextUtilitiesGuardsEnsure._guard_check_type(
                    value,
                    condition,
                    context_name,
                    error_msg,
                )
            return (
                error_msg
                or f"{context_name} type check failed for {type(value).__name__}"
            )
        if isinstance(condition, p.ValidatorSpec):
            if not FlextUtilitiesGuardsEnsure.is_container(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value: t.NormalizedValue = value
            return FlextUtilitiesGuardsEnsure._guard_check_validator(
                typed_value,
                condition,
                context_name,
                error_msg,
            )
        if isinstance(condition, str):
            if not FlextUtilitiesGuardsEnsure.is_container(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            typed_value_s: t.NormalizedValue = value
            return FlextUtilitiesGuardsEnsure._guard_check_string_shortcut(
                typed_value_s,
                condition,
                context_name,
                error_msg,
            )
        if callable(condition):
            return FlextUtilitiesGuardsEnsure._guard_check_predicate(
                value,
                condition,
                context_name,
                error_msg,
            )
        return error_msg or f"{context_name} invalid guard condition type"

    @staticmethod
    def _guard_check_predicate(
        value: t.GuardInput,
        condition: Callable[..., t.GuardInput | bool],
        context_name: str,
        error_msg: str | None,
    ) -> str:
        try:
            if not bool(condition(value)):
                if error_msg is None:
                    func_name = (
                        condition.__name__
                        if hasattr(condition, "__name__")
                        else "custom"
                    )
                    return f"{context_name} failed {func_name} check"
                return error_msg
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            if error_msg is None:
                return f"{context_name} guard check raised: {e}"
            return error_msg
        return ""

    @staticmethod
    def _is_numeric_not_bool(value: t.NormalizedValue) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    _SHORTCUT_VALIDATORS: ClassVar[
        Mapping[str, tuple[Callable[[t.NormalizedValue], bool], str]]
    ] = {
        "non_empty": (
            lambda v: bool(v) and isinstance(v, (str, list, dict)),
            "must be non-empty",
        ),
        "positive": (
            lambda v: FlextUtilitiesGuardsEnsure._is_numeric_not_bool(v) and v > 0,  # type: ignore[operator]
            "must be positive number",
        ),
        "non_negative": (
            lambda v: FlextUtilitiesGuardsEnsure._is_numeric_not_bool(v) and v >= 0,  # type: ignore[operator]
            "must be non-negative number",
        ),
        "dict": (
            lambda v: hasattr(v, "items") and not isinstance(v, (str, bytes)),
            "must be dict-like",
        ),
        "list": (
            lambda v: (
                hasattr(v, "__iter__")
                and not isinstance(v, (str, bytes))
                and not hasattr(v, "items")
            ),
            "must be list-like",
        ),
        "string": (lambda v: isinstance(v, str), "must be string"),
        "int": (
            lambda v: isinstance(v, int) and not isinstance(v, bool),
            "must be int",
        ),
        "float": (
            lambda v: FlextUtilitiesGuardsEnsure._is_numeric_not_bool(v),
            "must be float",
        ),
        "bool": (lambda v: isinstance(v, bool), "must be bool"),
    }

    @staticmethod
    def _guard_check_string_shortcut(
        value: t.NormalizedValue,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        entry = FlextUtilitiesGuardsEnsure._SHORTCUT_VALIDATORS.get(
            condition.lower(),
        )
        if entry is None:
            return error_msg or f"{context_name} unknown guard shortcut: {condition}"
        predicate, default_msg = entry
        if predicate(value):
            return ""
        return error_msg or f"{context_name} {default_msg}"

    @staticmethod
    def _guard_check_type(
        value: t.NormalizedValue,
        condition: type | tuple[type, ...],
        context_name: str,
        error_msg: str | None,
    ) -> str:
        type_match = isinstance(value, condition)
        if not type_match:
            if error_msg is None:
                type_name = (
                    condition.__name__
                    if isinstance(condition, type)
                    else " | ".join(c.__name__ for c in condition)
                )
                return f"{context_name} must be {type_name}, got {value.__class__.__name__}"
            return error_msg
        return ""

    @staticmethod
    def _guard_check_validator(
        value: t.NormalizedValue,
        condition: p.ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if not FlextUtilitiesGuardsEnsure.is_container(value):
            return error_msg or f"{context_name} must be a valid configuration value"
        if not condition(value):
            if error_msg is None:
                desc = (
                    getattr(condition, "description", "validation")
                    if hasattr(condition, "description")
                    else "validation"
                )
                return f"{context_name} failed {desc} check"
            return error_msg
        return ""

    @staticmethod
    def _guard_handle_failure[T](
        error_message: str,
        *,
        return_value: bool,
        default: T | None,
    ) -> r[T] | T:
        if return_value:
            if default is not None:
                return default
            return r[T].fail(error_message)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(error_message)

    _EQUALITY_OPS: ClassVar[
        Mapping[str, Callable[[t.NormalizedValue, t.NormalizedValue], bool]]
    ] = {
        "eq": lambda val, cmp: val == cmp,
        "ne": lambda val, cmp: val != cmp,
    }

    _MEMBERSHIP_OPS: ClassVar[
        Mapping[str, Callable[[t.NormalizedValue, t.ContainerList], bool]]
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
    def _resolve_numeric(value: t.NormalizedValue) -> t.Numeric:
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
        value: t.NormalizedValue,
        contains: t.NormalizedValue,
    ) -> bool:
        """Check if iterable value contains the target."""
        if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
            iterable_value: Iterable[t.NormalizedValue] = value
            return any(item == contains for item in iterable_value)
        return False

    @staticmethod
    def chk(
        value: t.NormalizedValue,
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
        for op_name, check_fn in FlextUtilitiesGuardsEnsure._MEMBERSHIP_OPS.items():
            spec_val: t.ContainerList | None = getattr(guard_spec, op_name, None)
            if spec_val is not None and not check_fn(value, spec_val):
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
    def empty(items: t.NormalizedValue | None) -> bool:
        if items is None:
            return True
        if isinstance(items, Sized):
            return not items
        return not bool(items)

    @staticmethod
    def ensure(
        value: t.NormalizedValue,
        *,
        target_type: str = "auto",
        default: str | t.ContainerList | t.NormalizedValue | None = None,
    ) -> str | t.ContainerList | t.NormalizedValue | t.ContainerMapping:
        if target_type == "str":
            str_default = default if isinstance(default, str) else ""
            return (
                value
                if isinstance(value, str)
                else str(value)
                if value is not None
                else str_default
            )
        if target_type == "str_list":
            str_list_default: t.StrSequence | None = None
            if isinstance(default, list):
                default_values: t.ContainerList = default
                str_list_default = [str(item) for item in default_values]
            if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                seq_value: t.ContainerList = value
                return list(seq_value)
            if value is None:
                result_str_list: t.ContainerList = (
                    list(str_list_default) if str_list_default else []
                )
                return result_str_list
            return [value]
        if target_type == "dict":
            dict_default: t.ContainerMapping | None = None
            if isinstance(default, Mapping):
                dict_default = default
            return FlextUtilitiesGuardsEnsure._ensure_to_dict(value, dict_default)
        if target_type == "auto" and isinstance(value, Mapping):
            mapping_value: t.ContainerMapping = value
            normalized_auto: t.MutableContainerMapping = {}
            for key, item_value in mapping_value.items():
                normalized_auto[str(key)] = item_value
            return normalized_auto
        list_default: t.ContainerList | None = None
        if FlextUtilitiesGuardsEnsure.is_object_list(default):
            list_default = default
        return FlextUtilitiesGuardsEnsure._ensure_to_list(value, list_default)

    @staticmethod
    def extract_mapping_or_none(value: t.NormalizedValue) -> r[t.ConfigMap]:
        if isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsEnsure.is_configuration_mapping(value):
            return r[t.ConfigMap].ok(value)
        return r[t.ConfigMap].fail("Value is not a configuration mapping")

    @staticmethod
    def _to_container_or_str(value: t.NormalizedValue) -> t.Container:
        """Normalize a value to Container: pass through if already, else str()."""
        return value if FlextUtilitiesGuardsEnsure.is_container(value) else str(value)

    @staticmethod
    def _check_validator(
        value: t.NormalizedValue,
        validator: Callable[[t.NormalizedValue], bool] | type | tuple[type, ...] | None,
    ) -> bool:
        """Evaluate validator against value. Returns True if guard passes."""
        if isinstance(validator, type):
            return isinstance(value, validator)
        if FlextUtilitiesGuardsEnsure.is_object_tuple(validator):
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
        default: t.NormalizedValue | None,
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
        value: t.NormalizedValue,
        validator: Callable[[t.NormalizedValue], bool]
        | type
        | tuple[type, ...]
        | None = None,
        *,
        default: t.NormalizedValue | None = None,
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

    @staticmethod
    def guard_result[T: t.GuardInput](
        value: T,
        *conditions: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
        | str,
        error_message: str | None = None,
        context: str | None = None,
        default: T | None = None,
        return_value: bool = False,
    ) -> r[T] | T:
        context_name = context or "Value"
        if not conditions:
            if bool(value):
                return value if return_value else r[T].ok(value)
            failure_message = error_message or f"{context_name} guard failed"
            return FlextUtilitiesGuardsEnsure._guard_handle_failure(
                failure_message,
                return_value=return_value,
                default=default,
            )
        for condition in conditions:
            condition_error = FlextUtilitiesGuardsEnsure._guard_check_condition(
                value,
                condition,
                context_name,
                error_message,
            )
            if condition_error:
                return FlextUtilitiesGuardsEnsure._guard_handle_failure(
                    condition_error,
                    return_value=return_value,
                    default=default,
                )
        return value if return_value else r[T].ok(value)

    # Validation methods (from consolidated guards_validation)
    @staticmethod
    def validate_hostname(hostname: str, field_name: str = "hostname") -> r[str]:

        hostname_pattern = (
            "^(?!-)[a-zA-Z0-9-]{1,63}(?<!-)(\\.[a-zA-Z0-9-]{1,63}(?<!-))*$"
        )
        return FlextUtilitiesGuardsEnsure.validate_pattern(
            hostname,
            hostname_pattern,
            field_name,
        )

    @staticmethod
    def validate_length[T: Sized](
        value: T,
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        field_name: str = "value",
    ) -> r[T]:
        try:
            length = len(value)
        except (TypeError, ValueError):
            return r[T].fail(f"{field_name} length is invalid")
        if min_length is not None and length < min_length:
            return r[T].fail(
                f"{field_name} must have at least {min_length} characters/items",
            )
        if max_length is not None and length > max_length:
            return r[T].fail(
                f"{field_name} must have at most {max_length} characters/items",
            )
        return r[T].ok(value)

    @staticmethod
    def validate_pattern(value: str, pattern: str, field_name: str = "value") -> r[str]:
        if re.search(pattern, value) is None:
            return r[str].fail(f"{field_name} has invalid format")
        return r[str].ok(value)

    @staticmethod
    def validate_port_number(port: int, field_name: str = "port") -> r[int]:
        max_port_number = 65535
        if isinstance(port, bool):
            return r[int].fail(f"{field_name} must be an integer")
        if port < 1 or port > max_port_number:
            return r[int].fail(f"{field_name} must be between 1 and {max_port_number}")
        return r[int].ok(port)

    @staticmethod
    def validate_positive(value: float, field_name: str = "value") -> r[t.Numeric]:
        if isinstance(value, bool) or value <= 0:
            return r[t.Numeric].fail(f"{field_name} must be positive")
        return r[t.Numeric].ok(value)

    @staticmethod
    def validate_uri(uri: str, field_name: str = "uri") -> r[str]:
        uri_pattern = "^[a-zA-Z][a-zA-Z0-9+.-]*://[^\\s]+$"
        return FlextUtilitiesGuardsEnsure.validate_pattern(uri, uri_pattern, field_name)

    @staticmethod
    def validate_pydantic_model[T: BaseModel](
        model_class: type[T],
        data: t.ContainerMapping,
    ) -> r[T]:
        try:
            validated = model_class.model_validate(data)
            return r[T].ok(validated)
        except (ValidationError, TypeError, ValueError) as exc:
            return r[T].fail(f"Validation failed: {exc}")


__all__ = ["FlextUtilitiesGuardsEnsure"]
