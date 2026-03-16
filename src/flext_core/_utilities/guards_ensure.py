from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized

from flext_core import p, r, t
from flext_core._utilities.guards_type import FlextUtilitiesGuardsType


class FlextUtilitiesGuardsEnsure(FlextUtilitiesGuardsType):
    @staticmethod
    def _ensure_to_dict(
        value: t.NormalizedValue | None, default: Mapping[str, t.NormalizedValue] | None
    ) -> Mapping[str, t.NormalizedValue]:
        if value is None:
            return default if default is not None else {}
        if isinstance(value, Mapping):
            normalized: dict[str, t.NormalizedValue] = {}
            for key, item_value in value.items():
                normalized[str(key)] = item_value
            return normalized
        return {"value": value}

    @staticmethod
    def _ensure_to_list(
        value: t.NormalizedValue | list[t.NormalizedValue] | None,
        default: list[t.NormalizedValue] | None,
    ) -> list[t.NormalizedValue]:
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _guard_check_condition[T: FlextUtilitiesGuardsType._GuardInput](
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
                    value, condition, context_name, error_msg
                )
            return (
                error_msg
                or f"{context_name} must be {condition.__name__}, got {type(value).__name__}"
            )
        if FlextUtilitiesGuardsEnsure._is_type_tuple(condition):
            if FlextUtilitiesGuardsEnsure.is_container(value):
                return FlextUtilitiesGuardsEnsure._guard_check_type(
                    value, condition, context_name, error_msg
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
            return FlextUtilitiesGuardsEnsure._guard_check_validator(
                value, condition, context_name, error_msg
            )
        if isinstance(condition, str):
            if not FlextUtilitiesGuardsEnsure.is_container(value):
                return (
                    error_msg or f"{context_name} must be a valid configuration value"
                )
            return FlextUtilitiesGuardsEnsure._guard_check_string_shortcut(
                value, condition, context_name, error_msg
            )
        if callable(condition):
            return FlextUtilitiesGuardsEnsure._guard_check_predicate(
                value, condition, context_name, error_msg
            )
        return error_msg or f"{context_name} invalid guard condition type"

    @staticmethod
    def _guard_check_predicate(
        value: FlextUtilitiesGuardsType._GuardInput,
        condition: Callable[..., FlextUtilitiesGuardsType._GuardInput | bool],
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
            return (
                f"{context_name} guard check raised: {e}"
                if error_msg is None
                else error_msg
            )
        return ""

    @staticmethod
    def _guard_check_string_shortcut(
        value: t.NormalizedValue,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        shortcut_lower = condition.lower()
        if shortcut_lower == "non_empty":
            if (
                (isinstance(value, str) and bool(value))
                or (isinstance(value, list) and len(value) > 0)
                or (isinstance(value, dict) and len(value) > 0)
            ):
                return ""
            return error_msg or f"{context_name} must be non-empty"
        if shortcut_lower == "positive":
            if (
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and (value > 0)
            ):
                return ""
            return error_msg or f"{context_name} must be positive number"
        if shortcut_lower == "non_negative":
            if (
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and (value >= 0)
            ):
                return ""
            return error_msg or f"{context_name} must be non-negative number"
        if shortcut_lower == "dict":
            return (
                ""
                if hasattr(value, "items") and (not isinstance(value, (str, bytes)))
                else error_msg or f"{context_name} must be dict-like"
            )
        if shortcut_lower == "list":
            if (
                hasattr(value, "__iter__")
                and (not isinstance(value, (str, bytes)))
                and (not hasattr(value, "items"))
            ):
                return ""
            return error_msg or f"{context_name} must be list-like"
        if shortcut_lower == "string":
            return (
                ""
                if isinstance(value, str)
                else error_msg or f"{context_name} must be string"
            )
        if shortcut_lower == "int":
            return (
                ""
                if isinstance(value, int) and (not isinstance(value, bool))
                else error_msg or f"{context_name} must be int"
            )
        if shortcut_lower == "float":
            return (
                ""
                if isinstance(value, int | float) and (not isinstance(value, bool))
                else error_msg or f"{context_name} must be float"
            )
        if shortcut_lower == "bool":
            return (
                ""
                if isinstance(value, bool)
                else error_msg or f"{context_name} must be bool"
            )
        return error_msg or f"{context_name} unknown guard shortcut: {condition}"

    @staticmethod
    def _guard_check_type(
        value: t.NormalizedValue,
        condition: type | tuple[type, ...],
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if isinstance(value, condition):
            return ""
        if error_msg is not None:
            return error_msg
        type_name = (
            condition.__name__
            if isinstance(condition, type)
            else " | ".join(c.__name__ for c in condition)
        )
        return f"{context_name} must be {type_name}, got {value.__class__.__name__}"

    @staticmethod
    def _guard_check_validator(
        value: t.NormalizedValue,
        condition: p.ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        if not FlextUtilitiesGuardsEnsure.is_container(value):
            return error_msg or f"{context_name} must be a valid configuration value"
        if condition(value):
            return ""
        if error_msg is not None:
            return error_msg
        desc = (
            getattr(condition, "description", "validation")
            if hasattr(condition, "description")
            else "validation"
        )
        return f"{context_name} failed {desc} check"

    @staticmethod
    def _guard_handle_failure[T](
        error_message: str, *, return_value: bool, default: T | None
    ) -> r[T] | T:
        if return_value:
            return default if default is not None else r[T].fail(error_message)
        return r[T].ok(default) if default is not None else r[T].fail(error_message)

    @staticmethod
    def chk(
        value: t.NormalizedValue,
        *,
        eq: t.NormalizedValue | None = None,
        ne: t.NormalizedValue | None = None,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
        is_: type | None = None,
        not_: type | None = None,
        in_: Sequence[t.NormalizedValue] | None = None,
        not_in: Sequence[t.NormalizedValue] | None = None,
        none: bool | None = None,
        empty: bool | None = None,
        match: str | None = None,
        contains: t.NormalizedValue | None = None,
        starts: str | None = None,
        ends: str | None = None,
    ) -> bool:
        if none is True and value is not None:
            return False
        if none is False and value is None:
            return False
        if is_ is not None and (not isinstance(value, is_)):
            return False
        if not_ is not None and isinstance(value, not_):
            return False
        if eq is not None and value != eq:
            return False
        if ne is not None and value == ne:
            return False
        if in_ is not None and value not in in_:
            return False
        if not_in is not None and value in not_in:
            return False
        check_val: int | float = 0
        if isinstance(value, (int, float)):
            check_val = value
        elif isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            check_val = len(value)
        elif hasattr(value, "__len__"):
            try:
                len_method = getattr(value, "__len__", None)
                if callable(len_method):
                    length = len_method()
                    if isinstance(length, int):
                        check_val = length
            except (TypeError, AttributeError):
                check_val = 0
        if gt is not None and check_val <= gt:
            return False
        if gte is not None and check_val < gte:
            return False
        if lt is not None and check_val >= lt:
            return False
        if lte is not None and check_val > lte:
            return False
        if empty is True and check_val != 0:
            return False
        if empty is False and check_val == 0:
            return False
        if isinstance(value, str):
            if match is not None and (not re.search(match, value)):
                return False
            if starts is not None and (not value.startswith(starts)):
                return False
            if ends is not None and (not value.endswith(ends)):
                return False
            if (
                contains is not None
                and isinstance(contains, str)
                and (contains not in value)
            ):
                return False
        elif contains is not None:
            if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
                found = False
                iterable_value: Iterable[t.NormalizedValue] = value
                for item in iterable_value:
                    if item == contains:
                        found = True
                        break
                if not found:
                    return False
            else:
                return False
        return True

    @staticmethod
    def empty(items: t.NormalizedValue | None) -> bool:
        if items is None:
            return True
        if isinstance(items, Sized):
            return len(items) == 0
        return not bool(items)

    @staticmethod
    def ensure(
        value: t.NormalizedValue,
        *,
        target_type: str = "auto",
        default: str | list[t.NormalizedValue] | t.NormalizedValue | None = None,
    ) -> (
        str
        | list[t.NormalizedValue]
        | t.NormalizedValue
        | Mapping[str, t.NormalizedValue]
    ):
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
            str_list_default: list[str] | None = None
            if isinstance(default, list):
                str_list_default = [str(item) for item in default]
            if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                return list(value)
            if value is None:
                return list(str_list_default) if str_list_default else []
            return [value]
        if target_type == "dict":
            dict_default = default if isinstance(default, Mapping) else None
            return FlextUtilitiesGuardsEnsure._ensure_to_dict(value, dict_default)
        if target_type == "auto" and isinstance(value, Mapping):
            normalized_auto: dict[str, t.NormalizedValue] = {}
            for key, item_value in value.items():
                normalized_auto[str(key)] = item_value
            return normalized_auto
        list_default: list[t.NormalizedValue] | None = (
            default if FlextUtilitiesGuardsEnsure.is_object_list(default) else None
        )
        return FlextUtilitiesGuardsEnsure._ensure_to_list(value, list_default)

    @staticmethod
    def extract_mapping_or_none(value: t.NormalizedValue) -> r[t.ConfigMap]:
        if isinstance(
            value, Mapping
        ) and FlextUtilitiesGuardsEnsure.is_configuration_mapping(value):
            return r[t.ConfigMap].ok(value)
        return r[t.ConfigMap].fail("Value is not a configuration mapping")

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
        guarded_value: t.NormalizedValue = value
        try:
            if isinstance(validator, type):
                if isinstance(value, validator):
                    return (
                        guarded_value
                        if return_value
                        and FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                        else str(guarded_value)
                        if return_value
                        else True
                    )
            elif FlextUtilitiesGuardsEnsure.is_object_tuple(validator):
                tuple_types = tuple(
                    item for item in validator if isinstance(item, type)
                )
                if len(tuple_types) == len(validator) and isinstance(
                    value, tuple_types
                ):
                    return (
                        guarded_value
                        if return_value
                        and FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                        else str(guarded_value)
                        if return_value
                        else True
                    )
            elif callable(validator):
                if validator(value):
                    return (
                        guarded_value
                        if return_value
                        and FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                        else str(guarded_value)
                        if return_value
                        else True
                    )
            elif value:
                return (
                    guarded_value
                    if return_value
                    and FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                    else str(guarded_value)
                    if return_value
                    else True
                )
            if default is not None:
                return (
                    default
                    if FlextUtilitiesGuardsEnsure.is_container(default)
                    else str(default)
                )
            return (
                r[t.Container].fail("Guard validation failed")
                if return_value
                else False
            )
        except (TypeError, ValueError, AttributeError):
            if default is not None:
                return (
                    default
                    if FlextUtilitiesGuardsEnsure.is_container(default)
                    else str(default)
                )
            return (
                r[t.Container].fail("Guard validation raised an exception")
                if return_value
                else False
            )

    @staticmethod
    def guard_result[T: FlextUtilitiesGuardsType._GuardInput](
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
        if len(conditions) == 0:
            if bool(value):
                return value if return_value else r[T].ok(value)
            failure_message = error_message or f"{context_name} guard failed"
            return FlextUtilitiesGuardsEnsure._guard_handle_failure(
                failure_message, return_value=return_value, default=default
            )
        for condition in conditions:
            condition_error = FlextUtilitiesGuardsEnsure._guard_check_condition(
                value, condition, context_name, error_message
            )
            if condition_error:
                return FlextUtilitiesGuardsEnsure._guard_handle_failure(
                    condition_error, return_value=return_value, default=default
                )
        return value if return_value else r[T].ok(value)


__all__ = ["FlextUtilitiesGuardsEnsure"]
