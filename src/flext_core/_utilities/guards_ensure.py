"""Type guard helpers with defensive assertions for normalized runtime data."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized

from pydantic import BaseModel, ValidationError

from flext_core import FlextUtilitiesGuardsType, m, p, r, t


class FlextUtilitiesGuardsEnsure(FlextUtilitiesGuardsType):
    """Ensure-style guard utility methods for data validation and normalization."""

    @staticmethod
    def _is_general_value_list(
        value: t.NormalizedValue,
    ) -> bool:
        """Check if input value is a general list type.

        Args:
            value: The value to check. (t.NormalizedValue): The normalized value to be checked.

        Returns:
            bool: True if value is a general list type, False otherwise.

        Raises:
            None

        """
        return isinstance(value, list)

    @staticmethod
    def _ensure_to_dict(
        value: t.NormalizedValue | None,
        default: t.ContainerMapping | None,
    ) -> t.ContainerMapping:
        if value is None:
            return default if default is not None else {}
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
            return default if default is not None else []
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
    def _guard_check_string_shortcut(
        value: t.NormalizedValue,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str:
        shortcut_lower = condition.lower()
        if shortcut_lower == "non_empty":
            if isinstance(value, str) and bool(value):
                return ""
            if isinstance(value, list) and value:
                return ""
            if isinstance(value, dict) and value:
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
            if hasattr(value, "items") and (not isinstance(value, (str, bytes))):
                return ""
            return error_msg or f"{context_name} must be dict-like"
        if shortcut_lower == "list":
            if (
                hasattr(value, "__iter__")
                and (not isinstance(value, (str, bytes)))
                and (not hasattr(value, "items"))
            ):
                return ""
            return error_msg or f"{context_name} must be list-like"
        if shortcut_lower == "string":
            if isinstance(value, str):
                return ""
            return error_msg or f"{context_name} must be string"
        if shortcut_lower == "int":
            if isinstance(value, int) and (not isinstance(value, bool)):
                return ""
            return error_msg or f"{context_name} must be int"
        if shortcut_lower == "float":
            if isinstance(value, (int, float)) and (not isinstance(value, bool)):
                return ""
            return error_msg or f"{context_name} must be float"
        if shortcut_lower == "bool":
            if isinstance(value, bool):
                return ""
            return error_msg or f"{context_name} must be bool"
        return error_msg or f"{context_name} unknown guard shortcut: {condition}"

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

    @staticmethod
    def chk(
        value: t.NormalizedValue,
        spec: m.GuardCheckSpec | None = None,
        **criteria: t.GuardInput,
    ) -> bool:
        guard_spec = spec if spec is not None else m.GuardCheckSpec()
        if criteria:
            criteria_spec = m.GuardCheckSpec.model_validate(criteria)
            guard_spec = guard_spec.model_copy(update=criteria_spec.model_dump())
        eq = guard_spec.eq
        ne = guard_spec.ne
        gt = guard_spec.gt
        gte = guard_spec.gte
        lt = guard_spec.lt
        lte = guard_spec.lte
        is_ = guard_spec.is_
        not_ = guard_spec.not_
        in_ = guard_spec.in_
        not_in = guard_spec.not_in
        none = guard_spec.none
        empty = guard_spec.empty
        match = guard_spec.match
        contains = guard_spec.contains
        starts = guard_spec.starts
        ends = guard_spec.ends
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
        check_val: t.Numeric = 0
        if isinstance(value, (int, float)):
            check_val = value
        elif isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
            sized_value: Sized = value
            check_val = len(sized_value)
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
                iterable_value: Iterable[t.NormalizedValue] = value
                found = False
                for item in iterable_value:
                    item_value: t.NormalizedValue = item
                    if item_value == contains:
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
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif FlextUtilitiesGuardsEnsure.is_object_tuple(validator):
                tuple_types = tuple(
                    item for item in validator if isinstance(item, type)
                )
                if len(tuple_types) == len(validator) and isinstance(
                    value,
                    tuple_types,
                ):
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif callable(validator):
                if validator(value):
                    if return_value:
                        return (
                            guarded_value
                            if FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                            else str(guarded_value)
                        )
                    return True
            elif value:
                if return_value:
                    return (
                        guarded_value
                        if FlextUtilitiesGuardsEnsure.is_container(guarded_value)
                        else str(guarded_value)
                    )
                return True
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
