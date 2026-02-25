from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core.typings import t


class FlextUtilitiesNormalize:
    @staticmethod
    def _normalize_object_mapping(
        value: Mapping[str, t.ConfigMapValue] | Mapping[object, object],
    ) -> dict[str, t.ConfigMapValue]:
        normalized_dict: dict[str, t.ConfigMapValue] = {}
        for raw_key, raw_value in value.items():
            if raw_value is None or isinstance(
                raw_value, str | int | float | bool | datetime | tuple
            ):
                normalized_value: t.ConfigMapValue = raw_value
            elif isinstance(raw_value, list):
                normalized_value = tuple(raw_value)
            elif isinstance(raw_value, Mapping):
                normalized_value = FlextUtilitiesNormalize._normalize_object_mapping(
                    raw_value
                )
            elif hasattr(raw_value, "items") and not isinstance(raw_value, str | bytes):
                text_value = str(raw_value)
                normalized_value = {text_value: text_value}
            else:
                normalized_value = str(raw_value)
            normalized_dict[str(raw_key)] = normalized_value
        return normalized_dict

    @staticmethod
    def sort_key(value: object) -> tuple[str, str]:
        if isinstance(value, str):
            normalized = value.casefold()
            return (normalized, value)
        return ("other", str(value))

    @staticmethod
    def normalize_component(value: t.ConfigMapValue) -> t.ConfigMapValue:
        return FlextUtilitiesCache.normalize_component(value)

    @staticmethod
    def _sort_dict_keys(data: t.ConfigMapValue) -> t.ConfigMapValue:
        if isinstance(data, Mapping):
            data_dict = FlextUtilitiesNormalize._normalize_object_mapping(data)
            sorted_result: Mapping[str, t.ConfigMapValue] = {
                str(k): FlextUtilitiesNormalize._sort_dict_keys(data_dict[k])
                for k in sorted(data_dict.keys(), key=lambda key: str(key).casefold())
            }
            return sorted_result
        return data

    @staticmethod
    def sort_dict_keys(obj: t.ConfigMapValue) -> t.ConfigMapValue:
        if isinstance(obj, Mapping):
            dict_obj = FlextUtilitiesNormalize._normalize_object_mapping(obj)
            sorted_items: list[tuple[str, t.ConfigMapValue]] = sorted(
                dict_obj.items(),
                key=lambda x: str(x[0]),
            )
            return {
                str(k): FlextUtilitiesNormalize._sort_dict_keys(v)
                for k, v in sorted_items
            }
        if isinstance(obj, tuple):
            tuple_items: list[t.ConfigMapValue] = [
                FlextUtilitiesNormalize._sort_dict_keys(
                    item
                    if (
                        isinstance(item, str | int | float | bool | datetime)
                        or item is None
                        or isinstance(item, Mapping | Sequence)
                    )
                    else str(item)
                )
                for item in obj
            ]
            return tuple(tuple_items)
        if isinstance(obj, list):
            return [
                FlextUtilitiesNormalize._sort_dict_keys(
                    item
                    if (
                        isinstance(item, str | int | float | bool | datetime)
                        or item is None
                        or isinstance(item, Mapping | Sequence)
                    )
                    else str(item)
                )
                for item in obj
            ]
        return obj


__all__ = ["FlextUtilitiesNormalize"]
