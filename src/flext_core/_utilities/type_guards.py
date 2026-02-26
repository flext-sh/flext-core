from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeIs, cast

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from flext_core.runtime import FlextRuntime
from flext_core.typings import t

type GuardCandidate = (
    t.GeneralValueType
    | BaseSettings
    | Enum
    | FlextRuntime.RuntimeResult[t.GeneralValueType]
)


class FlextUtilitiesTypeGuards:
    @staticmethod
    def _is_general_value(value: GuardCandidate) -> bool:
        if value is None or isinstance(value, str | int | float | bool | datetime):
            return True
        if isinstance(value, BaseModel | Path):
            return True
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            return all(
                FlextUtilitiesTypeGuards._is_general_value(item) for item in value
            )
        if isinstance(value, Mapping):
            mapping_value = cast("Mapping[t.GeneralValueType, GuardCandidate]", value)
            return all(
                isinstance(key, str)
                and FlextUtilitiesTypeGuards._is_general_value(item)
                for key, item in mapping_value.items()
            )
        return False

    @staticmethod
    def is_str(value: GuardCandidate) -> TypeIs[str]:
        return isinstance(value, str)

    @staticmethod
    def is_int(value: GuardCandidate) -> TypeIs[int]:
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def is_float(value: GuardCandidate) -> TypeIs[float]:
        return isinstance(value, float)

    @staticmethod
    def is_bool(value: GuardCandidate) -> TypeIs[bool]:
        return isinstance(value, bool)

    @staticmethod
    def is_none(value: GuardCandidate) -> TypeIs[None]:
        return value is None

    @staticmethod
    def is_list(value: GuardCandidate) -> TypeIs[list[t.GeneralValueType]]:
        return isinstance(value, list) and all(
            FlextUtilitiesTypeGuards._is_general_value(item) for item in value
        )

    @staticmethod
    def is_dict(value: GuardCandidate) -> TypeIs[dict[str, t.GeneralValueType]]:
        if not isinstance(value, dict):
            return False
        dict_value = cast("dict[t.GeneralValueType, GuardCandidate]", value)
        return all(
            isinstance(key, str) and FlextUtilitiesTypeGuards._is_general_value(item)
            for key, item in dict_value.items()
        )

    @staticmethod
    def is_sequence(value: GuardCandidate) -> TypeIs[Sequence[t.GeneralValueType]]:
        return (
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes | bytearray)
            and all(FlextUtilitiesTypeGuards._is_general_value(item) for item in value)
        )

    @staticmethod
    def is_mapping(value: GuardCandidate) -> TypeIs[Mapping[str, t.GeneralValueType]]:
        if not isinstance(value, Mapping):
            return False
        mapping_value = cast("Mapping[t.GeneralValueType, GuardCandidate]", value)
        return all(
            isinstance(key, str) and FlextUtilitiesTypeGuards._is_general_value(item)
            for key, item in mapping_value.items()
        )

    @staticmethod
    def is_base_model(value: GuardCandidate) -> TypeIs[BaseModel]:
        return isinstance(value, BaseModel)

    @staticmethod
    def is_base_settings(value: GuardCandidate) -> TypeIs[BaseSettings]:
        return isinstance(value, BaseSettings)

    @staticmethod
    def is_result(
        value: GuardCandidate,
    ) -> TypeIs[FlextRuntime.RuntimeResult[t.GeneralValueType]]:
        return isinstance(value, FlextRuntime.RuntimeResult)

    @staticmethod
    def is_path(value: GuardCandidate) -> TypeIs[Path]:
        return isinstance(value, Path)

    @staticmethod
    def is_datetime(value: GuardCandidate) -> TypeIs[datetime]:
        return isinstance(value, datetime)

    @staticmethod
    def is_enum(value: GuardCandidate) -> TypeIs[Enum]:
        return isinstance(value, Enum)

    @staticmethod
    def is_non_empty_str(value: GuardCandidate) -> TypeIs[str]:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def is_positive_int(value: GuardCandidate) -> TypeIs[int]:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0


__all__ = ["FlextUtilitiesTypeGuards"]
