"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuardsTypeCore, t


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def is_object_list(
        value: t.RecursiveContainer,
    ) -> TypeIs[t.RecursiveContainerList]:
        """Narrow value to a recursive container list."""
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[t.RecursiveContainer, ...]]:
        """Narrow value to a recursive container tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def is_configuration_dict(
        value: t.GuardInput,
    ) -> TypeIs[t.Dict]:
        """Check if value is a Dict model or mapping with container values."""
        if isinstance(value, t.Dict):
            for item_value in value.root.values():
                if isinstance(
                    item_value,
                    BaseModel,
                ) or not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                    return False
            return True
        return isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)

    @staticmethod
    def is_configuration_mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.ConfigMap]:
        """Check if value is a ConfigMap/Dict or mapping with container values."""
        if not isinstance(value, (t.ConfigMap, t.Dict, Mapping)):
            return False
        candidate: Mapping[str, t.ValueOrModel] = (
            value.root if isinstance(value, (t.ConfigMap, t.Dict)) else value
        )
        for item_value in candidate.values():
            if isinstance(
                item_value,
                BaseModel,
            ) or not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                return False
        return True

    @staticmethod
    def is_pydantic_model(value: t.GuardInput) -> TypeIs[BaseModel]:
        """Narrow value to Pydantic BaseModel with callable model_dump."""
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__ = ["FlextUtilitiesGuardsTypeModel"]
