"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs, cast

from flext_core import FlextUtilitiesGuardsTypeCore, t
from flext_core._models.pydantic import FlextModelsPydantic


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def base_model(value: object) -> TypeIs[t.ModelCarrier]:
        """Narrow a broad runtime value to the canonical model carrier alias."""
        return isinstance(value, FlextModelsPydantic.BaseModel)

    @staticmethod
    def model_type(value: object) -> TypeIs[t.ModelClass[t.ModelCarrier]]:
        """Narrow a runtime value to a canonical Pydantic model class."""
        return isinstance(value, type) and issubclass(
            value,
            FlextModelsPydantic.BaseModel,
        )

    @staticmethod
    def object_list(
        value: t.RecursiveContainer,
    ) -> TypeIs[t.RecursiveContainerList]:
        """Narrow value to a recursive container list."""
        return isinstance(value, list)

    @staticmethod
    def object_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[t.RecursiveContainer, ...]]:
        """Narrow value to a recursive container tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def configuration_dict(
        value: t.GuardInput,
    ) -> TypeIs[t.Dict]:
        """Check if value is a Dict model or mapping with container values."""
        if isinstance(value, t.Dict):
            for item_value in value.root.values():
                if isinstance(
                    item_value,
                    FlextModelsPydantic.BaseModel,
                ) or not FlextUtilitiesGuardsTypeCore.container(item_value):
                    return False
            return True
        return isinstance(
            value,
            Mapping,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(
            cast("Mapping[str, t.Container]", value),
        )

    @staticmethod
    def configuration_mapping(
        value: t.GuardInput,
    ) -> TypeIs[t.ConfigMap]:
        """Check if value is a ConfigMap/Dict or mapping with container values."""
        if not isinstance(value, (t.ConfigMap, t.Dict, Mapping)):
            return False
        candidate: Mapping[str, t.ValueOrModel] = (
            value.root
            if isinstance(value, (t.ConfigMap, t.Dict))
            else cast("Mapping[str, t.ValueOrModel]", value)
        )
        for item_value in candidate.values():
            if isinstance(
                item_value,
                FlextModelsPydantic.BaseModel,
            ) or not FlextUtilitiesGuardsTypeCore.container(item_value):
                return False
        return True

    @staticmethod
    def pydantic_model(value: object) -> TypeIs[t.ModelCarrier]:
        """Narrow value to the canonical Pydantic model carrier."""
        return (
            isinstance(value, FlextModelsPydantic.BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__: list[str] = ["FlextUtilitiesGuardsTypeModel"]
