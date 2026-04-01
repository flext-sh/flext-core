"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

from pydantic import BaseModel

from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core.typings import t


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def is_object_list(
        value: t.NormalizedValue,
    ) -> TypeIs[t.ContainerList]:
        """Narrow value to list of normalized values."""
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: t.GuardInput,
    ) -> TypeIs[tuple[t.NormalizedValue, ...]]:
        """Narrow value to tuple of normalized values."""
        return isinstance(value, tuple)

    @staticmethod
    def is_configuration_dict(
        value: t.ValueOrModel,
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
        value: t.ContainerMapping | t.ConfigMap | t.Dict,
    ) -> TypeIs[t.ConfigMap]:
        """Check if value is a ConfigMap/Dict or mapping with container values."""
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
    def is_pydantic_model(value: t.ValueOrModel) -> TypeIs[BaseModel]:
        """Narrow value to Pydantic BaseModel with callable model_dump."""
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__ = ["FlextUtilitiesGuardsTypeModel"]
