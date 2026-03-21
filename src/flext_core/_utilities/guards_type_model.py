from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuardsTypeCore, p, t


class FlextUtilitiesGuardsTypeModel:
    @staticmethod
    def is_object_list(
        value: object,
    ) -> TypeIs[list[t.NormalizedValue]]:
        return isinstance(value, list)

    @staticmethod
    def is_object_tuple(
        value: object,
    ) -> TypeIs[tuple[t.NormalizedValue, ...]]:
        return isinstance(value, tuple)

    @staticmethod
    def is_config_value(value: t.NormalizedValue) -> TypeIs[t.NormalizedValue]:
        if value is None or FlextUtilitiesGuardsTypeCore.is_scalar(value):
            return True
        if isinstance(value, (list, tuple)):
            for item in value:
                if not (item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)):
                    return False
            return True
        if isinstance(value, Mapping):
            for item in value.values():
                if not (item is None or FlextUtilitiesGuardsTypeCore.is_scalar(item)):
                    return False
            return True
        return False

    @staticmethod
    def is_configuration_dict(
        value: object,
    ) -> TypeIs[t.Dict]:
        if isinstance(value, t.Dict):
            for item_value in value.root.values():
                if not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                    return False
            return True
        return FlextUtilitiesGuardsTypeCore.is_mapping(
            value,
        ) and FlextUtilitiesGuardsTypeCore.all_container_mapping_values(value)

    @staticmethod
    def is_configuration_mapping(
        value: Mapping[str, t.NormalizedValue] | t.ConfigMap | t.Dict,
    ) -> TypeIs[t.ConfigMap]:
        candidate: Mapping[str, t.ValueOrModel] = (
            value.root if isinstance(value, (t.ConfigMap, t.Dict)) else value
        )
        for item_value in candidate.values():
            if not FlextUtilitiesGuardsTypeCore.is_container(item_value):
                return False
        return True

    @staticmethod
    def is_pydantic_model(value: object) -> TypeIs[p.HasModelDump]:
        return (
            isinstance(value, BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__ = ["FlextUtilitiesGuardsTypeModel"]
