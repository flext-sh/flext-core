"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import TypeGuard, TypeIs

from flext_core import (
    FlextModelsContainers,
    FlextModelsPydantic,
    FlextUtilitiesGuardsTypeCore,
    p,
    t,
)


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def base_model(value: t.GuardInput | p.Model) -> TypeIs[t.DomainModelCarrier]:
        """Narrow a broad runtime value to the canonical domain model carrier alias.

        Returns TypeIs[t.DomainModelCarrier] so that on the False branch both
        BaseModel and p.Model protocol instances are narrowed out, preventing
        pyright reportGeneralTypeIssues on downstream isinstance checks.
        """
        return isinstance(value, FlextModelsPydantic.BaseModel)

    @staticmethod
    def model_type(
        value: t.TypeHintSpecifier,
    ) -> TypeIs[t.ModelClass[t.ModelCarrier]]:
        """Narrow a runtime value to a canonical Pydantic model class."""
        return isinstance(value, type) and issubclass(
            value,
            FlextModelsPydantic.BaseModel,
        )

    @staticmethod
    def object_list(
        value: t.GuardInput,
    ) -> TypeIs[t.ObjectList]:
        """Narrow value to a container list."""
        return isinstance(value, list)

    @staticmethod
    def object_tuple(
        value: t.GuardInput | Callable[[t.Container], bool] | None,
    ) -> TypeIs[tuple[t.Container, ...]]:
        """Narrow value to a container tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def configuration_dict(
        value: t.GuardInput,
    ) -> TypeGuard[FlextModelsContainers.Dict]:
        """Check if value is a Dict model or mapping with container values."""
        if isinstance(value, FlextModelsContainers.Dict):
            for item_value in value.root.values():
                if isinstance(
                    item_value,
                    FlextModelsPydantic.BaseModel,
                ) or not FlextUtilitiesGuardsTypeCore.container(item_value):
                    return False
            return True
        return FlextUtilitiesGuardsTypeCore.mapping(value)

    @staticmethod
    def configuration_mapping(
        value: t.GuardInput,
    ) -> TypeGuard[FlextModelsContainers.ConfigMap]:
        """Check if value is a ConfigMap/Dict or mapping with container values."""
        if isinstance(
            value, (FlextModelsContainers.ConfigMap, FlextModelsContainers.Dict)
        ):
            for item_value in value.root.values():
                if isinstance(
                    item_value,
                    FlextModelsPydantic.BaseModel,
                ) or not FlextUtilitiesGuardsTypeCore.container(item_value):
                    return False
            return True
        return FlextUtilitiesGuardsTypeCore.mapping(value)

    @staticmethod
    def pydantic_model(
        value: t.GuardInput | p.Model | object | None,
    ) -> TypeIs[t.ModelCarrier]:
        """Narrow value to the canonical Pydantic model carrier."""
        return (
            isinstance(value, FlextModelsPydantic.BaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__: list[str] = ["FlextUtilitiesGuardsTypeModel"]
