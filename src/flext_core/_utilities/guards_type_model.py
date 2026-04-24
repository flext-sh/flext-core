"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import TypeIs

from pydantic import BaseModel as PydanticBaseModel

from flext_core import (
    FlextModelsPydantic as mp,
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
        pyright reportGeneralTypeIssues on downstream isinstance checks. Uses
        PydanticBaseModel (the common ancestor of both BaseModel and
        RootModel facades) so that either flavour narrows correctly.
        """
        return isinstance(value, PydanticBaseModel)

    @staticmethod
    def model_type(
        value: t.TypeHintSpecifier,
    ) -> TypeIs[t.ModelClass[mp.BaseModel]]:
        """Narrow a runtime value to a canonical Pydantic model class."""
        return isinstance(value, type) and issubclass(
            value,
            PydanticBaseModel,
        )

    @staticmethod
    def object_tuple(
        value: t.GuardInput | Callable[[t.JsonValue], bool] | None,
    ) -> TypeIs[t.VariadicTuple[t.JsonValue]]:
        """Narrow value to a container tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def pydantic_model(
        value: t.GuardInput | p.Model | t.JsonValue | PydanticBaseModel | None,
    ) -> TypeIs[mp.BaseModel]:
        """Narrow value to the canonical Pydantic model carrier.

        Accepts both ``FlextModelsPydantic.BaseModel`` and
        ``FlextModelsPydantic.RootModel`` subclasses — they share
        ``PydanticBaseModel`` as a common ancestor and both expose
        ``model_dump`` / ``model_validate``.
        """
        return (
            isinstance(value, PydanticBaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesGuardsTypeModel"]
