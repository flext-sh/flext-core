"""Pydantic and data model type guard implementations for Flext core."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

from pydantic import BaseModel as PydanticBaseModel

from flext_core import FlextTypes as t

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._models.pydantic import FlextModelsPydantic as mp
    from flext_core._protocols.base import FlextProtocolsBase as pb
    from flext_core._protocols.result import FlextProtocolsResult as pr


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def has_model_dump(
        value: t.GuardInput | pr.HasModelDump | pb.Model | t.JsonValue | None,
    ) -> TypeIs[pr.HasModelDump]:
        """Narrow value to objects exposing a callable ``model_dump``."""
        model_dump = getattr(value, "model_dump", None)
        return callable(model_dump)

    @staticmethod
    def model_type(value: t.TypeHintSpecifier) -> TypeIs[t.ModelClass[mp.BaseModel]]:
        """Narrow a runtime value to a canonical Pydantic model class."""
        return isinstance(value, type) and issubclass(value, PydanticBaseModel)

    @staticmethod
    def object_tuple(
        value: t.GuardInput | Callable[[t.JsonValue], bool] | None,
    ) -> TypeIs[t.VariadicTuple[t.JsonValue]]:
        """Narrow value to a container tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def pydantic_model(
        value: t.GuardInput | pb.Model | t.JsonValue | PydanticBaseModel | None,
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
