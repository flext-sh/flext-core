"""Pydantic and data model type guard implementations for Flext core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

from pydantic import BaseModel as PydanticBaseModel

from flext_core import t

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._models.pydantic import FlextModelsPydantic as mp
    from .._protocols.base import FlextProtocolsBase as pb
    from .._protocols.result import FlextProtocolsResult as pr


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def has_model_dump(
        value: t.GuardInput | pr.HasModelDump | pb.Model | t.JsonValue | None,
    ) -> TypeIs[pr.HasModelDump]:
        """Narrow value to objects exposing a callable ``model_dump``.

        Returns:
            The resulting ``TypeIs[pr.HasModelDump]``.

        """
        model_dump = getattr(value, "model_dump", None)
        return callable(model_dump)

    @staticmethod
    def model_type(value: t.TypeHintSpecifier) -> TypeIs[t.ModelClass[mp.BaseModel]]:
        """Narrow a runtime value to a canonical Pydantic model class.

        Returns:
            The resulting ``TypeIs[t.ModelClass[mp.BaseModel]]``.

        """
        return isinstance(value, type) and issubclass(value, PydanticBaseModel)

    @staticmethod
    def object_tuple(
        value: t.GuardInput | Callable[[t.JsonValue], bool] | None,
    ) -> TypeIs[t.VariadicTuple[t.JsonValue]]:
        """Narrow value to a container tuple.

        Returns:
            The resulting ``TypeIs[t.VariadicTuple[t.JsonValue]]``.

        """
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

        Returns:
            The resulting ``TypeIs[mp.BaseModel]``.

        """
        return (
            isinstance(value, PydanticBaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesGuardsTypeModel"]
