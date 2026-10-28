"""Pydantic and data model type guard implementations for Flext core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, TypeIs

from pydantic import BaseModel as PydanticBaseModel

from flext_core import m, p, t

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextUtilitiesGuardsTypeModel:
    """Pydantic and data model type guards."""

    @staticmethod
    def has_model_dump(
        value: p.GuardInput | p.HasModelDump | p.Model | t.JsonValue | None,
    ) -> TypeIs[p.HasModelDump]:
        """Narrow value to objects exposing a callable ``model_dump``.

        Returns:
            The resulting ``TypeIs[pr.HasModelDump]``.

        """
        model_dump = getattr(value, "model_dump", None)
        return callable(model_dump)

    @staticmethod
    def model_type(
        value: t.TypeHintSpecifier,
    ) -> TypeIs[t.ModelClass[m.BaseModel]]:
        """Narrow a runtime value to a canonical Pydantic model class.

        Returns:
            The resulting ``TypeIs[t.ModelClass[mp.BaseModel]]``.

        """
        return isinstance(value, type) and issubclass(value, PydanticBaseModel)

    @staticmethod
    def object_tuple(
        value: p.GuardInput | Callable[[t.JsonValue], bool] | None,
    ) -> TypeIs[tuple[t.JsonValue, ...]]:
        """Narrow value to a container tuple.

        Returns:
            The resulting ``TypeIs[tuple[t.JsonValue, ...]]``.

        """
        return isinstance(value, tuple)

    @staticmethod
    def pydantic_model(
        value: p.GuardInput | p.Model | t.JsonValue | PydanticBaseModel | None,
    ) -> TypeIs[m.BaseModel]:
        """Narrow value to the canonical Pydantic model carrier.

        Accepts both ``m.BaseModel`` and
        ``m.RootModel`` subclasses — they share
        ``PydanticBaseModel`` as a common ancestor and both expose
        ``model_dump`` / ``model_validate``.

        Returns:
            The resulting ``TypeIs[mp.BaseModel]``.

        from pydantic import BaseModel as PydanticBaseModel

        """
        return (
            isinstance(value, PydanticBaseModel)
            and hasattr(value, "model_dump")
            and callable(value.model_dump)
        )


__all__: MutableSequence[str] = ["FlextUtilitiesGuardsTypeModel"]
