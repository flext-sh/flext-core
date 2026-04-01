"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core.result import FlextResult as r
from flext_core.typings import t


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    @staticmethod
    def _normalize_model_input(
        data: BaseModel | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> Mapping[str, t.ValueOrModel]:
        if isinstance(
            data, BaseModel
        ) and FlextUtilitiesGuardsTypeModel.is_pydantic_model(data):
            root_value = getattr(data, "root", None)
            if FlextUtilitiesGuardsTypeCore.is_mapping(root_value):
                return {str(key): value for key, value in root_value.items()}
            dumped = data.model_dump()
            return {str(key): value for key, value in dumped.items()}
        return {str(key): value for key, value in data.items()}

    @staticmethod
    def dump(
        model: BaseModel,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> t.ScalarMapping:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            by_alias: Whether to use field aliases.
            exclude_none: Whether to exclude None values.
            exclude_unset: Whether to exclude unset values.
            exclude_defaults: Whether to exclude default values.
            include: Set of field names to include.
            exclude: Set of field names to exclude.

        Returns:
            Dictionary representation of the model.

        """
        return model.model_dump(
            by_alias=by_alias,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            include=include,
            exclude=exclude,
        )

    @classmethod
    def load[M: BaseModel](
        cls,
        model_cls: type[M],
        data: BaseModel | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> r[M]:
        """Load a model from a mapping-like input using Pydantic validation."""
        return r[M].create_from_callable(
            lambda: model_cls.model_validate(cls._normalize_model_input(data)),
        )


__all__ = ["FlextUtilitiesModel"]
