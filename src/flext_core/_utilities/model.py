"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuards, r, t
from flext_core._models.base import FlextModelFoundation
from flext_core._utilities.args import FlextUtilitiesArgs


class ModelDumpOptions(FlextModelFoundation.FlexibleInternalModel):
    by_alias: bool | None = None
    exclude_none: bool | None = None
    exclude_unset: bool | None = None
    exclude_defaults: bool | None = None
    include: set[str] | None = None
    exclude: set[str] | None = None


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    @staticmethod
    def _normalize_model_input(
        data: BaseModel | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> Mapping[str, t.ValueOrModel]:
        if isinstance(data, BaseModel) and FlextUtilitiesGuards.is_pydantic_model(data):
            root_value = getattr(data, "root", None)
            if FlextUtilitiesGuards.is_mapping(root_value):
                return {str(key): value for key, value in root_value.items()}
            dumped = data.model_dump()
            return {str(key): value for key, value in dumped.items()}
        return {str(key): value for key, value in data.items()}

    @staticmethod
    def dump(
        model: BaseModel,
        options: ModelDumpOptions | None = None,
        **kwargs: object,
    ) -> t.ScalarMapping:
        """Unified Pydantic serialization with options.

        Generic replacement for: model.model_dump() with consistent return type.

        Args:
            model: Pydantic model instance to serialize.
            options: Optional Pydantic model_dump arguments within the config model.
            **kwargs: Inline fallback serialization arguments mapped to ModelDumpOptions automatically.

        Returns:
            Dictionary representation of the model.

        """
        opts = FlextUtilitiesArgs.resolve_options(
            options, kwargs, ModelDumpOptions
        ).unwrap_or(ModelDumpOptions())
        opts_dict = opts.model_dump(exclude_none=True)
        return model.model_dump(**opts_dict)

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
