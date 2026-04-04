"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Literal

from pydantic import BaseModel, Field

from flext_core import r, t
from flext_core._models.base import FlextModelsBase
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel


class ModelDumpOptions(FlextModelsBase.FlexibleInternalModel):
    """Options controlling Pydantic model_dump() serialization behavior."""

    by_alias: bool | None = Field(
        default=None, description="Serialize using field aliases"
    )
    exclude_none: bool | None = Field(
        default=None, description="Exclude None-valued fields"
    )
    exclude_unset: bool | None = Field(
        default=None, description="Exclude fields not explicitly set"
    )
    exclude_defaults: bool | None = Field(
        default=None, description="Exclude fields matching defaults"
    )
    include: set[str] | None = Field(
        default=None, description="Whitelist of field names to include"
    )
    exclude: set[str] | None = Field(
        default=None, description="Blacklist of field names to exclude"
    )


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    @staticmethod
    def safe_get_attribute(
        obj: t.RuntimeData | type | ModuleType,
        attr: str,
        default: t.ValueOrModel | None = None,
    ) -> t.ValueOrModel | None:
        """Safe attribute access without raising AttributeError."""
        return getattr(obj, attr) if hasattr(obj, attr) else default

    @staticmethod
    def _normalize_model_input(
        data: BaseModel | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> Mapping[str, t.ValueOrModel]:
        if isinstance(
            data,
            BaseModel,
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
        options: ModelDumpOptions | None = None,
        **kwargs: t.ValueOrModel,
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

    @staticmethod
    def append_metadata_sequence_item(
        metadata: t.Dict,
        key: Literal["failed_items", "warning_items"],
        item: t.ValueOrModel,
    ) -> None:
        """Append one normalized item to a metadata sequence bucket."""
        raw_items = metadata.root.get(key)
        result_list: t.MutableContainerList = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if isinstance(
                    raw_item,
                    (str, int, float, bool, datetime, Path, list, dict, tuple),
                ):
                    result_list.append(raw_item)
                elif raw_item is not None:
                    result_list.append(str(raw_item))
        if isinstance(item, (str, int, float, bool, datetime, Path, list, dict, tuple)):
            result_list.append(item)
        elif item is not None:
            result_list.append(str(item))
        metadata.root[key] = result_list

    @staticmethod
    def upsert_skip_reason(
        metadata: t.Dict,
        item: t.ValueOrModel,
        reason: str,
    ) -> None:
        """Store one skip reason keyed by the stringified item representation."""
        raw_reasons = metadata.root.get("skip_reasons", {})
        reasons: t.MutableStrMapping = {}
        if isinstance(raw_reasons, Mapping):
            reasons = {str(key): str(value) for key, value in raw_reasons.items()}
        reasons[str(item)] = reason
        metadata.root["skip_reasons"] = reasons


__all__ = ["FlextUtilitiesModel"]
