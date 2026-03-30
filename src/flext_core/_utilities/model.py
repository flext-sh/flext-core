"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from flext_core import m, r, t


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization."""

    @staticmethod
    def _normalize_model_input(
        data: BaseModel | Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> Mapping[str, t.ValueOrModel]:
        if isinstance(data, BaseModel):
            root_value = getattr(data, "root", None)
            if isinstance(root_value, Mapping):
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

    @staticmethod
    def from_kwargs[M: BaseModel](model_cls: type[M], **kwargs: t.Scalar) -> r[M]:
        """Create Pydantic model from kwargs with p.Result.

        Accepts any type in kwargs - Pydantic 2 field_validators will handle
        type conversions automatically (e.g., str → Path, dict → BaseModel, etc.).
        All parameter validation and conversion happens via Pydantic 2 Field constraints
        and field_validators defined in the model.

        """
        instance_result = r[M].create_from_callable(lambda: model_cls(**kwargs))
        if instance_result.is_failure:
            return r[M].fail(f"Model validation failed: {instance_result.error}")
        instance = instance_result.value
        if not isinstance(instance, model_cls):
            return r[M].fail(
                f"Expected {model_cls.__name__}, got {type(instance).__name__}",
            )
        return r[M].ok(instance)

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

    @classmethod
    def merge_defaults[M: BaseModel](
        cls,
        model_cls: type[M],
        defaults: Mapping[str, t.ValueOrModel] | t.ConfigMap,
        overrides: Mapping[str, t.ValueOrModel] | t.ConfigMap,
    ) -> r[M]:
        """Merge default values with overrides and validate as a target model."""
        merged: dict[str, t.ValueOrModel] = dict(cls._normalize_model_input(defaults))
        merged.update(cls._normalize_model_input(overrides))
        return cls.load(model_cls, merged)

    @staticmethod
    def ensure_metadata(
        metadata: BaseModel | Mapping[str, t.MetadataValue] | None,
    ) -> m.Metadata:
        """Normalize loose metadata input into the canonical Metadata model."""
        if metadata is None:
            return m.Metadata()
        if isinstance(metadata, m.Metadata):
            return metadata
        if isinstance(metadata, BaseModel):
            metadata_mapping = metadata.model_dump()
        elif isinstance(metadata, Mapping):
            metadata_mapping = dict(metadata.items())
        else:
            msg = "metadata must be None, dict, or Metadata"
            raise TypeError(msg)
        normalized = m.Validators.metadata_map_adapter().validate_python(
            metadata_mapping,
        )
        return m.Metadata(attributes=normalized)


__all__ = ["FlextUtilitiesModel"]
