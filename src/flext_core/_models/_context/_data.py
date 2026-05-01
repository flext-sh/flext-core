"""Context data models with serialization validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from types import MappingProxyType
from typing import Annotated

from pydantic import BeforeValidator, field_validator

from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core.constants import FlextConstants as c
from flext_core.typings import FlextTypes as t


class FlextModelsContextData:
    """Namespace for context data models."""

    @staticmethod
    def _coerce_scalar_mapping(
        items: t.MappingKV[str, object],
    ) -> t.MappingKV[str, t.Scalar]:
        """Return an immutable mapping with non-scalar values stringified."""
        return MappingProxyType({
            k: val if isinstance(val, (str, int, float, bool)) else str(val)
            for k, val in items.items()
        })

    @staticmethod
    def normalize_to_mapping(
        v: t.MappingKV[str, t.Scalar] | t.JsonPayload | None,
    ) -> t.MappingKV[str, t.Scalar]:
        """Convert value to an immutable flat mapping with scalar values only."""
        if v is None:
            return MappingProxyType({})
        if isinstance(v, Mapping):
            return FlextModelsContextData._coerce_scalar_mapping(v)
        if isinstance(v, mp.BaseModel):
            return FlextModelsContextData._coerce_scalar_mapping(v.model_dump())
        msg = c.ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING.format(
            type_name=type(v).__name__,
        )
        raise ValueError(msg)

    @staticmethod
    def normalize_metadata_before(
        v: t.JsonPayload | None,
    ) -> t.JsonPayload | None:
        """Normalize input to Metadata or return as-is."""
        if v is None or isinstance(v, m.Metadata):
            return v
        if isinstance(v, dict):
            try:
                return m.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: v,
                })
            except Exception:
                return v
        return v

    class SerializableDataValidatorMixin:
        """Mixin validating that data is JSON-serializable for context models."""

        @field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls,
            v: t.MappingKV[str, t.Scalar] | mp.BaseModel | None,
        ) -> t.MappingKV[str, t.Scalar]:
            """Validate that data values are JSON-serializable."""
            if v is None:
                return MappingProxyType({})
            if isinstance(v, Mapping):
                return MappingProxyType({
                    k: (
                        str(val)
                        if not isinstance(val, (str, int, float, bool))
                        else val
                    )
                    for k, val in v.items()
                })
            return MappingProxyType({
                k: (str(val) if not isinstance(val, (str, int, float, bool)) else val)
                for k, val in v.model_dump().items()
            })

    class ContextData(
        SerializableDataValidatorMixin,
        m.FlexibleInternalModel,
    ):
        """Lightweight container for initializing context state."""

        data: Annotated[
            t.MappingKV[str, t.Scalar],
            mp.Field(
                description="Initial context data as key-value pairs",
            ),
        ] = mp.Field(default_factory=lambda: MappingProxyType({}))
        metadata: Annotated[
            m.Metadata | t.MappingKV[str, t.Scalar] | None,
            BeforeValidator(
                lambda v: FlextModelsContextData.normalize_metadata_before(v),
            ),
            mp.Field(
                default=None,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = None

        @classmethod
        def normalize_to_serializable_value(
            cls,
            val: t.Scalar,
        ) -> t.Scalar:
            """Return scalar value as-is (already serializable)."""
            return val

        @staticmethod
        def normalize_to_container(
            val: t.Scalar,
        ) -> t.Scalar:
            """Return scalar value as-is."""
            return val if isinstance(val, (str, int, float, bool)) else str(val)


__all__: t.MutableSequenceOf[str] = ["FlextModelsContextData"]
