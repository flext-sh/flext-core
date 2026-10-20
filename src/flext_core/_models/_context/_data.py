"""Context initialization data and native serialization validators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated

from flext_core import c, p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic

_EMPTY_SCALAR_MAPPING: t.MappingKV[str, t.Scalar] = MappingProxyType({})


class FlextModelsContextData:
    """Context data declarations with validation at the Pydantic boundary."""

    class ContextData(FlextModelsBase.FlexibleInternalModel):
        """Lightweight container for initializing context state."""

        data: Annotated[
            t.MappingKV[str, t.Scalar],
            FlextModelsPydantic.Field(
                description="Initial context data as key-value pairs"
            ),
        ] = FlextModelsPydantic.Field(default_factory=lambda: _EMPTY_SCALAR_MAPPING)
        metadata: Annotated[
            FlextModelsBase.Metadata | t.MappingKV[str, t.Scalar] | None,
            FlextModelsPydantic.Field(
                description="Context metadata (creation info, source, etc.)"
            ),
        ] = None

        @FlextModelsPydantic.field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls, value: p.ModelInput | None
        ) -> t.MappingKV[str, t.Scalar]:
            """Validate the immutable flat scalar representation at ingress.

            Returns:
                The immutable scalar mapping.

            Raises:
                TypeError: If the input cannot represent a mapping.

            """
            if value is None:
                return _EMPTY_SCALAR_MAPPING
            data = (
                value.model_dump()
                if isinstance(value, FlextModelsPydantic.BaseModel)
                else value
            )
            if not isinstance(data, Mapping):
                raise TypeError(
                    c.ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING.format(
                        type_name=type(value).__name__
                    )
                )
            return MappingProxyType({
                key: item if isinstance(item, c.PRIMITIVES_TYPES) else str(item)
                for key, item in data.items()
            })

        @FlextModelsPydantic.field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata_before(cls, value: p.MetadataInput) -> p.MetadataInput:
            """Validate attribute mappings as metadata without a failure fallback.

            Returns:
                The validated metadata or unchanged nonmapping input.

            """
            if isinstance(value, dict):
                return FlextModelsBase.Metadata.model_validate({
                    c.FIELD_ATTRIBUTES: value
                })
            return value


__all__: t.StrSequence = ("FlextModelsContextData",)
