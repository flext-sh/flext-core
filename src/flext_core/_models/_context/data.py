"""Context initialization data and native serialization validators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._models import FlextModelsBase, FlextModelsPydantic

_EMPTY_SCALAR_MAPPING: Mapping[str, flext_core.t.Scalar] = MappingProxyType({})


class FlextModelsContextData:
    """Context data declarations with validation at the Pydantic boundary."""

    class ContextData(FlextModelsBase.FlexibleInternalModel):
        """Lightweight container for initializing context state."""

        data: Annotated[
            Mapping[str, flext_core.t.Scalar],
            FlextModelsPydantic.Field(
                description="Initial context data as key-value pairs"
            ),
        ] = FlextModelsPydantic.Field(default_factory=lambda: _EMPTY_SCALAR_MAPPING)
        metadata: Annotated[
            FlextModelsBase.Metadata | Mapping[str, flext_core.t.Scalar] | None,
            FlextModelsPydantic.Field(
                description="Context metadata (creation info, source, etc.)"
            ),
        ] = None

        @FlextModelsPydantic.field_validator("data", mode="before")
        @classmethod
        def validate_dict_serializable(
            cls, value: flext_core.p.ModelInput | None
        ) -> Mapping[str, flext_core.t.Scalar]:
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
                    flext_core.c.ERR_CONTEXT_CANNOT_NORMALIZE_TYPE_TO_MAPPING.format(
                        type_name=type(value).__name__
                    )
                )
            return MappingProxyType({
                key: item
                if isinstance(item, flext_core.c.PRIMITIVES_TYPES)
                else str(item)
                for key, item in data.items()
            })

        @FlextModelsPydantic.field_validator("metadata", mode="before")
        @classmethod
        def validate_metadata_before(
            cls, value: flext_core.p.MetadataInput
        ) -> flext_core.p.MetadataInput:
            """Validate attribute mappings as metadata without a failure fallback.

            Returns:
                The validated metadata or unchanged nonmapping input.

            """
            if isinstance(value, dict):
                return FlextModelsBase.Metadata.model_validate({
                    flext_core.c.FIELD_ATTRIBUTES: value
                })
            return value


__all__: Sequence[str] = ("FlextModelsContextData",)
