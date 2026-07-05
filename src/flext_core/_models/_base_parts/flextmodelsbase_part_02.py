"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, ClassVar, override

from flext_core._models._base_parts.flextmodelsbase_part_01 import (
    FlextModelsBase as FlextModelsBasePart01,
)
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as ur,
)
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug
from flext_core.constants import FlextConstants as c


class FlextModelsBase(FlextModelsBasePart01):
    StrictModel = FlextModelsBasePart01.StrictModel

    class Metadata(mp.BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra=c.EXTRA_CONFIG_FORBID,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )
        created_at: Annotated[
            datetime,
            mp.Field(
                description="Timestamp when the metadata record was first created (configured timezone).",
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = mp.Field(default_factory=ug.now)
        updated_at: Annotated[
            datetime,
            mp.Field(
                description="Timestamp of the most recent metadata update (configured timezone).",
                title="Updated At",
                examples=["2026-03-03T10:05:00+00:00"],
            ),
        ] = mp.Field(default_factory=ug.now)
        version: Annotated[
            str,
            mp.Field(
                default="1.0.0",
                description="Semantic version string representing the metadata schema revision.",
                title="Metadata Version",
                examples=["1.0.0", "1.2.3"],
            ),
        ] = "1.0.0"
        created_by: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Identifier of the actor that originally created this metadata.",
                title="Created By",
                examples=["system", "user-123"],
            ),
        ] = None
        modified_by: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Identifier of the actor that last modified this metadata.",
                title="Modified By",
                examples=["system", "user-456"],
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            mp.Field(
                description="Normalized labels used to classify and filter this metadata.",
                title="Tags",
                examples=[["billing", "critical"]],
            ),
        ] = mp.Field(default_factory=tuple)
        attributes: Annotated[
            t.JsonMapping,
            mp.BeforeValidator(ur.validate_metadata_attributes),
            mp.Field(
                description="Arbitrary metadata attributes stored as key-value pairs.",
                title="Attributes",
                examples=[{"source": "api", "priority": "high"}],
            ),
        ] = mp.Field(default_factory=lambda: MappingProxyType({}))
        metadata_value: Annotated[
            t.Scalar | None,
            mp.Field(default=None, description="Scalar metadata value."),
        ] = None

    class ContractModel(StrictModel):
        """Immutable base model with strict validation."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            validate_return=True,
            arbitrary_types_allowed=True,
            ser_json_timedelta=c.SERIALIZATION_ISO8601,
            ser_json_bytes=c.SERIALIZATION_BASE64,
            hide_input_in_errors=True,
            frozen=True,
        )

    class FrozenValueModel(ContractModel):
        """Value model with equality/hash by value."""

        @override
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented
            self_dump: t.JsonMapping = self.model_dump()
            other_dump: t.JsonMapping = other.model_dump()
            return self_dump == other_dump

        def __hash__(self) -> int:
            data = self.model_dump()
            return hash(tuple(sorted(((k, str(v)) for k, v in data.items()))))

    class MutableConfiguredMixin:
        """Shared preset for mutable mixins with assignment validation."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class NormalizedMutableConfiguredMixin(MutableConfiguredMixin):
        """Shared preset for mutable mixins with whitespace normalization."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            str_strip_whitespace=True,
        )

    class IdentifiableMixin(NormalizedMutableConfiguredMixin):
        """Mixin for unique identifiers."""

        unique_id: Annotated[
            t.NonEmptyStr,
            mp.Field(
                description="Unique identifier",
                frozen=False,
            ),
        ] = mp.Field(default_factory=lambda: str(uuid.uuid4()))


__all__: list[str] = ["FlextModelsBase"]
