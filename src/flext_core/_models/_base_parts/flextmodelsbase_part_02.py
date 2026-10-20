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
from collections.abc import Hashable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, dataclass_transform, override

from pydantic import ConfigDict

from flext_core import c, p, t
from flext_core._models import FlextModelsPydantic
from flext_core._models._base_parts import FlextModelsBasePart01
from flext_core._utilities import FlextUtilitiesGenerators, FlextUtilitiesPydantic


class FlextModelsBasePart02(FlextModelsBasePart01):
    class Metadata(FlextModelsPydantic.BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra=c.EXTRA_CONFIG_FORBID,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )
        created_at: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description=(
                    "Timestamp when the metadata record was first created (configured "
                    "timezone)."
                ),
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = FlextModelsPydantic.Field(default_factory=FlextUtilitiesGenerators.now)
        updated_at: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description=(
                    "Timestamp of the most recent metadata update (configured "
                    "timezone)."
                ),
                title="Updated At",
                examples=["2026-03-03T10:05:00+00:00"],
            ),
        ] = FlextModelsPydantic.Field(default_factory=FlextUtilitiesGenerators.now)
        version: Annotated[
            str,
            FlextModelsPydantic.Field(
                default=c.METADATA_SCHEMA_VERSION_DEFAULT,
                description=(
                    "Semantic version string representing the metadata schema revision."
                ),
                title="Metadata Version",
                examples=["1.0.0", "1.2.3"],
            ),
        ] = c.METADATA_SCHEMA_VERSION_DEFAULT
        created_by: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description=(
                    "Identifier of the actor that originally created this metadata."
                ),
                title="Created By",
                examples=["system", "user-123"],
            ),
        ] = None
        modified_by: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Identifier of the actor that last modified this metadata.",
                title="Modified By",
                examples=["system", "user-456"],
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            FlextModelsPydantic.Field(
                description=(
                    "Normalized labels used to classify and filter this metadata."
                ),
                title="Tags",
                examples=[["billing", "critical"]],
            ),
        ] = FlextModelsPydantic.Field(
            default=(),
            validate_default=True,
        )
        attributes: Annotated[
            t.JsonMapping,
            FlextModelsPydantic.BeforeValidator(
                lambda value: (
                    {
                        key: str(item) if isinstance(item, Path) else item
                        for key, item in value.items()
                    }
                    if isinstance(value, Mapping)
                    else value
                )
            ),
            FlextModelsPydantic.Field(
                description="Arbitrary metadata attributes stored as key-value pairs.",
                title="Attributes",
                examples=[{"source": "api", "priority": "high"}],
            ),
        ] = FlextModelsPydantic.Field(
            default_factory=dict,
            validate_default=True,
        )
        metadata_value: Annotated[
            t.Scalar | None,
            FlextModelsPydantic.Field(
                default=None, description="Scalar metadata value."
            ),
        ] = None

        @FlextModelsPydantic.field_validator("attributes", mode="before")
        @classmethod
        def validate_metadata_attributes(cls, value: p.MetadataInput) -> t.JsonMapping:
            """Validate attributes through the canonical metadata boundary.

            Returns:
                The validated JSON metadata attributes.

            """
            return FlextUtilitiesPydantic.validate_metadata_attributes(value)

    class ContractModel(FlextModelsBasePart01.StrictModel):
        """Immutable base model with strict validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_return=True,
            arbitrary_types_allowed=True,
            ser_json_timedelta=c.SERIALIZATION_ISO8601,
            ser_json_bytes=c.SERIALIZATION_BASE64,
            val_json_bytes=c.SERIALIZATION_BASE64,
            hide_input_in_errors=True,
            frozen=True,
        )

    class FrozenValueModel(ContractModel, Hashable):
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

    @dataclass_transform(kw_only_default=True)
    class MutableConfiguredMixin:
        """Shared preset for mutable mixins with assignment validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class NormalizedMutableConfiguredMixin(MutableConfiguredMixin):
        """Shared preset for mutable mixins with whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=True)

    class IdentifiableMixin(NormalizedMutableConfiguredMixin):
        """Mixin for unique identifiers."""

        unique_id: Annotated[
            t.NonEmptyStr,
            FlextModelsPydantic.Field(description="Unique identifier", frozen=False),
        ] = FlextModelsPydantic.Field(default_factory=lambda: str(uuid.uuid4()))


__all__: list[str] = ["FlextModelsBasePart02"]
