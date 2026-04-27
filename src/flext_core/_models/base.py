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
from collections.abc import (
    Mapping,
)
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, ClassVar, Self, override

from flext_core import (
    FlextModelsPydantic as mp,
    FlextRuntime as ur,
    FlextUtilitiesEnforcement as ue,
    FlextUtilitiesPydantic as up,
    c,
    t,
)


class FlextModelsBase:
    """Container for base model classes - Tier 0, 100% standalone."""

    class EnforcedModel(mp.BaseModel):
        """Base model that enforces architectural rules on subclasses."""

        @classmethod
        @override
        def __pydantic_init_subclass__(cls, **kwargs: t.JsonValue) -> None:
            super().__pydantic_init_subclass__(**kwargs)
            ue.run(cls)

    class ManagedModel(EnforcedModel):
        """Shared preset for assignment validation with forbidden extra fields."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            validate_assignment=True,
            extra=c.EXTRA_CONFIG_FORBID,
        )

    class EnumManagedModel(ManagedModel):
        """Shared preset for managed models that serialize enum values."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            use_enum_values=True,
        )

    class NormalizedModel(EnumManagedModel):
        """Shared preset for managed models with whitespace normalization."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            str_strip_whitespace=True,
        )

    class StrictManagedModel(NormalizedModel):
        """Shared preset for strict managed validation boundaries."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            strict=True,
            validate_default=True,
        )

    class StrictModel(StrictManagedModel):
        """Reusable strict model preset for validated domain boundaries."""

    class FrozenModel(StrictModel):
        """Immutable strict domain model preset."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(frozen=True)

    class ArbitraryTypesModel(EnumManagedModel):
        """Base model with arbitrary types support."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            arbitrary_types_allowed=True,
        )

    class StrictBoundaryModel(FrozenModel):
        """Strict boundary model for API/external boundaries."""

    class FlexibleInternalModel(NormalizedModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="ignore",
        )

    class ImmutableValueModel(ManagedModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True,
        )

    class TaggedModel(EnforcedModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class FlexibleModel(ArbitraryTypesModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="ignore",
        )

    class DynamicModel(FlexibleModel):
        """Dynamic domain model preset with string whitespace normalization."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            str_strip_whitespace=True,
        )

    class FrozenDynamicModel(DynamicModel):
        """Immutable dynamic domain model preset."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(frozen=True)

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
                description="Timestamp when the metadata record was first created in UTC.",
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = mp.Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime,
            mp.Field(
                description="Timestamp of the most recent metadata update in UTC.",
                title="Updated At",
                examples=["2026-03-03T10:05:00+00:00"],
            ),
        ] = mp.Field(default_factory=lambda: datetime.now(UTC))
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
            Mapping[str, t.JsonValue],
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
            return bool(self.model_dump() == other.model_dump())

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

    class TimestampableMixin(MutableConfiguredMixin):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        created_at: Annotated[
            datetime,
            mp.AfterValidator(lambda v: ur.ensure_utc_datetime(v)),
            mp.Field(
                description="Creation timestamp (UTC)",
                frozen=True,
            ),
        ] = mp.Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime | None,
            mp.AfterValidator(lambda v: ur.ensure_utc_datetime(v)),
            mp.Field(default=None, description="Last update timestamp (UTC)"),
        ] = None

        @up.field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 for JSON."""
            return value.isoformat() if value else None

        @up.model_validator(mode="after")
        def validate_timestamp_consistency(self) -> Self:
            """Validate timestamp consistency."""
            if self.updated_at is not None and self.updated_at < self.created_at:
                raise ValueError(c.ERR_MODEL_UPDATED_AT_BEFORE_CREATED_AT)
            return self

    class VersionableMixin(MutableConfiguredMixin):
        """Mixin for versioning with optimistic locking."""

        version: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS

        @up.model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency."""
            if self.version < c.DEFAULT_RETRY_DELAY_SECONDS:
                raise ValueError(
                    c.ERR_MODEL_VERSION_BELOW_MINIMUM.format(
                        version=self.version,
                        minimum=c.DEFAULT_RETRY_DELAY_SECONDS,
                    ),
                )
            return self

    class RetryConfigurationMixin(mp.BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(populate_by_name=True)
        max_retries: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                description="Maximum retry attempts",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        initial_delay_seconds: Annotated[
            t.PositiveFloat,
            mp.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Initial delay between retries",
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS
        max_delay_seconds: Annotated[
            t.PositiveFloat,
            mp.Field(
                default=c.DEFAULT_MAX_DELAY_SECONDS,
                description="Maximum delay between retries",
            ),
        ] = c.DEFAULT_MAX_DELAY_SECONDS

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__: list[str] = ["FlextModelsBase"]
