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
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Self, override

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from flext_core import FlextRuntime, FlextUtilitiesEnforcement, c, t


class FlextModelsBase:
    """Container for base model classes - Tier 0, 100% standalone."""

    class EnforcedModel(BaseModel):
        """Base model that enforces architectural rules on subclasses."""

        @classmethod
        def with_defaults(cls, **overrides: t.Container) -> Self:
            """Build model from its declared defaults plus explicit overrides."""
            return cls.model_validate(overrides)

        @classmethod
        @override
        def __pydantic_init_subclass__(cls, **kwargs: t.Container) -> None:
            super().__pydantic_init_subclass__(**kwargs)
            FlextUtilitiesEnforcement.run(cls)

    class ManagedModel(EnforcedModel):
        """Shared preset for assignment validation with forbidden extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra=c.ExtraConfig.FORBID.value,
        )

    class EnumManagedModel(ManagedModel):
        """Shared preset for managed models that serialize enum values."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            use_enum_values=True,
        )

    class NormalizedModel(EnumManagedModel):
        """Shared preset for managed models with whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            str_strip_whitespace=True,
        )

    class StrictManagedModel(NormalizedModel):
        """Shared preset for strict managed validation boundaries."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            validate_default=True,
        )

    class StrictModel(StrictManagedModel):
        """Reusable strict model preset for validated domain boundaries."""

    class FrozenModel(StrictModel):
        """Immutable strict domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class ArbitraryTypesModel(EnumManagedModel):
        """Base model with arbitrary types support."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
        )

    class StrictBoundaryModel(FrozenModel):
        """Strict boundary model for API/external boundaries."""

    class FlexibleInternalModel(NormalizedModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra="ignore",
        )

    class ImmutableValueModel(ManagedModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
        )

    class TaggedModel(EnforcedModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class FlexibleModel(ArbitraryTypesModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra="ignore",
        )

    class DynamicModel(FlexibleModel):
        """Dynamic domain model preset with string whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            str_strip_whitespace=True,
        )

    class FrozenDynamicModel(DynamicModel):
        """Immutable dynamic domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class Metadata(BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra=c.ExtraConfig.FORBID.value,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )
        created_at: Annotated[
            datetime,
            Field(
                description="Timestamp when the metadata record was first created in UTC.",
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime,
            Field(
                description="Timestamp of the most recent metadata update in UTC.",
                title="Updated At",
                examples=["2026-03-03T10:05:00+00:00"],
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        version: Annotated[
            str,
            Field(
                default="1.0.0",
                description="Semantic version string representing the metadata schema revision.",
                title="Metadata Version",
                examples=["1.0.0", "1.2.3"],
            ),
        ] = "1.0.0"
        created_by: Annotated[
            str | None,
            Field(
                default=None,
                description="Identifier of the actor that originally created this metadata.",
                title="Created By",
                examples=["system", "user-123"],
            ),
        ] = None
        modified_by: Annotated[
            str | None,
            Field(
                default=None,
                description="Identifier of the actor that last modified this metadata.",
                title="Modified By",
                examples=["system", "user-456"],
            ),
        ] = None
        tags: Annotated[
            t.StrSequence,
            Field(
                description="Normalized labels used to classify and filter this metadata.",
                title="Tags",
                examples=[["billing", "critical"]],
            ),
        ] = Field(default_factory=list)
        attributes: Annotated[
            Mapping[str, t.MetadataValue],
            BeforeValidator(FlextRuntime.validate_metadata_attributes),
            Field(
                description="Arbitrary metadata attributes stored as key-value pairs.",
                title="Attributes",
                examples=[{"source": "api", "priority": "high"}],
            ),
        ] = Field(default_factory=dict)
        metadata_value: Annotated[
            t.Scalar | None,
            Field(default=None, description="Scalar metadata value."),
        ] = None

    class ContractModel(StrictModel):
        """Immutable base model with strict validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_return=True,
            arbitrary_types_allowed=True,
            ser_json_timedelta=c.Serialization.ISO8601.value,
            ser_json_bytes=c.Serialization.BASE64.value,
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

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

    class NormalizedMutableConfiguredMixin(MutableConfiguredMixin):
        """Shared preset for mutable mixins with whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            str_strip_whitespace=True,
        )

    class IdentifiableMixin(NormalizedMutableConfiguredMixin):
        """Mixin for unique identifiers."""

        unique_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique identifier",
                frozen=False,
            ),
        ] = Field(default_factory=lambda: str(uuid.uuid4()))

        def regenerate_id(self) -> None:
            """Regenerate the unique_id with a new UUID."""
            self.unique_id = str(uuid.uuid4())

    class TimestampableMixin(MutableConfiguredMixin):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        created_at: Annotated[
            datetime,
            AfterValidator(lambda v: FlextRuntime.ensure_utc_datetime(v)),
            Field(
                description="Creation timestamp (UTC)",
                frozen=True,
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime | None,
            AfterValidator(lambda v: FlextRuntime.ensure_utc_datetime(v)),
            Field(default=None, description="Last update timestamp (UTC)"),
        ] = None

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 for JSON."""
            return value.isoformat() if value else None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

        @model_validator(mode="after")
        def validate_timestamp_consistency(self) -> Self:
            """Validate timestamp consistency."""
            if self.updated_at is not None and self.updated_at < self.created_at:
                msg = "updated_at cannot be before created_at"
                raise ValueError(msg)
            return self

    class VersionableMixin(MutableConfiguredMixin):
        """Mixin for versioning with optimistic locking."""

        version: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS

        def increment_version(self) -> None:
            """Increment the version number."""
            self.version += 1

        @model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency."""
            if self.version < c.DEFAULT_RETRY_DELAY_SECONDS:
                msg = f"Version {self.version} is below minimum {c.DEFAULT_RETRY_DELAY_SECONDS}"
                raise ValueError(msg)
            return self

    class RetryConfigurationMixin(BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)
        max_retries: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                description="Maximum retry attempts",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        initial_delay_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Initial delay between retries",
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS
        max_delay_seconds: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_MAX_DELAY_SECONDS,
                description="Maximum delay between retries",
            ),
        ] = c.DEFAULT_MAX_DELAY_SECONDS

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__ = ["FlextModelsBase"]
