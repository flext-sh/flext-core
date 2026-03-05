"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelFoundation
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Literal, Self, override

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from flext_core import c, t
from flext_core._models.containers import FlextModelsContainers


def _ensure_utc_datetime(v: datetime | None) -> datetime | None:
    """Ensure datetime is UTC timezone."""
    if v is not None and v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc_datetime)]


class FlextModelFoundation:
    """Container for base model classes - Tier 0, 100% standalone."""

    class Validators:
        """Pydantic v2 validators - single namespace for all field validators."""

        _tags_adapter: ClassVar[TypeAdapter[list[str]] | None] = None
        _list_adapter: ClassVar[TypeAdapter[list[t.ContainerValue]] | None] = None
        _strict_string_adapter: ClassVar[
            TypeAdapter[Annotated[str, Field(strict=True)]] | None
        ] = None
        _metadata_map_adapter: ClassVar[
            TypeAdapter[dict[str, t.MetadataValue]] | None
        ] = None
        _config_adapter: ClassVar[TypeAdapter[dict[str, t.ContainerValue]] | None] = (
            None
        )

        @classmethod
        def config_adapter(cls) -> TypeAdapter[dict[str, t.ContainerValue]]:
            """Lazy-load config TypeAdapter on first access."""
            if cls._config_adapter is None:
                cls._config_adapter = TypeAdapter(dict[str, t.ContainerValue])
            return cls._config_adapter

        @classmethod
        def list_adapter(cls) -> TypeAdapter[list[t.ContainerValue]]:
            """Lazy-load list TypeAdapter on first access."""
            if cls._list_adapter is None:
                cls._list_adapter = TypeAdapter(list[t.ContainerValue])
            return cls._list_adapter

        @classmethod
        def metadata_map_adapter(
            cls,
        ) -> TypeAdapter[dict[str, t.MetadataValue]]:
            """Lazy-load metadata map TypeAdapter on first access."""
            if cls._metadata_map_adapter is None:
                cls._metadata_map_adapter = TypeAdapter(
                    dict[str, t.MetadataValue],
                )
            return cls._metadata_map_adapter

        @classmethod
        def strict_string_adapter(
            cls,
        ) -> TypeAdapter[Annotated[str, Field(strict=True)]]:
            """Lazy-load strict string TypeAdapter on first access."""
            if cls._strict_string_adapter is None:
                cls._strict_string_adapter = TypeAdapter(
                    Annotated[str, Field(strict=True)],
                )
            return cls._strict_string_adapter

        @classmethod
        def tags_adapter(cls) -> TypeAdapter[list[str]]:
            """Lazy-load tags TypeAdapter on first access."""
            if cls._tags_adapter is None:
                cls._tags_adapter = TypeAdapter(list[str])
            return cls._tags_adapter

        @staticmethod
        def ensure_utc_datetime(v: datetime | None) -> datetime | None:
            """Ensure datetime is UTC timezone."""
            return _ensure_utc_datetime(v)

        @staticmethod
        def normalize_to_list(v: t.ContainerValue) -> list[t.ContainerValue]:
            """Normalize value to list format."""
            try:
                return FlextModelFoundation.Validators.list_adapter().validate_python(v)
            except ValidationError:
                return [v]

        @staticmethod
        def strip_whitespace(v: str) -> str:
            """Strip leading and trailing whitespace from string."""
            return v.strip()

        @staticmethod
        def validate_config_dict(
            v: t.ContainerValue,
        ) -> Mapping[str, t.ContainerValue]:
            """Validate configuration dictionary structure."""
            try:
                normalized = (
                    FlextModelFoundation.Validators.config_adapter().validate_python(v)
                )
            except ValidationError as exc:
                msg = "Configuration must be a dictionary"
                raise TypeError(msg) from exc
            out: dict[str, t.ContainerValue] = {}
            for key, item in normalized.items():
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
                out[key] = item
            return out

        @staticmethod
        def validate_tags_list(v: t.ContainerValue) -> list[str]:
            """Validate and normalize tags list."""
            try:
                raw_tags = (
                    FlextModelFoundation.Validators.list_adapter().validate_python(v)
                )
            except ValidationError as exc:
                msg = "Tags must be a list"
                raise TypeError(msg) from exc
            normalized: list[str] = []
            seen: set[str] = set()
            for tag in raw_tags:
                try:
                    clean_tag = (
                        FlextModelFoundation.Validators.strict_string_adapter()
                        .validate_python(tag)
                        .strip()
                        .lower()
                    )
                except ValidationError as exc:
                    msg = "Tag must be string"
                    raise TypeError(msg) from exc
                if clean_tag and clean_tag not in seen:
                    normalized.append(clean_tag)
                    seen.add(clean_tag)
            return normalized

    # ═══════════════════════════════════════════════════════════════════════════
    # BASE MODEL CONFIGURATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config = ConfigDict(
            defer_build=True,
            validate_assignment=True,
            extra=c.ModelConfig.EXTRA_FORBID,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class StrictBoundaryModel(BaseModel):
        """Strict boundary model for API/external boundaries."""

        model_config = ConfigDict(
            defer_build=True,
            strict=True,
            validate_assignment=True,
            extra="forbid",
            str_strip_whitespace=True,
            use_enum_values=True,
            frozen=True,
        )

    class FlexibleInternalModel(BaseModel):
        """Flexible internal model for domain logic."""

        model_config = ConfigDict(
            defer_build=True,
            validate_assignment=True,
            extra="ignore",
            str_strip_whitespace=True,
            use_enum_values=True,
        )

    class ImmutableValueModel(BaseModel):
        """Immutable value model for value objects."""

        model_config = ConfigDict(
            defer_build=True,
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

    class TaggedModel(BaseModel):
        """Base pattern for tagged discriminated unions."""

        model_config = ConfigDict(defer_build=True, extra="forbid")
        tag: ClassVar[str]

    class DynamicConfigModel(BaseModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config = ConfigDict(
            defer_build=True,
            validate_assignment=True,
            extra="allow",
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # METADATA MODEL
    # ═══════════════════════════════════════════════════════════════════════════

    class Metadata(BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config = ConfigDict(
            defer_build=True,
            extra=c.ModelConfig.EXTRA_FORBID,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )

        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the metadata record was first created in UTC.",
            title="Created At",
            examples=["2026-03-03T10:00:00+00:00"],
        )
        updated_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp of the most recent metadata update in UTC.",
            title="Updated At",
            examples=["2026-03-03T10:05:00+00:00"],
        )
        version: str = Field(
            default="1.0.0",
            description="Semantic version string representing the metadata schema revision.",
            title="Metadata Version",
            examples=["1.0.0", "1.2.3"],
        )
        created_by: str | None = Field(
            default=None,
            description="Identifier of the actor that originally created this metadata.",
            title="Created By",
            examples=["system", "user-123"],
        )
        modified_by: str | None = Field(
            default=None,
            description="Identifier of the actor that last modified this metadata.",
            title="Modified By",
            examples=["system", "user-456"],
        )
        tags: list[str] = Field(
            default_factory=list,
            description="Normalized labels used to classify and filter this metadata.",
            title="Tags",
            examples=[["billing", "critical"]],
        )
        attributes: Mapping[str, t.MetadataValue] = Field(
            default_factory=dict,
            description="Arbitrary metadata attributes stored as key-value pairs.",
            title="Attributes",
            examples=[{"source": "api", "priority": "high"}],
        )
        metadata_value: t.Scalar | None = Field(
            default=None,
            description="Scalar metadata value.",
        )

        @field_validator("attributes", mode="before")
        @classmethod
        def _validate_attributes(
            cls,
            value: t.MetadataValue | Mapping[str, t.MetadataValue] | None,
        ) -> Mapping[str, t.MetadataValue]:
            if value is None:
                return {}
            try:
                result = FlextModelsContainers.Dict.model_validate(value).root
            except ValidationError:
                if not isinstance(value, BaseModel):
                    msg = "attributes must be dict-like"
                    raise TypeError(msg) from None
                dumped = value.model_dump()
                try:
                    result = FlextModelsContainers.Dict.model_validate(dumped).root
                except ValidationError as exc:
                    msg = "attributes BaseModel must dump to mapping"
                    raise TypeError(msg) from exc
            for key in result:
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
            return (
                FlextModelFoundation.Validators.metadata_map_adapter().validate_python(
                    result,
                )
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # DISCRIMINATED UNIONS
    # ═══════════════════════════════════════════════════════════════════════════

    class CommandMessage(BaseModel):
        """Command message with discriminated union support."""

        message_type: Literal["command"] = "command"
        command_type: str
        issuer_id: str | None = None
        data: FlextModelsContainers.Dict = Field(
            default_factory=FlextModelsContainers.Dict,
            description="Command payload containing input data required for execution.",
        )

    class QueryMessage(BaseModel):
        """Query message with discriminated union support."""

        message_type: Literal["query"] = "query"
        query_type: str
        filters: FlextModelsContainers.Dict = Field(
            default_factory=FlextModelsContainers.Dict,
            description="Filter criteria used to constrain query results.",
        )
        pagination: FlextModelsContainers.Dict | None = None

    class EventMessage(BaseModel):
        """Event message with discriminated union support."""

        message_type: Literal["event"] = "event"
        event_type: str
        aggregate_id: str
        data: FlextModelsContainers.Dict = Field(
            default_factory=FlextModelsContainers.Dict,
            description="Event payload with domain data describing what happened.",
        )
        metadata: FlextModelFoundation.Metadata = Field(
            default_factory=lambda: FlextModelFoundation.Metadata(),
            description="Structured metadata associated with this event message.",
        )

    MessageUnion = Annotated[
        CommandMessage | QueryMessage | EventMessage,
        Discriminator("message_type"),
    ]

    class SuccessResult(BaseModel):
        """Success result for discriminated union."""

        result_type: Literal["success"] = "success"
        value: t.ContainerValue
        metadata: FlextModelFoundation.Metadata = Field(
            default_factory=lambda: FlextModelFoundation.Metadata(),
            description="Structured metadata attached to a successful operation result.",
        )

    class FailureResult(BaseModel):
        """Failure result for discriminated union."""

        result_type: Literal["failure"] = "failure"
        error: str
        error_code: str | None = None
        error_data: FlextModelFoundation.Metadata | None = None

    class PartialResult(BaseModel):
        """Partial result for discriminated union."""

        result_type: Literal["partial"] = "partial"
        value: t.ContainerValue
        warnings: list[str] = Field(
            default_factory=list,
            description="Non-fatal warning messages generated during partial processing.",
        )
        partial_success_rate: float

    OperationResult = Annotated[
        SuccessResult | FailureResult | PartialResult,
        Discriminator("result_type"),
    ]

    class ValidOutcome(BaseModel):
        """Valid validation outcome."""

        outcome_type: Literal["valid"] = "valid"
        validated_data: t.ContainerValue
        validation_time_ms: float

    class InvalidOutcome(BaseModel):
        """Invalid validation outcome."""

        outcome_type: Literal["invalid"] = "invalid"
        errors: list[str]
        error_codes: list[str] = Field(
            default_factory=list,
            description="Machine-readable error codes that classify validation failures.",
        )

    class WarningOutcome(BaseModel):
        """Warning validation outcome."""

        outcome_type: Literal["warning"] = "warning"
        validated_data: t.ContainerValue
        warnings: list[str]
        validation_time_ms: float

    ValidationOutcome = Annotated[
        ValidOutcome | InvalidOutcome | WarningOutcome,
        Discriminator("outcome_type"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # FROZEN MODELS
    # ═══════════════════════════════════════════════════════════════════════════

    class FrozenStrictModel(BaseModel):
        """Immutable base model with strict validation."""

        model_config = ConfigDict(
            defer_build=True,
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra=c.ModelConfig.EXTRA_FORBID,
            ser_json_timedelta=c.Utilities.SERIALIZATION_ISO8601,
            ser_json_bytes=c.Utilities.SERIALIZATION_BASE64,
            hide_input_in_errors=True,
            frozen=True,
        )

    class FrozenValueModel(FrozenStrictModel):
        """Value model with equality/hash by value."""

        @override
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented
            return bool(self.model_dump() == other.model_dump())

        def __hash__(self) -> int:
            data = self.model_dump()
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))

    # ═══════════════════════════════════════════════════════════════════════════
    # MIXINS
    # ═══════════════════════════════════════════════════════════════════════════

    class IdentifiableMixin(BaseModel):
        """Mixin for unique identifiers."""

        model_config = ConfigDict(
            defer_build=True,
            arbitrary_types_allowed=True,
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        unique_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Unique identifier",
            min_length=1,
            frozen=False,
        )

        def regenerate_id(self) -> None:
            """Regenerate the unique_id with a new UUID."""
            self.unique_id = str(uuid.uuid4())

    class TimestampableMixin(BaseModel):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        model_config = ConfigDict(
            defer_build=True,
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        created_at: UTCDatetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Creation timestamp (UTC)",
            frozen=True,
        )
        updated_at: UTCDatetime | None = Field(
            default=None,
            description="Last update timestamp (UTC)",
        )

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

    class VersionableMixin(BaseModel):
        """Mixin for versioning with optimistic locking."""

        model_config = ConfigDict(
            defer_build=True,
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        version: int = Field(
            default=c.Performance.DEFAULT_VERSION,
            ge=c.Performance.MIN_VERSION,
            description="Version number for optimistic locking",
            frozen=False,
        )

        def increment_version(self) -> None:
            """Increment the version number."""
            self.version += 1

        @model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency."""
            if self.version < c.Performance.DEFAULT_VERSION:
                msg = f"Version {self.version} is below minimum {c.Performance.DEFAULT_VERSION}"
                raise ValueError(msg)
            return self

    class RetryConfigurationMixin(BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config = ConfigDict(populate_by_name=True)

        max_retries: int = Field(
            default=c.Reliability.DEFAULT_MAX_RETRIES,
            ge=c.ZERO,
            alias="max_attempts",
            description="Maximum retry attempts",
        )
        initial_delay_seconds: float = Field(
            default=c.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
            gt=c.ZERO,
            description="Initial delay between retries",
        )
        max_delay_seconds: float = Field(
            default=c.Reliability.RETRY_BACKOFF_MAX,
            gt=c.ZERO,
            description="Maximum delay between retries",
        )

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__ = ["FlextModelFoundation"]
