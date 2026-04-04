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
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self, override

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from flext_core import FlextRuntime, c, t
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement


class FlextModelsBase:
    """Container for base model classes - Tier 0, 100% standalone."""

    @staticmethod
    def _ensure_utc_datetime(v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    class Validators:
        """Pydantic v2 validators - single namespace for all field validators."""

        _tags_adapter: ClassVar[TypeAdapter[t.StrSequence] | None] = None
        _list_adapter: ClassVar[TypeAdapter[t.FlatContainerList] | None] = None
        _strict_string_adapter: ClassVar[
            TypeAdapter[Annotated[str, Field(strict=True)]] | None
        ] = None
        _metadata_map_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.MetadataValue]] | None
        ] = None
        _config_adapter: ClassVar[TypeAdapter[t.FlatContainerMapping] | None] = None
        _dict_container_adapter: ClassVar[
            TypeAdapter[t.FlatContainerMapping] | None
        ] = None
        _list_container_adapter: ClassVar[TypeAdapter[t.FlatContainerList] | None] = (
            None
        )
        _tuple_container_adapter: ClassVar[
            TypeAdapter[tuple[t.Container, ...]] | None
        ] = None
        _primitives_adapter: ClassVar[TypeAdapter[t.Primitives] | None] = None
        _dict_str_metadata_adapter: ClassVar[TypeAdapter[t.ContainerMapping] | None] = (
            None
        )
        _list_serializable_adapter: ClassVar[
            TypeAdapter[Sequence[t.Serializable]] | None
        ] = None
        _tuple_serializable_adapter: ClassVar[
            TypeAdapter[tuple[t.Serializable, ...]] | None
        ] = None
        _set_container_adapter: ClassVar[TypeAdapter[set[t.Container]] | None] = None
        _set_str_adapter: ClassVar[TypeAdapter[set[str]] | None] = None
        _set_scalar_adapter: ClassVar[TypeAdapter[set[t.Scalar]] | None] = None
        _sortable_dict_adapter: ClassVar[
            TypeAdapter[Mapping[t.SortableObjectType, t.Serializable | None]] | None
        ] = None
        _strict_json_list_adapter: ClassVar[
            TypeAdapter[Sequence[t.StrictValue]] | None
        ] = None
        _strict_json_scalar_adapter: ClassVar[TypeAdapter[t.Scalar] | None] = None
        _scalar_adapter: ClassVar[TypeAdapter[t.Scalar] | None] = None
        _float_adapter: ClassVar[TypeAdapter[t.FloatValue] | None] = None
        _str_adapter: ClassVar[TypeAdapter[t.TextValue] | None] = None
        _str_list_adapter: ClassVar[TypeAdapter[t.StrSequence] | None] = None
        _str_or_bytes_adapter: ClassVar[TypeAdapter[t.TextOrBinaryContent] | None] = (
            None
        )
        _enum_type_adapter: ClassVar[TypeAdapter[type[StrEnum]] | None] = None
        _serializable_adapter: ClassVar[TypeAdapter[t.Serializable] | None] = None
        _metadata_json_dict_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.Primitives]] | None
        ] = None
        _flat_metadata_dict_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.Primitives]] | None
        ] = None
        _structlog_processor_adapter: ClassVar[
            TypeAdapter[Callable[..., t.Container]] | None
        ] = None

        @classmethod
        def config_adapter(cls) -> TypeAdapter[t.FlatContainerMapping]:
            """Lazy-load config TypeAdapter on first access."""
            if cls._config_adapter is None:
                cls._config_adapter = TypeAdapter(t.FlatContainerMapping)
            return cls._config_adapter

        @classmethod
        def list_adapter(cls) -> TypeAdapter[t.FlatContainerList]:
            """Lazy-load list TypeAdapter on first access."""
            if cls._list_adapter is None:
                cls._list_adapter = TypeAdapter(t.FlatContainerList)
            return cls._list_adapter

        @classmethod
        def metadata_map_adapter(
            cls,
        ) -> TypeAdapter[Mapping[str, t.MetadataValue]]:
            """Lazy-load metadata map TypeAdapter on first access."""
            if cls._metadata_map_adapter is None:
                cls._metadata_map_adapter = TypeAdapter(Mapping[str, t.MetadataValue])
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
        def tags_adapter(cls) -> TypeAdapter[t.StrSequence]:
            """Lazy-load tags TypeAdapter on first access."""
            if cls._tags_adapter is None:
                cls._tags_adapter = TypeAdapter(t.StrSequence)
            return cls._tags_adapter

        @classmethod
        def dict_container_adapter(cls) -> TypeAdapter[t.FlatContainerMapping]:
            """Lazy-load Mapping[str, Container] TypeAdapter on first access."""
            if cls._dict_container_adapter is None:
                cls._dict_container_adapter = TypeAdapter(t.FlatContainerMapping)
            return cls._dict_container_adapter

        @classmethod
        def list_container_adapter(cls) -> TypeAdapter[t.FlatContainerList]:
            """Lazy-load t.FlatContainerList TypeAdapter on first access."""
            if cls._list_container_adapter is None:
                cls._list_container_adapter = TypeAdapter(t.FlatContainerList)
            return cls._list_container_adapter

        @classmethod
        def tuple_container_adapter(cls) -> TypeAdapter[tuple[t.Container, ...]]:
            """Lazy-load tuple[Container, ...] TypeAdapter on first access."""
            if cls._tuple_container_adapter is None:
                cls._tuple_container_adapter = TypeAdapter(tuple[t.Container, ...])
            return cls._tuple_container_adapter

        @classmethod
        def primitives_adapter(cls) -> TypeAdapter[t.Primitives]:
            """Lazy-load Primitives TypeAdapter on first access."""
            if cls._primitives_adapter is None:
                cls._primitives_adapter = TypeAdapter(t.Primitives)
            return cls._primitives_adapter

        @classmethod
        def dict_str_metadata_adapter(
            cls,
        ) -> TypeAdapter[t.ContainerMapping]:
            if cls._dict_str_metadata_adapter is None:
                cls._dict_str_metadata_adapter = TypeAdapter(
                    t.ContainerMapping,
                )
            return cls._dict_str_metadata_adapter

        @classmethod
        def list_serializable_adapter(cls) -> TypeAdapter[Sequence[t.Serializable]]:
            if cls._list_serializable_adapter is None:
                cls._list_serializable_adapter = TypeAdapter(Sequence[t.Serializable])
            return cls._list_serializable_adapter

        @classmethod
        def tuple_serializable_adapter(cls) -> TypeAdapter[tuple[t.Serializable, ...]]:
            if cls._tuple_serializable_adapter is None:
                cls._tuple_serializable_adapter = TypeAdapter(
                    tuple[t.Serializable, ...],
                )
            return cls._tuple_serializable_adapter

        @classmethod
        def set_container_adapter(cls) -> TypeAdapter[set[t.Container]]:
            if cls._set_container_adapter is None:
                cls._set_container_adapter = TypeAdapter(set[t.Container])
            return cls._set_container_adapter

        @classmethod
        def set_str_adapter(cls) -> TypeAdapter[set[str]]:
            if cls._set_str_adapter is None:
                cls._set_str_adapter = TypeAdapter(set[str])
            return cls._set_str_adapter

        @classmethod
        def set_scalar_adapter(cls) -> TypeAdapter[set[t.Scalar]]:
            if cls._set_scalar_adapter is None:
                cls._set_scalar_adapter = TypeAdapter(set[t.Scalar])
            return cls._set_scalar_adapter

        @classmethod
        def sortable_dict_adapter(
            cls,
        ) -> TypeAdapter[Mapping[t.SortableObjectType, t.Serializable | None]]:
            if cls._sortable_dict_adapter is None:
                cls._sortable_dict_adapter = TypeAdapter(
                    Mapping[t.SortableObjectType, t.Serializable | None],
                )
            return cls._sortable_dict_adapter

        @classmethod
        def strict_json_list_adapter(
            cls,
        ) -> TypeAdapter[Sequence[t.StrictValue]]:
            if cls._strict_json_list_adapter is None:
                cls._strict_json_list_adapter = TypeAdapter(Sequence[t.StrictValue])
            return cls._strict_json_list_adapter

        @classmethod
        def strict_json_scalar_adapter(cls) -> TypeAdapter[t.Scalar]:
            if cls._strict_json_scalar_adapter is None:
                cls._strict_json_scalar_adapter = TypeAdapter(t.Scalar)
            return cls._strict_json_scalar_adapter

        @classmethod
        def scalar_adapter(cls) -> TypeAdapter[t.Scalar]:
            if cls._scalar_adapter is None:
                cls._scalar_adapter = TypeAdapter(t.Scalar)
            return cls._scalar_adapter

        @classmethod
        def float_adapter(cls) -> TypeAdapter[t.FloatValue]:
            if cls._float_adapter is None:
                cls._float_adapter = TypeAdapter(t.FloatValue)
            return cls._float_adapter

        @classmethod
        def str_adapter(cls) -> TypeAdapter[t.TextValue]:
            if cls._str_adapter is None:
                cls._str_adapter = TypeAdapter(t.TextValue)
            return cls._str_adapter

        @classmethod
        def str_list_adapter(cls) -> TypeAdapter[t.StrSequence]:
            if cls._str_list_adapter is None:
                cls._str_list_adapter = TypeAdapter(t.StrSequence)
            return cls._str_list_adapter

        @classmethod
        def str_or_bytes_adapter(cls) -> TypeAdapter[t.TextOrBinaryContent]:
            if cls._str_or_bytes_adapter is None:
                cls._str_or_bytes_adapter = TypeAdapter(t.TextOrBinaryContent)
            return cls._str_or_bytes_adapter

        @classmethod
        def enum_type_adapter(cls) -> TypeAdapter[type[StrEnum]]:
            if cls._enum_type_adapter is None:
                cls._enum_type_adapter = TypeAdapter(type[StrEnum])
            return cls._enum_type_adapter

        @classmethod
        def serializable_adapter(cls) -> TypeAdapter[t.Serializable]:
            if cls._serializable_adapter is None:
                cls._serializable_adapter = TypeAdapter(t.Serializable)
            return cls._serializable_adapter

        @classmethod
        def metadata_json_dict_adapter(
            cls,
        ) -> TypeAdapter[Mapping[str, t.Primitives]]:
            if cls._metadata_json_dict_adapter is None:
                cls._metadata_json_dict_adapter = TypeAdapter(
                    Mapping[str, t.Primitives],
                )
            return cls._metadata_json_dict_adapter

        @classmethod
        def flat_metadata_dict_adapter(
            cls,
        ) -> TypeAdapter[Mapping[str, t.Primitives]]:
            if cls._flat_metadata_dict_adapter is None:
                cls._flat_metadata_dict_adapter = TypeAdapter(
                    Mapping[str, t.Primitives],
                )
            return cls._flat_metadata_dict_adapter

        @classmethod
        def structlog_processor_adapter(
            cls,
        ) -> TypeAdapter[Callable[..., t.Container]]:
            if cls._structlog_processor_adapter is None:
                cls._structlog_processor_adapter = TypeAdapter(
                    Callable[..., t.Container],
                )
            return cls._structlog_processor_adapter

        @staticmethod
        def ensure_utc_datetime(v: datetime | None) -> datetime | None:
            """Ensure datetime is UTC timezone."""
            return FlextModelsBase._ensure_utc_datetime(v)

        @staticmethod
        def normalize_to_list(v: t.ValueOrModel) -> t.FlatContainerList:
            """Normalize value to list format."""
            try:
                return FlextModelsBase.Validators.list_adapter().validate_python(v)
            except (TypeError, ValueError):
                if FlextRuntime.is_scalar(v):
                    return [v]
                return [str(v)]

        @staticmethod
        def strip_whitespace(v: str) -> str:
            """Strip leading and trailing whitespace from string."""
            return v.strip()

        @staticmethod
        def validate_config_dict(
            v: t.ValueOrModel,
        ) -> t.FlatContainerMapping:
            """Validate configuration dictionary structure."""
            try:
                normalized = (
                    FlextModelsBase.Validators.config_adapter().validate_python(v)
                )
            except (TypeError, ValueError) as exc:
                msg = "Configuration must be a dictionary"
                raise TypeError(msg) from exc
            out: t.MutableFlatContainerMapping = {}
            for key, item in normalized.items():
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
                out[key] = item
            return out

        @staticmethod
        def validate_tags_list(v: t.ValueOrModel) -> t.StrSequence:
            """Validate and normalize tags list."""
            try:
                raw_tags: t.FlatContainerList = (
                    FlextModelsBase.Validators.list_adapter().validate_python(v)
                )
            except (TypeError, ValueError) as exc:
                msg = "Tags must be a list"
                raise TypeError(msg) from exc
            normalized: MutableSequence[str] = []
            seen: set[str] = set()
            for tag in raw_tags:
                try:
                    clean_tag = (
                        FlextModelsBase.Validators
                        .strict_string_adapter()
                        .validate_python(tag)
                        .strip()
                        .lower()
                    )
                except (TypeError, ValueError) as exc:
                    msg = "Tag must be string"
                    raise TypeError(msg) from exc
                if clean_tag and clean_tag not in seen:
                    normalized.append(clean_tag)
                    seen.add(clean_tag)
            return normalized

    class EnforcedModel(BaseModel):
        """Base model that enforces architectural rules on subclasses."""

        @classmethod
        @override
        def __pydantic_init_subclass__(cls, **kwargs: t.Container) -> None:
            super().__pydantic_init_subclass__(**kwargs)
            FlextUtilitiesEnforcement.run(cls)

    class DomainModel(EnforcedModel):
        """Reusable strict model preset for validated domain boundaries."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            validate_assignment=True,
            extra=c.EXTRA_FORBID,
            validate_default=True,
            use_enum_values=True,
            str_strip_whitespace=True,
        )

    class FrozenDomainModel(DomainModel):
        """Immutable strict domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class ArbitraryTypesModel(EnforcedModel):
        """Base model with arbitrary types support."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra=c.EXTRA_FORBID,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class StrictBoundaryModel(EnforcedModel):
        """Strict boundary model for API/external boundaries."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            validate_assignment=True,
            extra="forbid",
            str_strip_whitespace=True,
            use_enum_values=True,
            frozen=True,
        )

    class FlexibleInternalModel(EnforcedModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="ignore",
            str_strip_whitespace=True,
            use_enum_values=True,
        )

    class ImmutableValueModel(EnforcedModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

    class TaggedModel(EnforcedModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class DynamicConfigModel(EnforcedModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="allow",
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class DynamicDomainModel(DynamicConfigModel):
        """Dynamic domain model preset with string whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            str_strip_whitespace=True,
        )

    class FrozenDynamicDomainModel(DynamicDomainModel):
        """Immutable dynamic domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class Metadata(BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra=c.EXTRA_FORBID,
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

        @field_validator("attributes", mode="before")
        @classmethod
        def _validate_attributes(
            cls,
            value: t.MetadataValue | Mapping[str, t.MetadataValue] | BaseModel | None,
        ) -> Mapping[str, t.MetadataValue]:
            if value is None:
                return {}
            if isinstance(value, BaseModel):
                result = value.model_dump()
            elif isinstance(value, Mapping):
                result = dict(value)
            else:
                msg = "attributes must be dict-like"
                raise TypeError(msg)
            for key in result:
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
            return FlextModelsBase.Validators.metadata_map_adapter().validate_python(
                result,
            )

    class CommandMessage(BaseModel):
        """Command message with discriminated union support."""

        message_type: Literal["command"] = "command"
        command_type: str
        issuer_id: t.NonEmptyStr | None = None
        data: Annotated[
            t.Dict,
            Field(
                description="Command payload containing input data required for execution.",
            ),
        ] = Field(default_factory=t.Dict)

    class QueryMessage(BaseModel):
        """Query message with discriminated union support."""

        message_type: Literal["query"] = "query"
        query_type: str
        filters: Annotated[
            t.Dict,
            Field(
                description="Filter criteria used to constrain query results.",
            ),
        ] = Field(default_factory=t.Dict)
        pagination: t.Dict | None = None

    class EventMessage(BaseModel):
        """Event message with discriminated union support."""

        message_type: Literal["event"] = "event"
        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        data: Annotated[
            t.Dict,
            Field(
                description="Event payload with domain data describing what happened.",
            ),
        ] = Field(default_factory=t.Dict)
        metadata: Annotated[
            BaseModel | None,
            Field(
                default=None,
                description="Structured metadata associated with this event message.",
            ),
        ] = None

    MessageUnion = Annotated[
        CommandMessage | QueryMessage | EventMessage,
        Discriminator("message_type"),
    ]

    class SuccessResult(BaseModel):
        """Success result for discriminated union."""

        result_type: Literal["success"] = "success"
        value: t.Container
        metadata: Annotated[
            BaseModel | None,
            Field(
                default=None,
                description="Structured metadata attached to a successful operation result.",
            ),
        ] = None

    class FailureResult(BaseModel):
        """Failure result for discriminated union."""

        result_type: Literal["failure"] = "failure"
        error: str
        error_code: str | None = None
        error_data: FlextModelsBase.Metadata | None = None

    class PartialResult(BaseModel):
        """Partial result for discriminated union."""

        result_type: Literal["partial"] = "partial"
        value: t.Container
        warnings: Annotated[
            t.StrSequence,
            Field(
                description="Non-fatal warning messages generated during partial processing.",
            ),
        ] = Field(default_factory=list)
        partial_success_rate: t.Percentage

    OperationResult = Annotated[
        SuccessResult | FailureResult | PartialResult,
        Discriminator("result_type"),
    ]

    class ValidOutcome(BaseModel):
        """Valid validation outcome."""

        outcome_type: Literal["valid"] = "valid"
        validated_data: t.Container
        validation_time_ms: float

    class InvalidOutcome(BaseModel):
        """Invalid validation outcome."""

        outcome_type: Literal["invalid"] = "invalid"
        errors: t.StrSequence
        error_codes: Annotated[
            t.StrSequence,
            Field(
                description="Machine-readable error codes that classify validation failures.",
            ),
        ] = Field(default_factory=list)

    class WarningOutcome(BaseModel):
        """Warning validation outcome."""

        outcome_type: Literal["warning"] = "warning"
        validated_data: t.Container
        warnings: t.StrSequence
        validation_time_ms: float

    ValidationOutcome = Annotated[
        ValidOutcome | InvalidOutcome | WarningOutcome,
        Discriminator("outcome_type"),
    ]

    class FrozenStrictModel(EnforcedModel):
        """Immutable base model with strict validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra=c.EXTRA_FORBID,
            ser_json_timedelta=c.SERIALIZATION_ISO8601,
            ser_json_bytes=c.SERIALIZATION_BASE64,
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
            return hash(tuple(sorted(((k, str(v)) for k, v in data.items()))))

    class IdentifiableMixin(BaseModel):
        """Mixin for unique identifiers."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            str_strip_whitespace=True,
        )
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

    class TimestampableMixin(BaseModel):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )
        created_at: Annotated[
            datetime,
            AfterValidator(lambda v: FlextModelsBase._ensure_utc_datetime(v)),
            Field(
                description="Creation timestamp (UTC)",
                frozen=True,
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime | None,
            AfterValidator(lambda v: FlextModelsBase._ensure_utc_datetime(v)),
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

    class VersionableMixin(BaseModel):
        """Mixin for versioning with optimistic locking."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )
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
