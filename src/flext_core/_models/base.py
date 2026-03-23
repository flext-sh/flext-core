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
from collections.abc import Callable, Mapping, Sequence
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

from flext_core import FlextUtilitiesGuardsTypeCore, c, t


class FlextModelFoundation:
    """Container for base model classes - Tier 0, 100% standalone."""

    @staticmethod
    def _ensure_utc_datetime(v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    class Validators:
        """Pydantic v2 validators - single namespace for all field validators."""

        _tags_adapter: ClassVar[TypeAdapter[Sequence[str]] | None] = None
        _list_adapter: ClassVar[TypeAdapter[Sequence[t.Container]] | None] = None
        _strict_string_adapter: ClassVar[
            TypeAdapter[Annotated[str, Field(strict=True)]] | None
        ] = None
        _metadata_map_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.MetadataValue]] | None
        ] = None
        _config_adapter: ClassVar[TypeAdapter[Mapping[str, t.Container]] | None] = None
        _dict_container_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.Container]] | None
        ] = None
        _list_container_adapter: ClassVar[TypeAdapter[Sequence[t.Container]] | None] = (
            None
        )
        _tuple_container_adapter: ClassVar[
            TypeAdapter[tuple[t.Container, ...]] | None
        ] = None
        _primitives_adapter: ClassVar[TypeAdapter[t.Primitives] | None] = None
        _dict_str_metadata_adapter: ClassVar[
            TypeAdapter[Mapping[str, t.NormalizedValue]] | None
        ] = None
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
        _float_adapter: ClassVar[TypeAdapter[float] | None] = None
        _str_adapter: ClassVar[TypeAdapter[str] | None] = None
        _str_list_adapter: ClassVar[TypeAdapter[Sequence[str]] | None] = None
        _str_or_bytes_adapter: ClassVar[TypeAdapter[str | bytes] | None] = None
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
        def config_adapter(cls) -> TypeAdapter[Mapping[str, t.Container]]:
            """Lazy-load config TypeAdapter on first access."""
            if cls._config_adapter is None:
                cls._config_adapter = TypeAdapter(Mapping[str, t.Container])
            return cls._config_adapter

        @classmethod
        def list_adapter(cls) -> TypeAdapter[Sequence[t.Container]]:
            """Lazy-load list TypeAdapter on first access."""
            if cls._list_adapter is None:
                cls._list_adapter = TypeAdapter(Sequence[t.Container])
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
        def tags_adapter(cls) -> TypeAdapter[Sequence[str]]:
            """Lazy-load tags TypeAdapter on first access."""
            if cls._tags_adapter is None:
                cls._tags_adapter = TypeAdapter(Sequence[str])
            return cls._tags_adapter

        @classmethod
        def dict_container_adapter(cls) -> TypeAdapter[Mapping[str, t.Container]]:
            """Lazy-load Mapping[str, Container] TypeAdapter on first access."""
            if cls._dict_container_adapter is None:
                cls._dict_container_adapter = TypeAdapter(Mapping[str, t.Container])
            return cls._dict_container_adapter

        @classmethod
        def list_container_adapter(cls) -> TypeAdapter[Sequence[t.Container]]:
            """Lazy-load Sequence[Container] TypeAdapter on first access."""
            if cls._list_container_adapter is None:
                cls._list_container_adapter = TypeAdapter(Sequence[t.Container])
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
        ) -> TypeAdapter[Mapping[str, t.NormalizedValue]]:
            if cls._dict_str_metadata_adapter is None:
                cls._dict_str_metadata_adapter = TypeAdapter(
                    Mapping[str, t.NormalizedValue],
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
        def float_adapter(cls) -> TypeAdapter[float]:
            if cls._float_adapter is None:
                cls._float_adapter = TypeAdapter(float)
            return cls._float_adapter

        @classmethod
        def str_adapter(cls) -> TypeAdapter[str]:
            if cls._str_adapter is None:
                cls._str_adapter = TypeAdapter(str)
            return cls._str_adapter

        @classmethod
        def str_list_adapter(cls) -> TypeAdapter[Sequence[str]]:
            if cls._str_list_adapter is None:
                cls._str_list_adapter = TypeAdapter(Sequence[str])
            return cls._str_list_adapter

        @classmethod
        def str_or_bytes_adapter(cls) -> TypeAdapter[str | bytes]:
            if cls._str_or_bytes_adapter is None:
                cls._str_or_bytes_adapter = TypeAdapter(str | bytes)
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
                    Mapping[str, t.Primitives]
                )
            return cls._metadata_json_dict_adapter

        @classmethod
        def flat_metadata_dict_adapter(
            cls,
        ) -> TypeAdapter[Mapping[str, t.Primitives]]:
            if cls._flat_metadata_dict_adapter is None:
                cls._flat_metadata_dict_adapter = TypeAdapter(
                    Mapping[str, t.Primitives]
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
            return FlextModelFoundation._ensure_utc_datetime(v)

        @staticmethod
        def normalize_to_list(v: t.ValueOrModel) -> Sequence[t.Container]:
            """Normalize value to list format."""
            try:
                return FlextModelFoundation.Validators.list_adapter().validate_python(v)
            except (TypeError, ValueError):
                if FlextUtilitiesGuardsTypeCore.is_scalar(v):
                    return [v]
                return [str(v)]

        @staticmethod
        def strip_whitespace(v: str) -> str:
            """Strip leading and trailing whitespace from string."""
            return v.strip()

        @staticmethod
        def validate_config_dict(
            v: t.ValueOrModel,
        ) -> Mapping[str, t.Container]:
            """Validate configuration dictionary structure."""
            try:
                normalized = (
                    FlextModelFoundation.Validators.config_adapter().validate_python(v)
                )
            except (TypeError, ValueError) as exc:
                msg = "Configuration must be a dictionary"
                raise TypeError(msg) from exc
            out: Mapping[str, t.Container] = {}
            for key, item in normalized.items():
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
                out[key] = item
            return out

        @staticmethod
        def validate_tags_list(v: t.ValueOrModel) -> Sequence[str]:
            """Validate and normalize tags list."""
            try:
                raw_tags: Sequence[t.Container] = (
                    FlextModelFoundation.Validators.list_adapter().validate_python(v)
                )
            except (TypeError, ValueError) as exc:
                msg = "Tags must be a list"
                raise TypeError(msg) from exc
            normalized: Sequence[str] = []
            seen: set[str] = set()
            for tag in raw_tags:
                try:
                    clean_tag = (
                        FlextModelFoundation.Validators
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

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra=c.EXTRA_FORBID,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class StrictBoundaryModel(BaseModel):
        """Strict boundary model for API/external boundaries."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            strict=True,
            validate_assignment=True,
            extra="forbid",
            str_strip_whitespace=True,
            use_enum_values=True,
            frozen=True,
        )

    class FlexibleInternalModel(BaseModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="ignore",
            str_strip_whitespace=True,
            use_enum_values=True,
        )

    class ImmutableValueModel(BaseModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

    class TaggedModel(BaseModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class DynamicConfigModel(BaseModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="allow",
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

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
                default_factory=lambda: datetime.now(UTC),
                description="Timestamp when the metadata record was first created in UTC.",
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime,
            Field(
                default_factory=lambda: datetime.now(UTC),
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
            Sequence[str],
            Field(
                default_factory=list,
                description="Normalized labels used to classify and filter this metadata.",
                title="Tags",
                examples=[["billing", "critical"]],
            ),
        ] = Field(default_factory=list)
        attributes: Annotated[
            Mapping[str, t.MetadataValue],
            Field(
                default_factory=dict,
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
                if not isinstance(result, Mapping):
                    msg = "attributes must dump to mapping"
                    raise TypeError(msg)
            elif isinstance(value, Mapping):
                result = dict(value.items())
            else:
                msg = "attributes must be dict-like"
                raise TypeError(msg)
            for key in result:
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
            return (
                FlextModelFoundation.Validators.metadata_map_adapter().validate_python(
                    result,
                )
            )

    class CommandMessage(BaseModel):
        """Command message with discriminated union support."""

        message_type: Literal["command"] = "command"
        command_type: str
        issuer_id: t.NonEmptyStr | None = None
        data: Annotated[
            t.Dict,
            Field(
                default_factory=t.Dict,
                description="Command payload containing input data required for execution.",
            ),
        ]

    class QueryMessage(BaseModel):
        """Query message with discriminated union support."""

        message_type: Literal["query"] = "query"
        query_type: str
        filters: Annotated[
            t.Dict,
            Field(
                default_factory=t.Dict,
                description="Filter criteria used to constrain query results.",
            ),
        ]
        pagination: t.Dict | None = None

    class EventMessage(BaseModel):
        """Event message with discriminated union support."""

        message_type: Literal["event"] = "event"
        event_type: t.NonEmptyStr
        aggregate_id: t.NonEmptyStr
        data: Annotated[
            t.Dict,
            Field(
                default_factory=t.Dict,
                description="Event payload with domain data describing what happened.",
            ),
        ]
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
        error_data: FlextModelFoundation.Metadata | None = None

    class PartialResult(BaseModel):
        """Partial result for discriminated union."""

        result_type: Literal["partial"] = "partial"
        value: t.Container
        warnings: Annotated[
            Sequence[str],
            Field(
                default_factory=list,
                description="Non-fatal warning messages generated during partial processing.",
            ),
        ]
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
        errors: Sequence[str]
        error_codes: Annotated[
            Sequence[str],
            Field(
                default_factory=list,
                description="Machine-readable error codes that classify validation failures.",
            ),
        ]

    class WarningOutcome(BaseModel):
        """Warning validation outcome."""

        outcome_type: Literal["warning"] = "warning"
        validated_data: t.Container
        warnings: Sequence[str]
        validation_time_ms: float

    ValidationOutcome = Annotated[
        ValidOutcome | InvalidOutcome | WarningOutcome,
        Discriminator("outcome_type"),
    ]

    class FrozenStrictModel(BaseModel):
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
        def __eq__(self, other: t.NormalizedValue) -> bool:
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
                default_factory=lambda: str(uuid.uuid4()),
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
            AfterValidator(lambda v: FlextModelFoundation._ensure_utc_datetime(v)),
            Field(
                default_factory=lambda: datetime.now(UTC),
                description="Creation timestamp (UTC)",
                frozen=True,
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime | None,
            AfterValidator(lambda v: FlextModelFoundation._ensure_utc_datetime(v)),
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
                default=c.DEFAULT_VERSION,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_VERSION

        def increment_version(self) -> None:
            """Increment the version number."""
            self.version += 1

        @model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency."""
            if self.version < c.DEFAULT_VERSION:
                msg = f"Version {self.version} is below minimum {c.DEFAULT_VERSION}"
                raise ValueError(msg)
            return self

    class RetryConfigurationMixin(BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)
        max_retries: Annotated[
            t.NonNegativeInt,
            Field(
                default=c.DEFAULT_MAX_RETRIES,
                alias="max_attempts",
                description="Maximum retry attempts",
            ),
        ] = c.DEFAULT_MAX_RETRIES
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
                default=c.RETRY_BACKOFF_MAX,
                description="Maximum delay between retries",
            ),
        ] = c.RETRY_BACKOFF_MAX

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__ = ["FlextModelFoundation"]
