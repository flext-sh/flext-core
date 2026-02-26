"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelFoundation
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import ModelWrapValidatorHandler

from flext_core.constants import c
from flext_core.typings import t


def _ensure_utc_datetime(v: datetime | None) -> datetime | None:
    """Ensure datetime is UTC timezone."""
    if v is not None and v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc_datetime)]


# Renamed to FlextModelFoundation for better clarity


class FlextModelFoundation:
    """Container for base model classes - Tier 0, 100% standalone."""

    class Validators:
        """Pydantic v2 validators - single namespace for all field validators."""

        tags_adapter: ClassVar[TypeAdapter[list[str]]] = TypeAdapter(list[str])
        list_adapter: ClassVar[TypeAdapter[list[t.GuardInputValue]]] = TypeAdapter(
            list[t.GuardInputValue]
        )
        strict_string_adapter: ClassVar[
            TypeAdapter[Annotated[str, Field(strict=True)]]
        ] = TypeAdapter(Annotated[str, Field(strict=True)])
        metadata_map_adapter: ClassVar[
            TypeAdapter[dict[str, t.MetadataAttributeValue]]
        ] = TypeAdapter(dict[str, t.MetadataAttributeValue])
        config_adapter: ClassVar[TypeAdapter[dict[str, t.GuardInputValue]]] = (
            TypeAdapter(dict[str, t.GuardInputValue])
        )

        @staticmethod
        def strip_whitespace(v: str) -> str:
            """Strip whitespace from string values."""
            return v.strip()

        @staticmethod
        def ensure_utc_datetime(v: datetime | None) -> datetime | None:
            """Ensure datetime is UTC timezone."""
            return _ensure_utc_datetime(v)

        @staticmethod
        def normalize_to_list(v: t.GuardInputValue) -> list[t.GuardInputValue]:
            """Normalize value to list format. Fixed types only."""
            try:
                return FlextModelFoundation.Validators.list_adapter.validate_python(v)
            except ValidationError:
                return [v]

        @staticmethod
        def validate_non_empty_string(v: str) -> str:
            """Validate that string is not empty after stripping."""
            stripped = v.strip()
            if not stripped:
                msg = "String cannot be empty or whitespace"
                raise ValueError(msg)
            return stripped

        @staticmethod
        def validate_email(v: str) -> str:
            """Validate email format using simple regex."""
            if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
                msg = "Invalid email format"
                raise ValueError(msg)
            return v

        @staticmethod
        def validate_url(v: str) -> str:
            """Validate URL format."""
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                msg = "Invalid URL format"
                raise ValueError(msg)
            return v

        @staticmethod
        def validate_semver(v: str) -> str:
            """Validate semantic version format."""
            if not re.match(r"^\d+\.\d+\.\d+(-[\w\.\-]+)?(\+[\w\.\-]+)?$", v):
                msg = "Invalid semantic version format"
                raise ValueError(msg)
            return v

        @staticmethod
        def validate_uuid_string(v: str) -> str:
            """Validate UUID string format."""
            try:
                _ = uuid.UUID(v)
                return v
            except (ValueError, TypeError):
                msg = "Invalid UUID format"
                raise ValueError(msg) from None

        @staticmethod
        def validate_config_dict(
            v: t.GuardInputValue,
        ) -> Mapping[str, t.GuardInputValue]:
            """Validate configuration dictionary structure. Returns dict for model storage."""
            try:
                normalized = (
                    FlextModelFoundation.Validators.config_adapter.validate_python(v)
                )
            except ValidationError as exc:
                msg = "Configuration must be a dictionary"
                raise TypeError(msg) from exc

            out = {}
            for key, item in normalized.items():
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)
                out[key] = item
            return out

        @staticmethod
        def validate_tags_list(v: t.GuardInputValue) -> list[str]:
            """Validate and normalize tags list."""
            try:
                raw_tags = FlextModelFoundation.Validators.list_adapter.validate_python(
                    v
                )
            except ValidationError as exc:
                msg = "Tags must be a list"
                raise TypeError(msg) from exc

            normalized: list[str] = []
            seen: set[str] = set()
            for tag in raw_tags:
                try:
                    clean_tag = (
                        FlextModelFoundation.Validators.strict_string_adapter
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

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra=c.ModelConfig.EXTRA_FORBID,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class StrictBoundaryModel(BaseModel):
        """Strict boundary model for API/external boundaries.

        Enforces strict validation, forbids extra fields, strips whitespace,
        and is immutable (frozen). Use at system boundaries where external
        data enters the system.
        """

        model_config = ConfigDict(
            strict=True,
            validate_assignment=True,
            extra="forbid",
            str_strip_whitespace=True,
            use_enum_values=True,
            frozen=True,
        )

    class FlexibleInternalModel(BaseModel):
        """Flexible internal model for domain logic.

        Allows assignment validation and whitespace stripping but ignores
        extra fields. Use for internal domain models where flexibility is needed.
        """

        model_config = ConfigDict(
            validate_assignment=True,
            extra="ignore",
            str_strip_whitespace=True,
            use_enum_values=True,
        )

    class ImmutableValueModel(BaseModel):
        """Immutable value model for value objects.

        Frozen and strict, forbids extra fields. Use for value objects
        that should never change after creation.
        """

        model_config = ConfigDict(
            frozen=True,
            validate_assignment=True,
            extra="forbid",
        )

    class TaggedModel(BaseModel):
        """Base pattern for tagged discriminated unions.

        Downstream models should define a `Literal[...]` runtime discriminator field
        (for example, `message_type`) plus a static class-level `tag` marker.
        This keeps union routing explicit and avoids ad-hoc `isinstance` trees.
        """

        model_config = ConfigDict(extra="forbid")
        tag: ClassVar[str]

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED PYDANTIC v2 FEATURES - Discriminated Unions
    # ═══════════════════════════════════════════════════════════════════════════

    class Metadata(BaseModel):
        """Standard metadata model.

        Business Rule: Provides standard metadata fields (timestamps, audit info,
        tags, attributes) for all entities. Enforces strict validation with frozen
        instances for immutability and thread safety.
        """

        model_config = ConfigDict(
            extra=c.ModelConfig.EXTRA_FORBID,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
            arbitrary_types_allowed=True,
        )

        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        version: str = Field(default="1.0.0")
        created_by: str | None = Field(default=None)
        modified_by: str | None = Field(default=None)
        tags: list[str] = Field(default_factory=list)
        attributes: Mapping[str, t.MetadataAttributeValue] = Field(default_factory=dict)

        @field_validator("attributes", mode="before")
        @classmethod
        def _validate_attributes(
            cls,
            value: t.MetadataAttributeValue
            | Mapping[str, t.MetadataAttributeValue]
            | None,
        ) -> Mapping[str, t.MetadataAttributeValue]:
            if value is None:
                return {}

            try:
                result = t.Dict.model_validate(value).root
            except ValidationError:
                if not isinstance(value, BaseModel):
                    msg = "attributes must be dict-like"
                    raise TypeError(msg) from None

                dumped = value.model_dump()
                try:
                    result = t.Dict.model_validate(dumped).root
                except ValidationError as exc:
                    msg = "attributes BaseModel must dump to mapping"
                    raise TypeError(msg) from exc

            for key in result:
                if key.startswith("_"):
                    msg = f"Keys starting with '_' are reserved: {key}"
                    raise ValueError(msg)

            return FlextModelFoundation.Validators.metadata_map_adapter.validate_python(
                result
            )

    # Command message type
    class CommandMessage(BaseModel):
        """Command message with discriminated union support."""

        message_type: Literal["command"] = "command"
        command_type: str
        issuer_id: str | None = None
        data: t.Dict = Field(default_factory=t.Dict)

    # Query message type
    class QueryMessage(BaseModel):
        """Query message with discriminated union support."""

        message_type: Literal["query"] = "query"
        query_type: str
        filters: t.Dict = Field(default_factory=t.Dict)
        pagination: t.Dict | None = None

    # Event message type
    class EventMessage(BaseModel):
        """Event message with discriminated union support."""

        message_type: Literal["event"] = "event"
        event_type: str
        aggregate_id: str
        data: t.Dict = Field(default_factory=t.Dict)
        metadata: FlextModelFoundation.Metadata = Field(
            default_factory=lambda: FlextModelFoundation.Metadata(),
        )

    # Discriminated union of all message types (defined after all classes)
    MessageUnion = Annotated[
        CommandMessage | QueryMessage | EventMessage, Discriminator("message_type")
    ]

    # Success result type
    class SuccessResult(BaseModel):
        """Success result for discriminated union."""

        result_type: Literal["success"] = "success"
        value: t.GuardInputValue
        metadata: FlextModelFoundation.Metadata = Field(
            default_factory=lambda: FlextModelFoundation.Metadata(),
        )

    # Failure result type
    class FailureResult(BaseModel):
        """Failure result for discriminated union."""

        result_type: Literal["failure"] = "failure"
        error: str
        error_code: str | None = None
        error_data: FlextModelFoundation.Metadata | None = None

    # Partial result type
    class PartialResult(BaseModel):
        """Partial result for discriminated union."""

        result_type: Literal["partial"] = "partial"
        value: t.GuardInputValue
        warnings: list[str] = Field(default_factory=list)
        partial_success_rate: float

    # Discriminated union of operation results
    OperationResult = Annotated[
        SuccessResult | FailureResult | PartialResult, Discriminator("result_type")
    ]

    # Valid outcome type
    class ValidOutcome(BaseModel):
        """Valid validation outcome."""

        outcome_type: Literal["valid"] = "valid"
        validated_data: t.GuardInputValue
        validation_time_ms: float

    # Invalid outcome type
    class InvalidOutcome(BaseModel):
        """Invalid validation outcome."""

        outcome_type: Literal["invalid"] = "invalid"
        errors: list[str]
        error_codes: list[str] = Field(default_factory=list)

    # Warning outcome type
    class WarningOutcome(BaseModel):
        """Warning validation outcome."""

        outcome_type: Literal["warning"] = "warning"
        validated_data: t.GuardInputValue
        warnings: list[str]
        validation_time_ms: float

    # Discriminated union of validation outcomes
    ValidationOutcome = Annotated[
        ValidOutcome | InvalidOutcome | WarningOutcome, Discriminator("outcome_type")
    ]

    class DynamicConfigModel(BaseModel):
        """Model for dynamic configuration - allows extra fields.

        Use this for parameters, filters, context, and other dynamic data
        where the exact fields are not known at compile time.
        """

        model_config = ConfigDict(
            validate_assignment=True,
            extra="allow",
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

    class FrozenStrictModel(BaseModel):
        """Immutable base model with strict validation."""

        model_config = ConfigDict(
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

        def __eq__(self, other: object) -> bool:
            """Compare by value using model_dump.

            Returns:
                bool: True if models are equal by value, False otherwise.

            """
            if not isinstance(other, type(self)):
                return NotImplemented
            return bool(self.model_dump() == other.model_dump())

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dicts.

            Returns:
                int: Hash value based on model's field values.

            """
            data = self.model_dump()
            return hash(tuple(sorted((k, str(v)) for k, v in data.items())))

    class IdentifiableMixin(BaseModel):
        """Mixin for unique identifiers with Pydantic v2 validation."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        unique_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Unique identifier for the model instance",
            min_length=1,
            frozen=False,  # Allow regeneration
        )

        @computed_field
        def id_short(self) -> str:
            """Get short version of ID (first 8 characters)."""
            return self.unique_id[:8]

        @computed_field
        def id_prefix(self) -> str | None:
            """Extract prefix from ID if it contains a separator."""
            separators = ["-", "_", ":"]
            for sep in separators:
                if sep in self.unique_id:
                    return self.unique_id.split(sep, 1)[0]
            return None

        @computed_field
        def is_uuid_format(self) -> bool:
            """Check if unique_id follows UUID format."""
            try:
                _ = uuid.UUID(self.unique_id)
                return True
            except (ValueError, TypeError):
                return False

        @computed_field
        def id_hash(self) -> int:
            """Get hash of the ID for use in sets/dicts."""
            return hash(self.unique_id)

        def regenerate_id(self) -> None:
            """Regenerate the unique_id with a new UUID."""
            self.unique_id = str(uuid.uuid4())

        @model_validator(mode="after")
        def validate_id_consistency(self) -> Self:
            """Ensure ID consistency after model creation."""
            # Additional validation can be added here
            return self

    class TimestampableMixin(BaseModel):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        created_at: UTCDatetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the model was created (UTC timezone)",
            frozen=True,  # Creation time should never change
        )
        updated_at: UTCDatetime | None = Field(
            default=None,
            description="Timestamp when the model was last updated (UTC timezone)",
        )

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 format for JSON.

            Note: Pydantic @field_serializer requires instance method signature
            even if self is not used in the implementation.

            Returns:
                str | None: ISO 8601 formatted timestamp string, or None if
                    value is None.

            """
            return value.isoformat() if value else None

        @computed_field
        def is_modified(self) -> bool:
            """Check if the model has been modified after creation.

            Returns:
                bool: True if updated_at is not None, False otherwise.

            """
            return self.updated_at is not None

        @property
        def age_seconds(self) -> float:
            """Calculate age of the model in seconds since creation.

            Returns:
                float: Age in seconds.

            """
            now = datetime.now(UTC)
            return (now - self.created_at).total_seconds()

        @property
        def age_minutes(self) -> float:
            """Calculate age of the model in minutes since creation.

            Returns:
                float: Age in minutes.

            """
            now = datetime.now(UTC)
            age_seconds = (now - self.created_at).total_seconds()
            return age_seconds / 60.0

        @property
        def age_hours(self) -> float:
            """Calculate age of the model in hours since creation.

            Returns:
                float: Age in hours.

            """
            now = datetime.now(UTC)
            age_seconds = (now - self.created_at).total_seconds()
            return age_seconds / 3600.0

        @property
        def age_days(self) -> float:
            """Calculate age of the model in days since creation.

            Returns:
                float: Age in days.

            """
            now = datetime.now(UTC)
            age_seconds = (now - self.created_at).total_seconds()
            return age_seconds / 86400.0

        @property
        def last_modified_age_seconds(self) -> float | None:
            """Calculate age since last modification in seconds.

            Returns:
                float | None: Age in seconds since last modification, or None if never modified.

            """
            if self.updated_at is None:
                return None
            now = datetime.now(UTC)
            return (now - self.updated_at).total_seconds()

        @computed_field
        def time_since_creation_formatted(self) -> str:
            """Get human-readable time since creation.

            Returns:
                str: Formatted string like "2d 3h 45m".

            """
            now = datetime.now(UTC)
            age = (now - self.created_at).total_seconds()
            days = int(age // 86400)
            hours = int((age % 86400) // 3600)
            minutes = int((age % 3600) // 60)

            parts: list[str] = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0 or days > 0:
                parts.append(f"{hours}h")
            parts.append(f"{minutes}m")

            return " ".join(parts)

        @property
        def is_recent(self) -> bool:
            """Check if the model was created within the last hour.

            Returns:
                bool: True if created within last hour.

            """
            now = datetime.now(UTC)
            age_minutes = (now - self.created_at).total_seconds() / 60.0
            return age_minutes <= c.Performance.RECENT_THRESHOLD_MINUTES

        @property
        def is_very_recent(self) -> bool:
            """Check if the model was created within the last 5 minutes.

            Returns:
                bool: True if created within last 5 minutes.

            """
            now = datetime.now(UTC)
            age_minutes = (now - self.created_at).total_seconds() / 60.0
            return age_minutes <= c.Performance.VERY_RECENT_THRESHOLD_MINUTES

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

        def touch(self) -> None:
            """Alias for update_timestamp for convenience."""
            self.update_timestamp()

        @model_validator(mode="after")
        def validate_timestamp_consistency(self) -> Self:
            """Validate timestamp consistency."""
            if self.updated_at is not None and self.updated_at < self.created_at:
                msg = "updated_at cannot be before created_at"
                raise ValueError(msg)
            return self

    class VersionableMixin(BaseModel):
        """Mixin for versioning with Pydantic v2 validation."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        version: int = Field(
            default=c.Performance.DEFAULT_VERSION,
            ge=c.Performance.MIN_VERSION,
            description="Version number for optimistic locking",
            frozen=False,  # Allow version changes
        )

        @computed_field
        def is_initial_version(self) -> bool:
            """Check if this is the initial version (version 1).

            Returns:
                bool: True if version equals DEFAULT_VERSION, False otherwise.

            """
            return self.version == c.Performance.DEFAULT_VERSION

        @computed_field
        def version_string(self) -> str:
            """Get version as string (e.g., "v1", "v2").

            Returns:
                str: Version formatted as string.

            """
            return f"v{self.version}"

        @computed_field
        def is_even_version(self) -> bool:
            """Check if version number is even.

            Returns:
                bool: True if version is even.

            """
            return self.version % 2 == 0

        @computed_field
        def is_odd_version(self) -> bool:
            """Check if version number is odd.

            Returns:
                bool: True if version is odd.

            """
            return self.version % 2 == 1

        @computed_field
        def version_category(self) -> str:
            """Categorize version number.

            Returns:
                str: Category ("initial", "low", "medium", "high").

            """
            if self.version == c.Performance.DEFAULT_VERSION:
                return "initial"
            if self.version <= c.Performance.VERSION_LOW_THRESHOLD:
                return "low"
            if self.version <= c.Performance.VERSION_MEDIUM_THRESHOLD:
                return "medium"
            return "high"

        def increment_version(self) -> None:
            """Increment the version number for optimistic locking."""
            self.version += 1

        def set_version(self, new_version: int) -> None:
            """Set version to a specific value.

            Args:
                new_version: The new version number.

            Raises:
                ValueError: If new_version is less than MIN_VERSION.

            """
            if new_version < c.Performance.MIN_VERSION:
                msg = f"Version must be >= {c.Performance.MIN_VERSION}"
                raise ValueError(msg)
            self.version = new_version

        def reset_to_initial_version(self) -> None:
            """Reset version to initial value (DEFAULT_VERSION)."""
            self.version = c.Performance.DEFAULT_VERSION

        @model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency after model creation."""
            # Version should not be less than default
            if self.version < c.Performance.DEFAULT_VERSION:
                msg = f"Version {self.version} is below minimum allowed {c.Performance.DEFAULT_VERSION}"
                raise ValueError(msg)
            return self

    class AuditableMixin(BaseModel):
        """Mixin for audit trail tracking with Pydantic v2 validation."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        created_by: str | None = Field(
            default=None,
            description="User/system that created this record",
            min_length=1,
        )
        updated_by: str | None = Field(
            default=None,
            description="User/system that last updated this record",
            min_length=1,
        )
        created_at: UTCDatetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when record was created (UTC timezone)",
            frozen=True,
        )
        updated_at: UTCDatetime | None = Field(
            default=None,
            description="Timestamp when record was last updated (UTC timezone)",
        )

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_audit_timestamps(self, value: datetime | None) -> str | None:
            """Serialize audit timestamps to ISO 8601 format for JSON.

            Note: Pydantic @field_serializer requires instance method signature
            even if self is not used in the implementation.

            Returns:
                str | None: ISO 8601 formatted timestamp string, or None if
                    value is None.

            """
            return value.isoformat() if value else None

        @computed_field
        def has_audit_info(self) -> bool:
            """Check if audit information is available.

            Returns:
                bool: True if created_by is set.

            """
            return self.created_by is not None

        @computed_field
        def was_modified_by_different_user(self) -> bool:
            """Check if creation and last update were by different users.

            Returns:
                bool: True if created_by != updated_by and both are set.

            """
            return (
                self.created_by is not None
                and self.updated_by is not None
                and self.created_by != self.updated_by
            )

        @computed_field
        def audit_summary(self) -> str:
            """Generate audit summary string.

            Returns:
                str: Formatted audit summary.

            """
            parts = []
            if self.created_by:
                parts.append(f"created by {self.created_by}")
            if self.updated_by and self.updated_by != self.created_by:
                parts.append(f"updated by {self.updated_by}")
            return " | ".join(parts) if parts else "no audit info"

        def set_created_by(self, user: str) -> None:
            """Set the creator of this record.

            Args:
                user: User/system identifier.

            """
            self.created_by = user

        def set_updated_by(self, user: str) -> None:
            """Set the last updater of this record and update timestamp.

            Args:
                user: User/system identifier.

            """
            self.updated_by = user
            self.updated_at = datetime.now(UTC)

        def audit_update(self, user: str) -> None:
            """Convenience method to update audit info.

            Args:
                user: User/system performing the update.

            """
            self.set_updated_by(user)

        @model_validator(mode="after")
        def validate_audit_consistency(self) -> Self:
            """Validate audit field consistency."""
            # If we have updated_at, we should have updated_by
            if self.updated_at is not None and self.updated_by is None:
                msg = "updated_at set but updated_by is None"
                raise ValueError(msg)
            # created_by should always be set if created_at is set
            if self.created_by is None:
                msg = "created_by must be set for auditable records"
                raise ValueError(msg)
            return self

    class SoftDeletableMixin(BaseModel):
        """Mixin for soft delete functionality with Pydantic v2 validation."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            str_strip_whitespace=True,
        )

        deleted_at: UTCDatetime | None = Field(
            default=None,
            description="Timestamp when record was soft deleted (UTC timezone)",
        )
        deleted_by: str | None = Field(
            default=None,
            description="User/system that performed the soft delete",
            min_length=1,
        )
        is_deleted: bool = Field(
            default=False,
            description="Flag indicating if record is soft deleted",
        )

        @field_serializer("deleted_at", when_used="json")
        def serialize_deleted_at(self, value: datetime | None) -> str | None:
            """Serialize deleted_at timestamp to ISO 8601 format for JSON.

            Note: Pydantic @field_serializer requires instance method signature
            even if self is not used in the implementation.

            Returns:
                str | None: ISO 8601 formatted timestamp string, or None if
                    value is None.

            """
            return value.isoformat() if value else None

        @computed_field
        def is_active(self) -> bool:
            """Check if record is active (not soft deleted).

            Returns:
                bool: True if record is not deleted.

            """
            return not self.is_deleted

        @computed_field
        def can_be_restored(self) -> bool:
            """Check if record can be restored (was soft deleted).

            Returns:
                bool: True if record was soft deleted and can be restored.

            """
            return self.is_deleted and self.deleted_at is not None

        def soft_delete(self, deleted_by: str | None = None) -> None:
            """Mark record as soft deleted.

            Args:
                deleted_by: User/system performing the deletion.

            """
            # Create new instance with all fields set correctly
            now = datetime.now(UTC)
            current_data = dict(self.model_dump())
            current_data.update({
                "is_deleted": True,
                "deleted_at": now,
                "deleted_by": deleted_by,
            })

            # Replace current instance data using __dict__.update to bypass
            # intermediate validation states during assignment
            validated = self.__class__.model_validate(current_data)
            for key, value in validated.__dict__.items():
                setattr(self, key, value)

        def restore(self) -> None:
            """Restore a soft deleted record."""
            self.is_deleted = False
            self.deleted_at = None
            self.deleted_by = None

        @model_validator(mode="after")
        def validate_soft_delete_consistency(self) -> Self:
            """Validate soft delete field consistency."""
            # If is_deleted is True, we should have deleted_at
            if self.is_deleted and self.deleted_at is None:
                msg = "is_deleted=True but deleted_at is None"
                raise ValueError(msg)

            # If deleted_at is set, is_deleted should be True
            if self.deleted_at is not None and not self.is_deleted:
                msg = "deleted_at is set but is_deleted=False"
                raise ValueError(msg)

            return self

    class TaggableMixin(BaseModel):
        """Mixin for tagging and categorization with Pydantic v2 validation."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        tags: list[str] = Field(
            default_factory=list,
            description="List of tags associated with this record",
        )
        categories: list[str] = Field(
            default_factory=list,
            description="List of categories for this record",
        )
        labels: MutableMapping[str, str] = Field(
            default_factory=dict,
            description="Key-value labels for flexible categorization",
        )

        @field_validator("tags", "categories")
        @classmethod
        def validate_tag_list(cls, v: list[str]) -> list[str]:
            """Validate tag and category lists."""
            # Remove duplicates and empty strings
            cleaned = []
            seen = set()
            for item in v:
                cleaned_item = item.strip()
                if cleaned_item and cleaned_item not in seen:
                    cleaned.append(cleaned_item)
                    seen.add(cleaned_item)
            return cleaned

        @field_validator("labels")
        @classmethod
        def validate_labels(cls, v: Mapping[str, str]) -> Mapping[str, str]:
            """Validate labels dictionary."""
            cleaned = {}
            for key, value in v.items():
                cleaned_key = key.strip()
                cleaned_value = value.strip()
                if cleaned_key and cleaned_value:
                    cleaned[cleaned_key] = cleaned_value
            return cleaned

        @property
        def tag_count(self) -> int:
            """Get number of tags.

            Returns:
                int: Number of tags.

            """
            return len(self.tags)

        @property
        def category_count(self) -> int:
            """Get number of categories.

            Returns:
                int: Number of categories.

            """
            return len(self.categories)

        @computed_field
        def has_tags(self) -> bool:
            """Check if record has any tags.

            Returns:
                bool: True if has at least one tag.

            """
            return len(self.tags) > 0

        @property
        def has_categories(self) -> bool:
            """Check if record has any categories.

            Returns:
                bool: True if has at least one category.

            """
            return len(self.categories) > 0

        @computed_field
        def all_labels(self) -> list[str]:
            """Get all label keys.

            Returns:
                list[str]: List of label keys.

            """
            return list(self.labels.keys())

        def add_tag(self, tag: str) -> None:
            """Add a tag if not already present.

            Args:
                tag: Tag to add.

            """
            if tag not in self.tags:
                self.tags.append(tag)

        def remove_tag(self, tag: str) -> None:
            """Remove a tag if present.

            Args:
                tag: Tag to remove.

            """
            if tag in self.tags:
                self.tags.remove(tag)

        def has_tag(self, tag: str) -> bool:
            """Check if record has a specific tag.

            Args:
                tag: Tag to check for.

            Returns:
                bool: True if tag is present.

            """
            return tag in self.tags

        def add_category(self, category: str) -> None:
            """Add a category if not already present.

            Args:
                category: Category to add.

            """
            if category not in self.categories:
                self.categories.append(category)

        def remove_category(self, category: str) -> None:
            """Remove a category if present.

            Args:
                category: Category to remove.

            """
            if category in self.categories:
                self.categories.remove(category)

        def has_category(self, category: str) -> bool:
            """Check if record has a specific category.

            Args:
                category: Category to check for.

            Returns:
                bool: True if category is present.

            """
            return category in self.categories

        def set_label(self, key: str, value: str) -> None:
            """Set a label key-value pair.

            Args:
                key: Label key.
                value: Label value.

            """
            self.labels[key] = value

        def get_label(self, key: str, default: str | None = None) -> str | None:
            """Get a label value.

            Args:
                key: Label key.
                default: Default value if key not found.

            Returns:
                str | None: Label value or default.

            """
            return self.labels.get(key, default)

        def remove_label(self, key: str) -> None:
            """Remove a label if present.

            Args:
                key: Label key to remove.

            """
            _ = self.labels.pop(key, None)

        @model_validator(mode="after")
        def validate_tag_consistency(self) -> Self:
            """Validate tag field consistency."""
            # Ensure no overlap between tags and labels keys
            label_keys = set(self.labels.keys())
            tag_set = set(self.tags)
            overlap = label_keys & tag_set
            if overlap:
                msg = f"Tags and labels cannot have overlapping keys: {overlap}"
                raise ValueError(msg)
            return self

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED MIXINS - Using Pydantic v2 Advanced Features
    # ═══════════════════════════════════════════════════════════════════════════

    class RetryConfigurationMixin(BaseModel):
        """Mixin for shared retry configuration properties.

        Business Rule: Consolidates common retry parameters across service
        and configuration domains to ensure consistent reliability behavior.
        """

        max_retries: int = Field(
            default=c.Reliability.DEFAULT_MAX_RETRIES,
            ge=c.ZERO,
            validation_alias=AliasChoices("max_retries", "max_attempts"),
            description="Maximum number of retry attempts",
        )
        initial_delay_seconds: float = Field(
            default=c.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
            gt=c.ZERO,
            description="Initial delay between retries in seconds",
        )
        max_delay_seconds: float = Field(
            default=c.Reliability.RETRY_BACKOFF_MAX,
            gt=c.ZERO,
            description="Maximum delay between retries in seconds",
        )

    class ValidatableMixin(BaseModel):
        """Mixin providing advanced validation capabilities using Pydantic v2."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        @model_validator(mode="after")
        def validate_business_rules(self) -> Self:
            """Override this method to add custom business rule validation."""
            return self

        @model_validator(mode="wrap")
        @classmethod
        def validate_performance(
            cls,
            value: t.GuardInputValue,
            handler: ModelWrapValidatorHandler[Self],
        ) -> Self:
            start_time = time.perf_counter()
            model = handler(value)
            elapsed_ms = (time.perf_counter() - start_time) * c.MILLISECONDS_MULTIPLIER
            if elapsed_ms > c.Validation.VALIDATION_TIMEOUT_MS:
                msg = (
                    f"Validation too slow: {elapsed_ms:.2f}ms > "
                    f"{c.Validation.VALIDATION_TIMEOUT_MS}ms"
                )
                raise ValueError(msg)
            return model

        @classmethod
        def validate_batch(cls, items: Sequence[t.GuardInputValue]) -> list[Self]:
            try:
                return [cls.model_validate(item) for item in items]
            except ValidationError as exc:
                item_errors: list[str] = []
                for error in exc.errors():
                    location = ".".join(str(part) for part in error.get("loc", ()))
                    message = str(error.get("msg", "validation error"))
                    item_errors.append(f"{location}: {message}")
                msg = f"Batch validation failed: {'; '.join(item_errors)}"
                raise ValueError(msg) from exc

        def validate_self(self) -> Self:
            """Re-validate the model instance."""
            return self.model_copy(deep=True)

        def is_valid(self) -> bool:
            """Check if the model is currently valid."""
            try:
                _ = self.validate_self()
                return True
            except Exception:
                return False

        def get_validation_errors(self) -> list[str]:
            """Get list of validation errors."""
            try:
                _ = self.validate_self()
                return []
            except Exception as e:
                return [str(e)]

    class SerializableMixin(BaseModel):
        """Mixin providing advanced serialization capabilities."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        def to_dict(
            self, *, exclude_none: bool = True, exclude_unset: bool = False
        ) -> Mapping[str, t.GuardInputValue]:
            """Convert to dictionary with advanced options."""
            return self.model_dump(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
            )

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON string."""
            return self.model_dump_json(indent=indent)

        @classmethod
        def from_dict(cls, data: Mapping[str, t.GuardInputValue]) -> Self:
            """Create instance from dictionary."""
            return cls.model_validate(data)

        @classmethod
        def from_json(cls, json_str: str) -> Self:
            """Create instance from JSON string."""
            return cls.model_validate_json(json_str)

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED PYDANTIC v2 FEATURES - Custom Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    class AdvancedSerializable(BaseModel):
        """Model demonstrating advanced serialization capabilities."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        metadata: t.Dict = Field(default_factory=t.Dict)

        @field_serializer("timestamp")
        def serialize_timestamp_iso(self, value: datetime) -> str:
            """Serialize timestamp to ISO format."""
            return value.isoformat()

        @field_serializer("metadata", when_used="json")
        def serialize_metadata_clean(self, value: t.Dict) -> Mapping[str, str]:
            """Serialize metadata with string conversion."""
            return {k: str(v) for k, v in value.root.items()}

        def to_json_multiple_formats(self) -> Mapping[str, str]:
            """Export in multiple JSON formats."""
            return {
                "iso_timestamps": self.model_dump_json(),
                "compact": self.model_dump_json(indent=None),
                "unix_timestamp": str(self.timestamp.timestamp()),
            }

        @computed_field
        def formatted_name(self) -> str:
            """Formatted version of name."""
            return f"[{self.name.upper()}]"

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED PYDANTIC v2 FEATURES - Dynamic Model Reconstruction
    # ═══════════════════════════════════════════════════════════════════════════

    class DynamicRebuildModel(BaseModel):
        """Model demonstrating dynamic schema reconstruction with model_rebuild()."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        value: int

        @computed_field
        def doubled_value(self) -> int:
            """Computed field that can be dynamically modified."""
            return self.value * 2

        @classmethod
        def create_with_extra_field(
            cls, field_name: str, field_type: type
        ) -> type[FlextModelFoundation.DynamicRebuildModel]:
            """Create a new model class with an additional field."""
            # Create new annotations dict
            new_annotations = cls.__annotations__.copy()
            new_annotations[field_name] = field_type

            # Create new class dynamically using type() for type-checker compatibility
            new_model: type[FlextModelFoundation.DynamicRebuildModel] = type(
                f"{cls.__name__}WithExtra",
                (cls,),
                {"__annotations__": new_annotations},
            )

            return new_model

        def add_runtime_field(self, name: str, value: t.GuardInputValue) -> None:
            """Add a field at runtime (stored in __dict__)."""
            setattr(self, name, value)

        def get_runtime_field(
            self, name: str, default: t.GuardInputValue = None
        ) -> t.GuardInputValue:
            """Get a runtime field value."""
            if hasattr(self, name):
                return super().__getattribute__(name)
            return default

        @classmethod
        def rebuild_with_validator(
            cls, validator_func: Callable[[t.GuardInputValue], t.GuardInputValue]
        ) -> type[FlextModelFoundation.DynamicRebuildModel]:
            """Rebuild model with additional validator."""

            # Create validator function (without decorators)
            def custom_validator(
                _cls: type,
                v: t.GuardInputValue,
            ) -> t.GuardInputValue:
                """Apply custom validator to value field."""
                return validator_func(v)

            # Create new class dynamically using type() for type-checker compatibility
            return type(
                f"{cls.__name__}WithValidator",
                (cls,),
                {
                    "custom_validator": field_validator("value")(
                        classmethod(custom_validator)
                    )
                },
            )

    class DynamicModel(BaseModel):
        """Model demonstrating dynamic reconstruction capabilities."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        name: str
        fields: t.Dict = Field(default_factory=t.Dict)

        @field_validator("fields", mode="before")
        @classmethod
        def validate_fields(
            cls,
            value: Mapping[str, t.GuardInputValue] | t.Dict,
        ) -> t.Dict:
            return t.Dict.model_validate(value)

        @classmethod
        def create_dynamic(cls, name: str, **fields: t.GuardInputValue) -> Self:
            """Create a dynamic model instance using model_construct."""
            return cls(name=name, fields=t.Dict(root=fields))

        def add_field(self, key: str, value: t.GuardInputValue) -> None:
            """Add a field dynamically."""
            updated_fields = dict(self.fields.root)
            updated_fields[key] = value
            self.fields = t.Dict(root=updated_fields)

        def rebuild_with_validation(self) -> Self:
            """Rebuild model with full validation using model_validate."""
            return self.model_copy(deep=True)

        @property
        def dynamic_field_count(self) -> int:
            """Count of dynamic fields."""
            return len(self.fields.root)

        @property
        def has_dynamic_fields(self) -> bool:
            """Check if model has dynamic fields."""
            return len(self.fields.root) > 0

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__ = ["FlextModelFoundation"]
