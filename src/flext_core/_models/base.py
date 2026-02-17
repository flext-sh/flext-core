"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    PlainValidator,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from flext_core.constants import c
from flext_core.typings import t


# Advanced Pydantic v2 Validators
def strip_whitespace(v: str) -> str:
    """Strip whitespace from string values."""
    return v.strip()


def ensure_utc_datetime(v: datetime | None) -> datetime | None:
    """Ensure datetime is UTC timezone."""
    if v is not None and v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v


def normalize_to_list(v: t.GeneralValueType) -> list[t.GeneralValueType]:
    """Normalize value to list format."""
    if isinstance(v, list):
        return v
    if isinstance(v, (tuple, set)):
        return list(v)
    return [v]


def validate_positive_number(v: float) -> int | float:
    """Validate that number is positive."""
    if v <= 0:
        msg = "Value must be positive"
        raise ValueError(msg)
    return v


def validate_non_empty_string(v: str) -> str:
    """Validate that string is not empty after stripping."""
    stripped = v.strip()
    if not stripped:
        msg = "String cannot be empty or whitespace"
        raise ValueError(msg)
    return stripped


# Custom field types using Pydantic v2 validators
StrippedString = Annotated[str, AfterValidator(strip_whitespace)]
ValidatedString = Annotated[str, AfterValidator(validate_non_empty_string)]
UTCDatetime = Annotated[datetime, AfterValidator(ensure_utc_datetime)]
PositiveInt = Annotated[int, AfterValidator(validate_positive_number)]
PositiveFloat = Annotated[float, AfterValidator(validate_positive_number)]
NormalizedList = Annotated[list[t.GeneralValueType], BeforeValidator(normalize_to_list)]


# Advanced custom types with PlainValidator
def validate_email(v: str) -> str:
    """Validate email format using simple regex."""
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
        msg = "Invalid email format"
        raise ValueError(msg)
    return v


def validate_url(v: str) -> str:
    """Validate URL format."""
    parsed = urlparse(v)
    if not parsed.scheme or not parsed.netloc:
        msg = "Invalid URL format"
        raise ValueError(msg)
    return v


def validate_semver(v: str) -> str:
    """Validate semantic version format."""
    if not re.match(r"^\d+\.\d+\.\d+(-[\w\.\-]+)?(\+[\w\.\-]+)?$", v):
        msg = "Invalid semantic version format"
        raise ValueError(msg)
    return v


def validate_uuid_string(v: str) -> str:
    """Validate UUID string format."""
    try:
        _ = uuid.UUID(v)
        return v
    except (ValueError, TypeError):
        msg = "Invalid UUID format"
        raise ValueError(msg) from None


# Custom field types with PlainValidator
EmailStr = Annotated[str, PlainValidator(validate_email)]
UrlStr = Annotated[str, PlainValidator(validate_url)]
SemVerStr = Annotated[str, PlainValidator(validate_semver)]
UUIDStr = Annotated[str, PlainValidator(validate_uuid_string)]


# Complex validators for nested structures
def validate_config_dict(v: object) -> dict[str, t.GeneralValueType]:
    """Validate configuration dictionary structure."""
    if not isinstance(v, dict):
        msg = "Configuration must be a dictionary"
        raise TypeError(msg)

    # Check for reserved keys
    for key in v:
        if key.startswith("_"):
            msg = f"Keys starting with '_' are reserved: {key}"
            raise ValueError(msg)

    return v


def validate_tags_list(v: object) -> list[str]:
    """Validate and normalize tags list."""
    if not isinstance(v, list):
        msg = "Tags must be a list"
        raise TypeError(msg)

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in v:
        if not isinstance(tag, str):
            msg = f"Tag must be string, got {type(tag)}"
            raise TypeError(msg)
        clean_tag = tag.strip().lower()
        if clean_tag and clean_tag not in seen:
            normalized.append(clean_tag)
            seen.add(clean_tag)

    return normalized


# Advanced custom types
ValidatedConfigDict = Annotated[
    dict[str, t.GeneralValueType], PlainValidator(validate_config_dict)
]
NormalizedTags = Annotated[list[str], PlainValidator(validate_tags_list)]


# Renamed to FlextModelFoundation for better clarity
class FlextModelFoundation:
    """Container for base model classes - Tier 0, 100% standalone."""

    class ArbitraryTypesModel(BaseModel):
        """Base model with arbitrary types support."""

        model_config = ConfigDict(
            validate_assignment=True,
            extra=c.ModelConfig.EXTRA_FORBID,
            arbitrary_types_allowed=True,
            use_enum_values=True,
        )

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
        attributes: dict[str, t.GeneralValueType] = Field(default_factory=dict)

        @field_validator("attributes", mode="before")
        @classmethod
        def _validate_attributes(
            cls,
            value: t.GeneralValueType | t.Dict | None,
        ) -> dict[str, t.GeneralValueType]:
            if value is None:
                return {}
            if isinstance(value, BaseModel):
                return dict(value.model_dump())
            if isinstance(value, t.Dict):
                return dict(value.root)
            if isinstance(value, Mapping):
                return {str(k): v for k, v in value.items()}
            msg = (
                f"attributes must be dict-like or BaseModel, got {type(value).__name__}"
            )
            raise TypeError(msg)

    # Command message type
    class CommandMessage(BaseModel):
        """Command message with discriminated union support."""

        message_type: Literal["command"] = "command"
        command_type: str
        issuer_id: str | None = None
        data: t.Dict = Field(default_factory=lambda: t.Dict(root={}))

    # Query message type
    class QueryMessage(BaseModel):
        """Query message with discriminated union support."""

        message_type: Literal["query"] = "query"
        query_type: str
        filters: t.Dict = Field(default_factory=lambda: t.Dict(root={}))
        pagination: t.Dict | None = None

    # Event message type
    class EventMessage(BaseModel):
        """Event message with discriminated union support."""

        message_type: Literal["event"] = "event"
        event_type: str
        aggregate_id: str
        data: t.Dict = Field(default_factory=lambda: t.Dict(root={}))
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
        value: t.GeneralValueType
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
        value: t.GeneralValueType
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
        validated_data: t.GeneralValueType
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
        validated_data: t.GeneralValueType
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
            if not isinstance(other, self.__class__):
                return NotImplemented
            return self.model_dump() == other.model_dump()

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
        )

        unique_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="Unique identifier for the model instance",
            min_length=1,
            frozen=False,  # Allow regeneration
        )

        @field_validator("unique_id")
        @classmethod
        def validate_unique_id(cls, v: str) -> str:
            """Validate that unique_id is not empty and strip whitespace."""
            if not v or not v.strip():
                msg = "unique_id cannot be empty or whitespace"
                raise ValueError(msg)
            return v.strip()

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
                uuid.UUID(self.unique_id)
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

        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the model was created (UTC timezone)",
            frozen=True,  # Creation time should never change
        )
        updated_at: datetime | None = Field(
            default=None,
            description="Timestamp when the model was last updated (UTC timezone)",
        )

        @field_validator("created_at", "updated_at")
        @classmethod
        def validate_timestamps(cls, v: datetime | None) -> datetime | None:
            """Ensure timestamps are UTC."""
            if v is not None and v.tzinfo is None:
                # Assume naive datetime is UTC
                return v.replace(tzinfo=UTC)
            return v

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

        @computed_field
        @property
        def age_seconds(self) -> float:
            """Calculate age of the model in seconds since creation.

            Returns:
                float: Age in seconds.

            """
            now = datetime.now(UTC)
            return (now - self.created_at).total_seconds()

        @computed_field
        @property
        def age_minutes(self) -> float:
            """Calculate age of the model in minutes since creation.

            Returns:
                float: Age in minutes.

            """
            return self.age_seconds / 60.0

        @computed_field
        @property
        def age_hours(self) -> float:
            """Calculate age of the model in hours since creation.

            Returns:
                float: Age in hours.

            """
            return self.age_minutes / 60.0

        @computed_field
        @property
        def age_days(self) -> float:
            """Calculate age of the model in days since creation.

            Returns:
                float: Age in days.

            """
            return self.age_hours / 24.0

        @computed_field
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
            age = self.age_seconds
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

        @computed_field
        @property
        def is_recent(self) -> bool:
            """Check if the model was created within the last hour.

            Returns:
                bool: True if created within last hour.

            """
            return self.age_minutes <= c.Performance.RECENT_THRESHOLD_MINUTES

        @computed_field
        @property
        def is_very_recent(self) -> bool:
            """Check if the model was created within the last 5 minutes.

            Returns:
                bool: True if created within last 5 minutes.

            """
            return self.age_minutes <= c.Performance.VERY_RECENT_THRESHOLD_MINUTES

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

        @field_validator("version")
        @classmethod
        def validate_version(cls, v: int) -> int:
            """Validate version is non-negative."""
            if v < 0:
                msg = "Version cannot be negative"
                raise ValueError(msg)
            return v

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
        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when record was created (UTC timezone)",
            frozen=True,
        )
        updated_at: datetime | None = Field(
            default=None,
            description="Timestamp when record was last updated (UTC timezone)",
        )

        @field_validator("created_by", "updated_by")
        @classmethod
        def validate_audit_users(cls, v: str | None) -> str | None:
            """Validate audit user fields."""
            if v is not None and not v.strip():
                msg = "Audit user cannot be empty or whitespace"
                raise ValueError(msg)
            return v.strip() if v else None

        @field_validator("created_at", "updated_at")
        @classmethod
        def validate_audit_timestamps(cls, v: datetime | None) -> datetime | None:
            """Ensure audit timestamps are UTC."""
            if v is not None and v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v

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
        )

        deleted_at: datetime | None = Field(
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

        @field_validator("deleted_by")
        @classmethod
        def validate_deleted_by(cls, v: str | None) -> str | None:
            """Validate deleted_by field."""
            if v is not None and not v.strip():
                msg = "deleted_by cannot be empty or whitespace"
                raise ValueError(msg)
            return v.strip() if v else None

        @field_validator("deleted_at")
        @classmethod
        def validate_deleted_at(cls, v: datetime | None) -> datetime | None:
            """Ensure deleted_at is UTC."""
            if v is not None and v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v

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
            current_data = self.model_dump()
            current_data.update({
                "is_deleted": True,
                "deleted_at": now,
                "deleted_by": deleted_by,
            })

            # Replace current instance data using __dict__.update to bypass
            # intermediate validation states during assignment
            validated = self.__class__.model_validate(current_data)
            for key, value in validated.__dict__.items():
                object.__setattr__(self, key, value)

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
        labels: dict[str, str] = Field(
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
        def validate_labels(cls, v: dict[str, str]) -> dict[str, str]:
            """Validate labels dictionary."""
            cleaned = {}
            for key, value in v.items():
                cleaned_key = key.strip()
                cleaned_value = value.strip()
                if cleaned_key and cleaned_value:
                    cleaned[cleaned_key] = cleaned_value
            return cleaned

        @computed_field
        @property
        def tag_count(self) -> int:
            """Get number of tags.

            Returns:
                int: Number of tags.

            """
            return len(self.tags)

        @computed_field
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
            return self.tag_count > 0

        @computed_field
        @property
        def has_categories(self) -> bool:
            """Check if record has any categories.

            Returns:
                bool: True if has at least one category.

            """
            return int(self.category_count) > 0

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
            self.labels.pop(key, None)

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

        def validate_self(self) -> Self:
            """Re-validate the model instance."""
            return self.__class__.model_validate(self.model_dump())

        def is_valid(self) -> bool:
            """Check if the model is currently valid."""
            try:
                self.validate_self()
                return True
            except Exception:
                return False

        def get_validation_errors(self) -> list[str]:
            """Get list of validation errors."""
            try:
                self.validate_self()
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
        ) -> dict[str, t.GeneralValueType]:
            """Convert to dictionary with advanced options."""
            return self.model_dump(
                exclude_none=exclude_none,
                exclude_unset=exclude_unset,
            )

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON string."""
            return self.model_dump_json(indent=indent)

        @classmethod
        def from_dict(cls, data: dict[str, t.GeneralValueType]) -> Self:
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
        metadata: t.Dict = Field(default_factory=lambda: t.Dict(root={}))

        @field_serializer("timestamp")
        def serialize_timestamp_iso(self, value: datetime) -> str:
            """Serialize timestamp to ISO format."""
            return value.isoformat()

        @field_serializer("metadata", when_used="json")
        def serialize_metadata_clean(self, value: t.Dict) -> dict[str, str]:
            """Serialize metadata with string conversion."""
            return {k: str(v) for k, v in value.root.items()}

        def to_json_multiple_formats(self) -> dict[str, str]:
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

        def add_runtime_field(self, name: str, value: t.GeneralValueType) -> None:
            """Add a field at runtime (stored in __dict__)."""
            object.__setattr__(self, name, value)

        def get_runtime_field(
            self, name: str, default: t.GeneralValueType = None
        ) -> t.GeneralValueType:
            """Get a runtime field value."""
            return getattr(self, name, default)

        @classmethod
        def rebuild_with_validator(
            cls, validator_func: Callable[[t.GeneralValueType], t.GeneralValueType]
        ) -> type[FlextModelFoundation.DynamicRebuildModel]:
            """Rebuild model with additional validator."""

            # Create validator function (without decorators)
            def custom_validator(
                _cls: type,
                v: t.GeneralValueType,
            ) -> t.GeneralValueType:
                """Apply custom validator to value field."""
                return validator_func(v)

            # Create new class dynamically using type() for type-checker compatibility
            new_model = type(
                f"{cls.__name__}WithValidator",
                (cls,),
                {
                    "custom_validator": field_validator("value")(
                        classmethod(custom_validator)
                    )
                },
            )

            return new_model

    class DynamicModel(BaseModel):
        """Model demonstrating dynamic reconstruction capabilities."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
        )

        name: str
        fields: t.Dict = Field(default_factory=lambda: t.Dict(root={}))

        @classmethod
        def create_dynamic(cls, name: str, **fields: t.GeneralValueType) -> Self:
            """Create a dynamic model instance using model_construct."""
            return cls.model_construct(name=name, fields=t.Dict(root=fields))

        def add_field(self, key: str, value: t.GeneralValueType) -> None:
            """Add a field dynamically."""
            self.fields.root[key] = value

        def rebuild_with_validation(self) -> Self:
            """Rebuild model with full validation using model_validate."""
            return self.__class__.model_validate(self.model_dump())

        @computed_field
        @property
        def dynamic_field_count(self) -> int:
            """Count of dynamic fields."""
            return len(self.fields)

        @computed_field
        @property
        def has_dynamic_fields(self) -> bool:
            """Check if model has dynamic fields."""
            return self.dynamic_field_count > 0

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


# Alias for backward compatibility during transition
# NOTE: Remove in v1.0.0, use FlextModelFoundation directly
# Ref: https://github.com/flext-team/flext-core/issues/rename-base-class
FlextModelsBase = FlextModelFoundation

__all__ = ["FlextModelFoundation", "FlextModelsBase"]
