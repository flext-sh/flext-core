"""FLEXT Core Mixins - Reusable Behavior Patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Enterprise mixins that provide common behaviors for classes, dramatically
reducing boilerplate code for timestamping, auditing, serialization,
validation, and other cross-cutting concerns.
"""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Mapping

# =============================================================================
# TIMESTAMP MIXINS - Automatic Time Tracking
# =============================================================================


class FlextTimestampMixin(BaseModel):
    """Mixin to add automatic timestamp tracking to models.

    Automatically handles created_at and updated_at timestamps with UTC timezone.

    Example:
        class User(FlextTimestampMixin):
            name: str
            email: str

        user = User(name="Alice", email="alice@example.com")
        assert user.created_at is not None
        assert user.updated_at is not None
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when entity was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when entity was last updated",
    )

    def touch_updated_at(self) -> None:
        """Update the updated_at timestamp to current time."""
        object.__setattr__(self, "updated_at", datetime.now(UTC))

    def get_age_seconds(self) -> float:
        """Get age of entity in seconds."""
        now = datetime.now(UTC)
        return (now - self.created_at).total_seconds()

    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if entity is older than specified age."""
        return self.get_age_seconds() > max_age_seconds


class FlextAuditMixin(FlextTimestampMixin):
    """Mixin to add audit trail functionality to models.

    Extends timestamp mixin with user tracking and version control.

    Example:
        class Document(FlextAuditMixin):
            title: str
            content: str

        doc = Document(
            title="My Doc",
            content="Content",
            created_by="user123"
        )
        doc.update_audit(updated_by="user456")
    """

    created_by: str | None = Field(
        default=None,
        description="User who created this entity",
    )
    updated_by: str | None = Field(
        default=None,
        description="User who last updated this entity",
    )
    version: int = Field(
        default=1,
        description="Version number for optimistic locking",
    )

    def update_audit(
        self,
        updated_by: str,
        *,
        increment_version: bool = True,
    ) -> None:
        """Update audit information."""
        object.__setattr__(self, "updated_by", updated_by)
        self.touch_updated_at()
        if increment_version:
            object.__setattr__(self, "version", self.version + 1)

    def get_audit_info(self) -> dict[str, Any]:
        """Get complete audit information."""
        return {
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "version": self.version,
            "age_seconds": self.get_age_seconds(),
        }


# =============================================================================
# SERIALIZATION MIXINS - Data Conversion
# =============================================================================


class FlextSerializationMixin(BaseModel):
    """Mixin to add enhanced serialization capabilities.

    Provides JSON, dict, and custom format serialization with type safety.

    Example:
        class Config(FlextSerializationMixin):
            database_url: str
            debug: bool = False

        config = Config(database_url="sqlite:///app.db")
        json_str = config.to_json_safe().unwrap()
        dict_data = config.to_dict_safe().unwrap()
    """

    def to_json_safe(self, **kwargs: Any) -> FlextResult[str]:
        """Safely serialize to JSON string."""
        try:
            return FlextResult.ok(self.model_dump_json(**kwargs))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"JSON serialization failed: {e}")

    def to_dict_safe(self, **kwargs: Any) -> FlextResult[dict[str, Any]]:
        """Safely serialize to dictionary."""
        try:
            return FlextResult.ok(self.model_dump(**kwargs))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Dict serialization failed: {e}")

    @classmethod
    def from_json_safe(cls, json_str: str) -> FlextResult[FlextSerializationMixin]:
        """Safely deserialize from JSON string."""
        try:
            return FlextResult.ok(cls.model_validate_json(json_str))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"JSON deserialization failed: {e}")

    @classmethod
    def from_dict_safe(cls, data: dict[str, Any]) -> FlextResult[FlextSerializationMixin]:
        """Safely deserialize from dictionary."""
        try:
            return FlextResult.ok(cls.model_validate(data))
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Dict deserialization failed: {e}")

    def to_pretty_json(self, indent: int = 2) -> FlextResult[str]:
        """Create pretty-formatted JSON."""
        result = self.to_dict_safe()
        if not result.is_success:
            return FlextResult.fail(f"Failed to convert to dict: {result.error}")

        try:
            pretty_json = json.dumps(result.data, indent=indent, ensure_ascii=False)
            return FlextResult.ok(pretty_json)
        except Exception as e:  # noqa: BLE001
            return FlextResult.fail(f"Pretty JSON formatting failed: {e}")


# =============================================================================
# VALIDATION MIXINS - Business Rule Validation
# =============================================================================


class FlextValidationMixin(ABC):
    """Mixin to add business validation capabilities.

    Provides framework for custom validation rules with error accumulation.

    Example:
        class User(BaseModel, FlextValidationMixin):
            name: str
            age: int
            email: str

            def validate_business_rules(self) -> FlextResult[None]:
                result = self.start_validation()

                if self.age < 18:
                    result.add_error("User must be 18 or older")

                if "@" not in self.email:
                    result.add_error("Invalid email format")

                return result.finalize()
    """

    def start_validation(self) -> FlextValidationResult:
        """Start a new validation session."""
        return FlextValidationResult()

    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]:
        """Override this method to implement business validation."""

    def is_valid(self) -> bool:
        """Check if entity passes all business validations."""
        return self.validate_business_rules().is_success


class FlextValidationResult:
    """Helper class for accumulating validation errors."""

    def __init__(self) -> None:
        """Initialize validation result."""
        self.errors: list[str] = []

    def add_error(self, error: str) -> FlextValidationResult:
        """Add a validation error."""
        self.errors.append(error)
        return self

    def add_errors(self, errors: list[str]) -> FlextValidationResult:
        """Add multiple validation errors."""
        self.errors.extend(errors)
        return self

    def finalize(self) -> FlextResult[None]:
        """Finalize validation and return result."""
        if self.errors:
            error_msg = "; ".join(self.errors)
            return FlextResult.fail(f"Validation failed: {error_msg}")
        return FlextResult.ok(None)

    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.errors) > 0


# =============================================================================
# COMPARISON MIXINS - Equality and Ordering
# =============================================================================


class FlextComparableMixin(BaseModel):
    """Mixin to add comparison capabilities based on ID.

    Provides equality and hash methods based on entity ID.

    Example:
        class Product(FlextComparableMixin):
            id: str
            name: str
            price: float

            def get_comparison_key(self) -> str:
                return self.id

        p1 = Product(id="123", name="Item", price=10.0)
        p2 = Product(id="123", name="Different Name", price=20.0)
        assert p1 == p2  # Same ID = equal
    """

    def get_comparison_key(self) -> str | dict[str, Any]:
        """Override to define what makes entities equal.

        Returns:
            Value used for equality comparison
        """
        # Try common ID fields
        for field_name in ["id", "entity_id", "pk", "key"]:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if isinstance(value, str):
                    return value
                # Convert other types to string
                return str(value)

        # Fallback to all fields
        return self.model_dump()

    def __eq__(self, other: object) -> bool:
        """Equality based on comparison key."""
        if not isinstance(other, FlextComparableMixin):
            return False
        return self.get_comparison_key() == other.get_comparison_key()

    def __hash__(self) -> int:
        """Hash based on comparison key."""
        key = self.get_comparison_key()
        if isinstance(key, dict):
            # Convert dict to hashable tuple
            return hash(tuple(sorted(key.items())))
        return hash(key)


# =============================================================================
# CACHING MIXINS - Instance Caching
# =============================================================================


class FlextCacheableMixin(BaseModel):
    """Mixin to add caching capabilities to entities.

    Provides instance-level caching for expensive computed properties.

    Example:
        class Report(FlextCacheableMixin):
            data: list[dict]

            @property
            def expensive_calculation(self) -> float:
                cache_key = "expensive_calc"
                cached = self.get_cached(cache_key)
                if cached is not None:
                    return cached

                # Expensive calculation
                result = sum(item["value"] for item in self.data)
                self.set_cached(cache_key, result)
                return result
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    cache: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Internal cache storage",
    )

    def get_cached(self, key: str) -> object | None:
        """Get cached value by key."""
        return self.cache.get(key)

    def set_cached(self, key: str, value: object) -> None:
        """Set cached value."""
        self.cache[key] = value

    def clear_cache(self, key: str | None = None) -> None:
        """Clear cache entry or entire cache."""
        if key is None:
            self.cache.clear()
        else:
            self.cache.pop(key, None)

    def cache_exists(self, key: str) -> bool:
        """Check if cache key exists."""
        return key in self.cache

    @property
    def cache_size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)


# =============================================================================
# METADATA MIXINS - Additional Information
# =============================================================================


class FlextMetadataMixin(BaseModel):
    """Mixin to add flexible metadata storage.

    Provides type-safe metadata storage for additional entity information.

    Example:
        class File(FlextMetadataMixin):
            filename: str
            size: int

        file = File(filename="doc.pdf", size=1024)
        file.set_metadata("author", "Alice")
        file.set_metadata("tags", ["document", "important"])

        author = file.get_metadata("author")  # "Alice"
        tags = file.get_metadata("tags", [])  # ["document", "important"]
    """

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata storage",
    )

    def set_metadata(self, key: str, value: object) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: object = None) -> object:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self.metadata

    def remove_metadata(self, key: str) -> object | None:
        """Remove and return metadata value."""
        value: object | None = self.metadata.pop(key, None)
        return value

    def clear_metadata(self) -> None:
        """Clear all metadata."""
        self.metadata.clear()

    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        """Update metadata with multiple values."""
        self.metadata.update(updates)

    def get_all_metadata(self) -> dict[str, Any]:
        """Get copy of all metadata."""
        return dict(self.metadata)


# =============================================================================
# COMBINED MIXINS - Common Combinations
# =============================================================================


class FlextEntityMixin(
    FlextAuditMixin,
    FlextSerializationMixin,
    FlextValidationMixin,
    FlextComparableMixin,
    FlextMetadataMixin,
):
    """Combined mixin with common entity behaviors.

    Combines audit trail, serialization, validation, comparison, and metadata
    capabilities into a single mixin for comprehensive entity support.

    Example:
        class User(FlextEntityMixin):
            id: FlextEntityId
            name: str
            email: str

            def get_comparison_key(self) -> str:
                return self.id

            def validate_business_rules(self) -> FlextResult[None]:
                result = self.start_validation()
                if "@" not in self.email:
                    result.add_error("Invalid email")
                return result.finalize()
    """

    # Override to provide sensible defaults
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=False,  # Allow updates for audit fields
    )


class FlextValueObjectMixin(
    FlextSerializationMixin,
    FlextValidationMixin,
    FlextComparableMixin,
):
    """Combined mixin for value objects.

    Value objects are immutable and compared by value, not identity.
    Excludes audit trail as value objects shouldn't change.

    Example:
        class Money(FlextValueObjectMixin):
            amount: float
            currency: str

            def validate_business_rules(self) -> FlextResult[None]:
                result = self.start_validation()
                if self.amount < 0:
                    result.add_error("Amount cannot be negative")
                return result.finalize()
    """

    model_config = ConfigDict(
        frozen=True,  # Value objects are immutable
        validate_assignment=True,
    )


# =============================================================================
# EXPORTS - Clean Public API
# =============================================================================

__all__ = [
    "FlextAuditMixin",
    "FlextCacheableMixin",
    "FlextComparableMixin",
    "FlextEntityMixin",
    "FlextMetadataMixin",
    "FlextSerializationMixin",
    "FlextTimestampMixin",
    "FlextValidationMixin",
    "FlextValidationResult",
    "FlextValueObjectMixin",
]
