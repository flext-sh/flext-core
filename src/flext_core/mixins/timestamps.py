"""FLEXT Timestamps Mixin - Timestamp tracking using centralized components.

This module provides timestamp mixins that leverage centralized FLEXT
ecosystem components for consistent temporal tracking patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols


class FlextTimestamps:
    """Unified timestamp tracking system using centralized FLEXT components."""

    @staticmethod
    def create_timestamp_fields(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Initialize timestamp fields on an object using FlextModels.Timestamp."""
        current_time = FlextModels.Timestamp(root=datetime.now(UTC))
        # Initialize internal fields to avoid property recursion
        obj._created_at = current_time.root
        obj._updated_at = current_time.root
        obj._timestamp_initialized = True

    @staticmethod
    def update_timestamp(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Update the timestamp on an object."""
        if not getattr(obj, "_timestamp_initialized", False):
            d = getattr(obj, "__dict__", {})
            # Initialize internals only if no public timestamps present
            if not (isinstance(d, dict) and ("updated_at" in d or "created_at" in d)):
                FlextTimestamps.create_timestamp_fields(obj)

        updated_time = FlextModels.Timestamp(root=datetime.now(UTC))
        if getattr(obj, "_timestamp_initialized", False):
            obj._updated_at = updated_time.root
        else:
            # For plain objects with public attribute; update both representations
            try:
                old = getattr(obj, "__dict__", {}).get("updated_at")
                new_val = updated_time.root
                if isinstance(old, datetime) and new_val == old:
                    new_val = old + timedelta(microseconds=1)
                obj.__dict__["updated_at"] = new_val
                # Maintain internal mirror for consistency
                obj._updated_at = new_val
            except Exception:
                obj._updated_at = updated_time.root

    @staticmethod
    def get_created_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime:
        """Get creation timestamp."""
        # Do not initialize internals for plain objects
        if getattr(obj, "_timestamp_initialized", False):
            return getattr(obj, "_created_at", datetime.now(UTC))
        # Plain objects with public attribute
        d = getattr(obj, "__dict__", {})
        if isinstance(d, dict) and "created_at" in d:
            return cast("datetime", d["created_at"])
        return datetime.now(UTC)

    @staticmethod
    def get_updated_at(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> datetime:
        """Get last update timestamp."""
        # Do not initialize internals for plain objects
        if getattr(obj, "_timestamp_initialized", False):
            return getattr(obj, "_updated_at", datetime.now(UTC))
        d = getattr(obj, "__dict__", {})
        if isinstance(d, dict) and "updated_at" in d:
            return cast("datetime", d["updated_at"])
        return datetime.now(UTC)

    @staticmethod
    def get_age_seconds(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> float:
        """Get age in seconds since creation."""
        created_at = FlextTimestamps.get_created_at(obj)
        age = datetime.now(UTC) - created_at
        return age.total_seconds()

    class Timestampable:
        """Mixin class providing automatic timestamp tracking."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize timestamp fields."""
            super().__init__(*args, **kwargs)
            FlextTimestamps.create_timestamp_fields(self)

        def update_timestamp(self) -> None:
            """Update the timestamp on this object."""
            FlextTimestamps.update_timestamp(self)

        @property
        def created_at(self) -> datetime:
            """Get creation timestamp."""
            return FlextTimestamps.get_created_at(self)

        @property
        def updated_at(self) -> datetime:
            """Get last update timestamp."""
            return FlextTimestamps.get_updated_at(self)

        @property
        def age_seconds(self) -> float:
            """Get age in seconds since creation."""
            return FlextTimestamps.get_age_seconds(self)


__all__ = [
    "FlextTimestamps",
]
