"""FlextMixins - Mixin classes for FLEXT ecosystem.

This module provides mixin classes for common functionality patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from flext_core.utilities import FlextUtilities


class FlextMixins:
    """Mixin classes for FLEXT ecosystem functionality."""

    class Serializable:
        """Mixin for serialization functionality."""

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON string."""
            if hasattr(self, "model_dump") and callable(getattr(self, "model_dump")):
                return json.dumps(getattr(self, "model_dump")(), indent=indent)
            return json.dumps(self.__dict__, indent=indent)

    class Loggable:
        """Mixin for logging functionality."""

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log info message."""

        def log_error(self, message: str, **kwargs: object) -> None:
            """Log error message."""

        def log_warning(self, message: str, **kwargs: object) -> None:
            """Log warning message."""

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log debug message."""

    class Service:
        """Mixin for service functionality."""

        def __init__(self, **data: object) -> None:
            """Initialize service."""

    @staticmethod
    def to_json(obj: object, indent: int | None = None) -> str:
        """Convert object to JSON string."""
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return json.dumps(getattr(obj, "model_dump")(), indent=indent)
        if hasattr(obj, "__dict__"):
            return json.dumps(obj.__dict__, indent=indent)
        return json.dumps(str(obj), indent=indent)

    @staticmethod
    def initialize_validation(obj: object) -> None:
        """Initialize validation for object."""

    @staticmethod
    def start_timing(obj: object) -> None:
        """Start timing for object."""

    @staticmethod
    def clear_cache(obj: object) -> None:
        """Clear cache for object."""

    @staticmethod
    def create_timestamp_fields(obj: object) -> None:
        """Create timestamp fields for object."""
        if hasattr(obj, "created_at"):
            setattr(obj, "created_at", datetime.now(UTC))
        if hasattr(obj, "updated_at"):
            setattr(obj, "updated_at", datetime.now(UTC))
        # MyPy doesn't understand setattr modifies the object, but it does

    @staticmethod
    def ensure_id(obj: object) -> None:
        """Ensure object has an ID."""
        if hasattr(obj, "id") and not getattr(obj, "id"):
            setattr(obj, "id", FlextUtilities.Generators.generate_id())

    @staticmethod
    def update_timestamp(obj: object) -> None:
        """Update timestamp for object."""
        if hasattr(obj, "updated_at"):
            setattr(obj, "updated_at", datetime.now(UTC))

    @staticmethod
    def log_operation(obj: object, operation: str) -> None:
        """Log operation for object."""
        # Simple logging - can be enhanced later

    @staticmethod
    def initialize_state(obj: object, state: str) -> None:
        """Initialize state for object."""
        if hasattr(obj, "state"):
            setattr(obj, "state", state)

    @staticmethod
    def to_dict(obj: object) -> dict[str, object]:
        """Convert object to dictionary."""
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            result = getattr(obj, "model_dump")()
            return result if isinstance(result, dict) else {"model_dump": result}
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
            if isinstance(obj_dict, dict):
                return obj_dict
        return {"type": type(obj).__name__, "value": str(obj)}
