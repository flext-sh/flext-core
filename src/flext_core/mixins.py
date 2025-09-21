"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import cast

from flext_core.loggings import FlextLogger


class FlextMixins:
    """Namespace containing the reusable mixins shared across FLEXT packages.

    They provide the logging, serialization, and state helpers needed to keep
    domain services aligned with the modernization plan without duplicating
    boilerplate.
    """

    class Serializable:
        """Serialization helpers reused in modernization-ready models."""

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON string."""
            if hasattr(self, "model_dump"):
                model_dump_method = getattr(self, "model_dump", None)
                if model_dump_method is not None and callable(model_dump_method):
                    return json.dumps(model_dump_method(), indent=indent)
            return json.dumps(self.__dict__, indent=indent)

    class Loggable:
        """Logging helper mixin aligned with context-first observability."""

        @property
        def logger(self) -> FlextLogger:
            """Get logger instance for this class."""
            return FlextLogger(self.__class__.__name__)

        def log_info(self, message: str, **kwargs: object) -> None:
            """Log info message."""
            self.logger.info(message, **kwargs)

        def log_error(self, message: str, **kwargs: object) -> None:
            """Log error message."""
            # Extract 'error' parameter if present for FlextLogger.error() method
            error = kwargs.pop("error", None)
            error_typed = cast("Exception | str | None", error)
            self.logger.error(message, error=error_typed, **kwargs)

        def log_warning(self, message: str, **kwargs: object) -> None:
            """Log warning message."""
            self.logger.warning(message, **kwargs)

        def log_debug(self, message: str, **kwargs: object) -> None:
            """Log debug message."""
            self.logger.debug(message, **kwargs)

    class Service:
        """Service bootstrap mixin shared across domain services."""

        def __init__(self, **data: object) -> None:
            """Initialize service with provided data and basic state."""
            # Store provided initialization data as attributes
            for key, value in data.items():
                with contextlib.suppress(Exception):
                    setattr(self, str(key), value)
            # Mark service as initialized for observability in tests
            with contextlib.suppress(Exception):
                setattr(self, "initialized", True)

    @staticmethod
    def to_json(obj: object, indent: int | None = None) -> str:
        """Convert object to JSON string."""
        if hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                return json.dumps(model_dump_method(), indent=indent)
        if hasattr(obj, "__dict__"):
            return json.dumps(obj.__dict__, indent=indent)
        return json.dumps(str(obj), indent=indent)

    @staticmethod
    def initialize_validation(obj: object) -> None:
        """Initialize validation for object by toggling a 'validated' flag."""
        with contextlib.suppress(Exception):
            setattr(obj, "validated", True)

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
        if hasattr(obj, "id"):
            id_value = getattr(obj, "id", None)
            if not id_value:
                setattr(obj, "id", str(uuid.uuid4()))

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
        if hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                result = model_dump_method()
                return result if isinstance(result, dict) else {"model_dump": result}
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return {"type": type(obj).__name__, "value": str(obj)}
