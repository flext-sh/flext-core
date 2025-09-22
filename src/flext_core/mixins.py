"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.utilities import FlextUtilities


class FlextMixins:
    """Single unified mixin class providing core behaviors for FLEXT ecosystem.

    Follows FLEXT quality standards:
    - Single class per module architecture
    - Type-safe Pydantic-only method signatures
    - No backward compatibility wrappers or aliases
    - Direct implementation without delegation layers

    All methods use Pydantic models for parameter validation and type safety.
    """

    # =============================================================================
    # SERIALIZATION METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def to_json(request: FlextModels.SerializationRequest) -> str:
        """Convert object to JSON string using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            JSON string representation of the object

        """
        obj = request.data

        # Try model_dump method first if requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                data = model_dump_method()
                return json.dumps(
                    data,
                    indent=request.indent,
                    sort_keys=request.sort_keys,
                    ensure_ascii=request.ensure_ascii,
                )

        # Try __dict__ method
        if hasattr(obj, "__dict__"):
            data = obj.__dict__
            return json.dumps(
                data,
                indent=request.indent,
                sort_keys=request.sort_keys,
                ensure_ascii=request.ensure_ascii,
            )

        # Fallback to string representation
        return json.dumps(
            str(obj),
            indent=request.indent,
            sort_keys=request.sort_keys,
            ensure_ascii=request.ensure_ascii,
        )

    @staticmethod
    def to_dict(request: FlextModels.SerializationRequest) -> dict[str, object]:
        """Convert object to dictionary using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            Dictionary representation of the object

        """
        obj = request.data

        # Try model_dump method first if requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                result = model_dump_method()
                if isinstance(result, dict):
                    # Type-safe conversion with explicit casting
                    return cast("dict[str, object]", result)
                return {"model_dump": result}

        # Try __dict__ method with proper type casting
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
            # obj.__dict__ is always a dict[str, object], so we can safely cast it
            return cast("dict[str, object]", obj_dict)

        # Fallback to type and value representation
        return {"type": type(obj).__name__, "value": str(obj)}

    # =============================================================================
    # VALIDATION METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def initialize_validation(obj: object, field_name: str) -> None:
        """Initialize validation for object.

        Args:
            obj: Object to set validation on
            field_name: Name of the field to set validation flag

        """
        # Set the validation flag
        with contextlib.suppress(Exception):
            setattr(obj, field_name, True)

    # =============================================================================
    # TIMESTAMP METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def create_timestamp_fields(config: FlextModels.TimestampConfig) -> None:
        """Create timestamp fields for object using TimestampConfig model.

        Args:
            config: TimestampConfig containing object and timestamp settings

        """
        obj = config.obj
        timezone = UTC if config.use_utc else None
        current_time = datetime.now(timezone)

        # Create created_at field if it exists and is not set
        created_field = config.field_names.get("created_at", "created_at")
        if hasattr(obj, created_field) and getattr(obj, created_field, None) is None:
            setattr(obj, created_field, current_time)

        # Create updated_at field if it exists and auto_update is enabled
        updated_field = config.field_names.get("updated_at", "updated_at")
        if hasattr(obj, updated_field) and config.auto_update:
            setattr(obj, updated_field, current_time)

    @staticmethod
    def update_timestamp(config: FlextModels.TimestampConfig) -> None:
        """Update timestamp for object using TimestampConfig model.

        Args:
            config: TimestampConfig containing object and timestamp settings

        """
        obj = config.obj

        # Only update the updated_at field if auto_update is enabled
        if config.auto_update:
            timezone = UTC if config.use_utc else None
            current_time = datetime.now(timezone)

            updated_field = config.field_names.get("updated_at", "updated_at")
            if hasattr(obj, updated_field):
                setattr(obj, updated_field, current_time)

    # =============================================================================
    # LOGGING METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def log_operation(config: FlextModels.LogOperation) -> None:
        """Log operation for object using LogOperation model.

        Args:
            config: LogOperation containing object and logging settings

        """
        # Create logger instance
        logger = FlextLogger(config.obj.__class__.__name__)

        # Prepare log context
        context = {
            "operation": config.operation,
            "object_type": type(config.obj).__name__,
            "timestamp": config.timestamp or datetime.now(UTC),
            **config.context,
        }

        # Log based on level - match Literal values exactly
        if config.level == "DEBUG":
            logger.debug(f"Operation: {config.operation}", extra=context)
        elif config.level == "INFO":
            logger.info(f"Operation: {config.operation}", extra=context)
        elif config.level == "WARNING":
            logger.warning(f"Operation: {config.operation}", extra=context)
        elif config.level == "ERROR":
            logger.error(f"Operation: {config.operation}", extra=context)
        elif config.level == "CRITICAL":
            logger.critical(f"Operation: {config.operation}", extra=context)

    # =============================================================================
    # STATE MANAGEMENT METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def initialize_state(request: FlextModels.StateInitializationRequest) -> None:
        """Initialize state for object using StateInitializationRequest model.

        Args:
            request: StateInitializationRequest containing object and state settings

        """
        obj = request.data

        # Validate and set state if object has the field
        if hasattr(obj, request.field_name):
            setattr(obj, request.field_name, request.state)

    # =============================================================================
    # UTILITY METHODS - Direct delegation to FlextUtilities
    # =============================================================================

    @staticmethod
    def clear_cache(obj: object) -> None:
        """Clear cache for object using FlextUtilities.

        Args:
            obj: Object to clear cache for

        """
        # Delegate cache clearing to FlextUtilities.Cache
        result = FlextUtilities.Cache.clear_object_cache(obj)

        # For backward compatibility, this method doesn't raise exceptions
        # But internally uses the FlextResult pattern for safety
        if result.is_failure:
            # Could optionally log the error using FlextLogger here
            pass

    @staticmethod
    def ensure_id(obj: object) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                # Use FlextUtilities to generate ID
                new_id = FlextUtilities.Generators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    # =============================================================================
    # MIXIN CLASSES - For inheritance hierarchy support
    # =============================================================================

    class Serializable:
        """Mixin for serialization capabilities.

        Provides marker class for objects that can be serialized using FlextMixins methods.
        """

    class Loggable:
        """Mixin for logging capabilities.

        Provides marker class for objects that can be logged using FlextMixins methods.
        """

    class Configurable:
        """Mixin for configuration capabilities.

        Components inheriting from this mixin should use native Pydantic accessors
        for configuration management. Retrieve values with direct attribute access
        (``config.debug``) or ``getattr`` and produce validated updates with
        attribute assignment or ``model_copy(update=...)``.

        Example:
            config = FlextConfig.get_global_instance()
            debug_mode = config.debug
            config.debug = True
            updated = config.model_copy(update={"timeout_seconds": 60})
        """

        pass

