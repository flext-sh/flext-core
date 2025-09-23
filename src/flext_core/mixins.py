"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextMixins:
    """Simplified mixin class providing essential behaviors for FLEXT ecosystem.

    Follows FLEXT quality standards:
    - Single class per module architecture
    - Type-safe Pydantic-only method signatures
    - No backward compatibility wrappers or aliases
    - Direct implementation leveraging existing FLEXT components

    Reduced complexity by delegating to FlextUtilities, FlextConfig, and other
    specialized FLEXT components where appropriate.
    """

    # =============================================================================
    # SERIALIZATION METHODS - Simplified using FlextUtilities patterns
    # =============================================================================

    @staticmethod
    def to_json(request: FlextModels.SerializationRequest) -> str:
        """Convert object to JSON string using SerializationRequest model.

        Simplified implementation leveraging Pydantic's model_dump when available,
        with fallback to __dict__ serialization.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            JSON string representation of the object

        """
        obj = request.data

        # Use Pydantic model_dump if available and requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            # Type narrow obj to have model_dump method
            model_obj = obj
            data = getattr(model_obj, "model_dump")()
            return json.dumps(
                data,
                indent=request.indent,
                sort_keys=request.sort_keys,
                ensure_ascii=request.ensure_ascii,
            )

        # Fallback to __dict__ for simple objects
        if hasattr(obj, "__dict__"):
            data = obj.__dict__
            return json.dumps(
                data,
                indent=request.indent,
                sort_keys=request.sort_keys,
                ensure_ascii=request.ensure_ascii,
            )

        # Final fallback to string representation
        return json.dumps(
            str(obj),
            indent=request.indent,
            sort_keys=request.sort_keys,
            ensure_ascii=request.ensure_ascii,
        )

    @staticmethod
    def to_dict(request: FlextModels.SerializationRequest) -> FlextTypes.Core.Dict:
        """Convert object to dictionary using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            Dictionary representation of the object

        """
        obj = request.data

        # Use Pydantic model_dump if available and requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            # Type narrow obj to have model_dump method
            model_obj = obj
            result = getattr(model_obj, "model_dump")()
            if isinstance(result, dict):
                return cast("FlextTypes.Core.Dict", result)
            return cast("FlextTypes.Core.Dict", {"model_dump": result})

        # Use __dict__ if available
        if hasattr(obj, "__dict__"):
            return cast("FlextTypes.Core.Dict", obj.__dict__)

        # Fallback to type representation
        return cast(
            "FlextTypes.Core.Dict", {"type": type(obj).__name__, "value": str(obj)}
        )

    # =============================================================================
    # TIMESTAMP METHODS - Using FlextConfig for global settings
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

        # Set created_at if not already set
        created_field = config.field_names.get("created_at", "created_at")
        if hasattr(obj, created_field) and getattr(obj, created_field, None) is None:
            setattr(obj, created_field, current_time)

        # Set updated_at if auto_update is enabled
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

        # Check global configuration using FlextConfig
        global_config = FlextConfig.get_global_instance()
        global_auto_update = getattr(global_config, "timestamp_auto_update", False)

        # Update if auto_update is enabled locally or globally
        auto_update_enabled = config.auto_update or global_auto_update

        if auto_update_enabled:
            timezone = UTC if config.use_utc else None
            current_time = datetime.now(timezone)

            updated_field = config.field_names.get("updated_at", "updated_at")
            if hasattr(obj, updated_field):
                setattr(obj, updated_field, current_time)

    # =============================================================================
    # LOGGING METHODS - Simplified using FlextLogger directly
    # =============================================================================

    @staticmethod
    def log_operation(config: FlextModels.LogOperation) -> None:
        """Log operation for object using LogOperation model.

        Args:
            config: LogOperation containing object and logging settings

        """
        logger = FlextLogger(config.obj.__class__.__name__)

        context = {
            "operation": config.operation,
            "object_type": type(config.obj).__name__,
            "timestamp": config.timestamp or datetime.now(UTC),
            **config.context,
        }

        # Use FlextConstants for log level mapping
        normalized_level = str(config.level).upper()
        level_methods: dict[str, Callable[..., None]] = {
            FlextConstants.Logging.DEBUG: logger.debug,
            FlextConstants.Logging.INFO: logger.info,
            FlextConstants.Logging.WARNING: logger.warning,
            FlextConstants.Logging.ERROR: logger.error,
            FlextConstants.Logging.CRITICAL: logger.critical,
        }

        log_method = level_methods.get(normalized_level, logger.info)
        log_method(f"Operation: {config.operation}", extra=context)

    # =============================================================================
    # SIMPLIFIED UTILITY METHODS - Direct delegation to FlextUtilities
    # =============================================================================

    @staticmethod
    def initialize_validation(obj: object, field_name: str) -> None:
        """Initialize validation for object.

        Simplified implementation that directly sets the validation flag.

        Args:
            obj: Object to set validation on
            field_name: Name of the field to set validation flag

        """
        with contextlib.suppress(Exception):
            setattr(obj, field_name, True)

    @staticmethod
    def initialize_state(request: FlextModels.StateInitializationRequest) -> None:
        """Initialize state for object using StateInitializationRequest model.

        Args:
            request: StateInitializationRequest containing object and state settings

        """
        obj = request.data
        if hasattr(obj, request.field_name):
            setattr(obj, request.field_name, request.state)

    @staticmethod
    def clear_cache(obj: object) -> None:
        """Clear cache for object using FlextUtilities.

        Direct delegation to FlextUtilities.Cache for cache clearing.

        Args:
            obj: Object to clear cache for

        """
        FlextUtilities.Cache.clear_object_cache(obj)

    @staticmethod
    def ensure_id(obj: object) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                new_id = FlextUtilities.Generators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    # =============================================================================
    # CONFIGURATION METHODS - Simplified using FlextConfig native methods
    # =============================================================================

    @staticmethod
    def get_config_parameter(obj: object, parameter: str) -> object:
        """Get parameter value from a Pydantic configuration object.

        Simplified implementation using Pydantic's model_dump for safe access.

        Args:
            obj: The configuration object (must have model_dump method)
            parameter: The parameter name to retrieve (must exist in model)

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model

        """
        # Check for Pydantic model with model_dump method
        if hasattr(obj, "model_dump"):
            model_dump_attr = getattr(obj, "model_dump")
            if callable(model_dump_attr):
                model_data = cast("FlextTypes.Core.Dict", model_dump_attr())
                if parameter not in model_data:
                    msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
                    raise KeyError(msg)
                return model_data[parameter]

        # Fallback for non-Pydantic objects - direct attribute access
        if not hasattr(obj, parameter):
            msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
            raise KeyError(msg)
        return getattr(obj, parameter)

    @staticmethod
    def set_config_parameter(obj: object, parameter: str, value: object) -> bool:
        """Set parameter value on a Pydantic configuration object with validation.

        Simplified implementation using direct attribute assignment with Pydantic validation.

        Args:
            obj: The configuration object (Pydantic BaseSettings instance)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        """
        try:
            # Check if parameter exists in model fields for Pydantic objects
            if hasattr(obj, "model_fields"):
                model_fields_attr = getattr(obj, "model_fields")
                if model_fields_attr is not None:
                    model_fields = cast("FlextTypes.Core.Dict", model_fields_attr)
                    if parameter not in model_fields:
                        return False

            # Use setattr which triggers Pydantic validation if applicable
            setattr(obj, parameter, value)
            return True

        except Exception:
            # object validation error or attribute error returns False
            return False

    @staticmethod
    def get_singleton_parameter(singleton_class: type, parameter: str) -> object:
        """Get parameter from a singleton configuration instance.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to retrieve

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model
            AttributeError: If class doesn't have get_global_instance method

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                return FlextMixins.get_config_parameter(instance, parameter)

        msg = (
            f"Class {singleton_class.__name__} does not have get_global_instance method"
        )
        raise AttributeError(msg)

    @staticmethod
    def set_singleton_parameter(
        singleton_class: type, parameter: str, value: object
    ) -> bool:
        """Set parameter on a singleton configuration instance with validation.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                return FlextMixins.set_config_parameter(instance, parameter, value)

        return False

    # =============================================================================
    # MIXIN CLASSES - Preserved for inheritance patterns
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
        attribute assignment or ``model_copy(update=...)```.

        Example:
            config = FlextConfig.get_global_instance()
            debug_mode = config.debug
            config.debug = True
            updated = config.model_copy(update={"timeout_seconds": 60})

        """
