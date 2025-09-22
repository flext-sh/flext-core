"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.typings import FlextTypes
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
    def to_dict(request: FlextModels.SerializationRequest) -> FlextTypes.Core.Dict:
        """Convert object to dictionary using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and serialization options

        Returns:
            Dictionary representation of the object as FlextTypes.Core.Dict

        """
        obj = request.data

        # Try model_dump method first if requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                result = model_dump_method()
                if isinstance(result, dict):
                    # Type-safe conversion with explicit casting
                    return cast("FlextTypes.Core.Dict", result)
                return cast(
                    "FlextTypes.Core.Dict",
                    {"model_dump": result},
                )

        # Try __dict__ method with proper type casting
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
            # obj.__dict__ is always a FlextTypes.Core.Dict, so we can safely cast it
            return cast("FlextTypes.Core.Dict", obj_dict)

        # Fallback to type and value representation
        return cast(
            "FlextTypes.Core.Dict",
            {"type": type(obj).__name__, "value": str(obj)},
        )

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

        normalized_level = str(config.level).upper()
        level_method_map: dict[str, Callable[..., None]] = {
            FlextConstants.Logging.DEBUG: logger.debug,
            FlextConstants.Logging.INFO: logger.info,
            FlextConstants.Logging.WARNING: logger.warning,
            FlextConstants.Logging.ERROR: logger.error,
            FlextConstants.Logging.CRITICAL: logger.critical,
        }

        log_method = level_method_map.get(normalized_level)

        if log_method is None:
            normalized_level = FlextConstants.Logging.DEFAULT_LEVEL
            log_method = level_method_map.get(normalized_level, logger.info)

        log_method(f"Operation: {config.operation}", extra=context)

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
    # CONFIGURATION METHODS - Direct Pydantic implementation
    # =============================================================================

    @staticmethod
    def get_config_parameter(obj: object, parameter: str) -> object:
        """Get any parameter value from a Pydantic configuration object.

        This method works seamlessly with Pydantic Settings by using
        the model's model_dump method for safe parameter access.
        The parameter MUST be defined in the model - no default fallback.

        Args:
            obj: The configuration object (must have model_dump method)
            parameter: The parameter name to retrieve (must exist in model)

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model

        Example:
            config = FlextConfig.get_global_instance()
            debug_mode = FlextMixins.get_config_parameter(config, 'debug')
            log_level = FlextMixins.get_config_parameter(config, 'log_level')

        """
        # Use Pydantic's model_dump to get all fields as dict
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            model_dump_method = getattr(obj, "model_dump")
            model_data: FlextTypes.Core.Dict = cast(
                "FlextTypes.Core.Dict", model_dump_method()
            )
            if parameter not in model_data:
                msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
                raise KeyError(msg)
            return model_data[parameter]

        # Fallback for non-Pydantic objects - check if attribute exists
        if not hasattr(obj, parameter):
            msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
            raise KeyError(msg)
        return getattr(obj, parameter)

    @staticmethod
    def set_config_parameter(obj: object, parameter: str, value: object) -> bool:
        """Set any parameter value on a Pydantic configuration object with full validation.

        This method works with Pydantic Settings and maintains all validation,
        including field validators, model validators, and secrets handling.

        Args:
            obj: The configuration object (Pydantic BaseSettings instance)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        Example:
            config = FlextConfig.get_global_instance()
            success = FlextMixins.set_config_parameter(config, 'debug', True)
            success = FlextMixins.set_config_parameter(config, 'log_level', 'DEBUG')

        """
        try:
            # Check if the parameter exists in the model fields
            if hasattr(obj, "model_fields"):
                model_fields: FlextTypes.Core.Dict = cast(
                    "FlextTypes.Core.Dict", getattr(obj, "model_fields")
                )
                if parameter not in model_fields:
                    return False

            # Use Pydantic's setattr which triggers validation
            setattr(obj, parameter, value)

            # If the object has model validation, trigger it
            if hasattr(obj, "model_validate") and hasattr(obj, "model_dump"):
                # Get current model data and re-validate to ensure consistency
                model_dump_method = getattr(obj, "model_dump")
                current_data: FlextTypes.Core.Dict = cast(
                    "FlextTypes.Core.Dict", model_dump_method()
                )
                current_data[parameter] = value
                # This will trigger all field and model validators
                model_validate_method = getattr(obj.__class__, "model_validate")
                validated_instance: BaseModel = model_validate_method(current_data)

                # Update all fields from the validated instance
                validated_dump: dict[str, Any] = validated_instance.model_dump()
                for field_name, field_value in validated_dump.items():
                    setattr(obj, field_name, field_value)

            return True

        except Exception:
            # Validation failed or other error occurred
            return False

    @staticmethod
    def get_singleton_parameter(singleton_class: type, parameter: str) -> object:
        """Get any parameter from a singleton configuration instance.

        This method assumes the singleton class has a get_global_instance() method
        and uses FlextMixins.get_config_parameter internally.
        The parameter MUST be defined in the model.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to retrieve (must exist in model)

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model
            AttributeError: If class doesn't have get_global_instance method

        Example:
            debug_mode = FlextMixins.get_singleton_parameter(FlextConfig, 'debug')
            log_level = FlextMixins.get_singleton_parameter(FlextConfig, 'log_level')

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
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
        """Set any parameter on a singleton configuration instance with validation.

        This method assumes the singleton class has a get_global_instance() method
        and uses FlextMixins.set_config_parameter internally with full Pydantic validation.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        Example:
            success = FlextMixins.set_singleton_parameter(FlextConfig, 'debug', True)
            success = FlextMixins.set_singleton_parameter(FlextConfig, 'log_level', 'DEBUG')

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = getattr(singleton_class, "get_global_instance")
            instance = get_global_instance_method()
            return FlextMixins.set_config_parameter(instance, parameter, value)

        msg = (
            f"Class {singleton_class.__name__} does not have get_global_instance method"
        )
        raise AttributeError(msg)

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
