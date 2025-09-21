"""Shared mixins anchoring serialization, logging, and timestamp helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from typing import cast

from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels


class FlextMixins:
    """Namespace containing the reusable mixins shared across FLEXT packages.

    They provide the logging, serialization, and state helpers needed to keep
    domain services aligned with the modernization plan without duplicating
    boilerplate.
    """

    class Serializable:
        """Serialization helpers reused in modernization-ready models."""

        def to_json(self, indent: int | None = None) -> str:
            """Convert to JSON string using FlextMixins infrastructure.

            Args:
                indent: Optional JSON indentation level

            Returns:
                JSON string representation of the object

            """
            # Delegate to the static method with self as the object
            return FlextMixins.to_json(self, indent)

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
                self.initialized = True

    @staticmethod
    def to_json(obj: object, indent: int | None = None) -> str:
        """Convert object to JSON string using Pydantic model validation.

        Args:
            obj: Object to serialize to JSON
            indent: Optional JSON indentation level (deprecated, use SerializationRequest)

        Returns:
            JSON string representation of the object

        Note:
            This method maintains backward compatibility. New code should use
            to_json_with_request() method with SerializationRequest model.

        """
        # Import here to avoid circular imports
        from flext_core.config import FlextConfig
        from flext_core.models import FlextModels

        # Get configuration settings
        config = FlextConfig()

        # Create SerializationRequest model for validation
        request = FlextModels.SerializationRequest(
            obj=obj,
            use_model_dump=True,
            indent=indent if indent is not None else config.json_indent,
            sort_keys=config.json_sort_keys,
            ensure_ascii=not config.ensure_json_serializable,
            encoding=config.serialization_encoding,
        )

        # Delegate to the model-based implementation
        return FlextMixins.to_json_with_request(request)

    @staticmethod
    def to_json_with_request(request: FlextModels.SerializationRequest) -> str:
        """Convert object to JSON string using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and options

        Returns:
            JSON string representation of the object

        """
        obj = request.obj

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
    def initialize_validation(obj: object) -> None:
        """Initialize validation for object using default configuration.

        Args:
            obj: Object to initialize validation for

        Note:
            This method maintains backward compatibility. New code should use
            initialize_validation_with_config() method with ValidationRequest model.

        """
        # Import here to avoid circular imports
        from flext_core.constants import FlextConstants
        from flext_core.models import FlextModels

        # Create ValidationRequest model with defaults
        request = FlextModels.ValidationRequest(
            obj=obj,
            validation_type=FlextConstants.Mixins.VALIDATION_BASIC,
            field_name=FlextConstants.Mixins.FIELD_VALIDATED,
            custom_validators=None,
            raise_on_failure=False,
        )

        # Delegate to the model-based implementation
        FlextMixins.initialize_validation_with_config(request)

    @staticmethod
    def initialize_validation_with_config(
        request: FlextModels.ValidationRequest,
    ) -> None:
        """Initialize validation for object using ValidationRequest model.

        Args:
            request: ValidationRequest containing object and validation settings

        """
        import contextlib

        obj = request.obj

        # Set the validation flag
        with contextlib.suppress(Exception):
            setattr(obj, request.field_name, True)

    @staticmethod
    def clear_cache(obj: object) -> None:
        """Clear cache for object using FlextUtilities.

        Args:
            obj: Object to clear cache for

        """
        from flext_core.utilities import FlextUtilities

        # Delegate cache clearing to FlextUtilities.Cache
        result = FlextUtilities.Cache.clear_object_cache(obj)

        # For backward compatibility, this method doesn't raise exceptions
        # But internally uses the FlextResult pattern for safety
        if result.is_failure:
            # Could optionally log the error using FlextLogger here
            pass

    @staticmethod
    def create_timestamp_fields(obj: object) -> None:
        """Create timestamp fields for object using default configuration.

        Args:
            obj: Object to add timestamp fields to

        Note:
            This method maintains backward compatibility. New code should use
            create_timestamp_fields_with_config() method with TimestampConfig model.

        """
        # Import here to avoid circular imports
        from flext_core.config import FlextConfig
        from flext_core.constants import FlextConstants
        from flext_core.models import FlextModels

        # Get configuration settings
        config = FlextConfig()

        # Create TimestampConfig model with defaults
        timestamp_config = FlextModels.TimestampConfig(
            obj=obj,
            use_utc=config.use_utc_timestamps,
            auto_update=config.timestamp_auto_update,
            field_names={
                "created_at": FlextConstants.Mixins.FIELD_CREATED_AT,
                "updated_at": FlextConstants.Mixins.FIELD_UPDATED_AT,
            },
            format_string=None,
        )

        # Delegate to the model-based implementation
        FlextMixins.create_timestamp_fields_with_config(timestamp_config)

    @staticmethod
    def create_timestamp_fields_with_config(config: FlextModels.TimestampConfig) -> None:
        """Create timestamp fields for object using TimestampConfig model.
        
        Args:
            config: TimestampConfig containing object and timestamp settings

        """
        from datetime import UTC, datetime

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
        # MyPy doesn't understand setattr modifies the object, but it does

    @staticmethod
    def ensure_id(obj: object) -> None:
        """Ensure object has an ID using FlextUtilities and FlextConstants.

        Args:
            obj: Object to ensure ID for

        """
        from flext_core.constants import FlextConstants
        from flext_core.utilities import FlextUtilities

        if hasattr(obj, FlextConstants.Mixins.FIELD_ID):
            id_value = getattr(obj, FlextConstants.Mixins.FIELD_ID, None)
            if not id_value:
                # Use FlextUtilities to generate ID
                new_id = FlextUtilities.Generators.generate_id()
                setattr(obj, FlextConstants.Mixins.FIELD_ID, new_id)

    @staticmethod
    def update_timestamp(obj: object) -> None:
        """Update timestamp for object using default configuration.

        Args:
            obj: Object to update timestamp for

        Note:
            This method maintains backward compatibility. New code should use
            update_timestamp_with_config() method with TimestampConfig model.

        """
        # Import here to avoid circular imports
        from flext_core.config import FlextConfig
        from flext_core.constants import FlextConstants
        from flext_core.models import FlextModels

        # Get configuration settings
        config = FlextConfig()

        # Create TimestampConfig model for updating timestamps only
        timestamp_config = FlextModels.TimestampConfig(
            obj=obj,
            use_utc=config.use_utc_timestamps,
            auto_update=config.timestamp_auto_update,
            field_names={
                "created_at": FlextConstants.Mixins.FIELD_CREATED_AT,
                "updated_at": FlextConstants.Mixins.FIELD_UPDATED_AT,
            },
            format_string=None,
        )

        # Delegate to the model-based implementation
        FlextMixins.update_timestamp_with_config(timestamp_config)

    @staticmethod
    def update_timestamp_with_config(config: FlextModels.TimestampConfig) -> None:
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

    @staticmethod
    def log_operation(obj: object, operation: str) -> None:
        """Log operation for object using default configuration.

        Args:
            obj: Object to log operation for
            operation: Operation name to log

        Note:
            This method maintains backward compatibility. New code should use
            log_operation_with_config() method with LogOperation model.

        """
        # Import here to avoid circular imports
        from flext_core.constants import FlextConstants
        from flext_core.models import FlextModels

        # Create LogOperation model with defaults
        log_config = FlextModels.LogOperation(
            obj=obj,
            operation=operation,
            level=FlextConstants.Mixins.LOG_LEVEL_INFO,
            context={},
            timestamp=None,
        )

        # Delegate to the model-based implementation
        FlextMixins.log_operation_with_config(log_config)

    @staticmethod
    def log_operation_with_config(config: FlextModels.LogOperation) -> None:
        """Log operation for object using LogOperation model.

        Args:
            config: LogOperation containing object and logging settings

        """
        from flext_core.loggings import FlextLogger

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
        # Simple logging - can be enhanced later

    @staticmethod
    def initialize_state(obj: object, state: str) -> None:
        """Initialize state for object using default configuration.

        Args:
            obj: Object to initialize state for
            state: Initial state value

        Note:
            This method maintains backward compatibility. New code should use
            initialize_state_with_config() method with StateInitializationRequest model.

        """
        # Import here to avoid circular imports
        from flext_core.constants import FlextConstants
        from flext_core.models import FlextModels

        # Create StateInitializationRequest model with defaults
        request = FlextModels.StateInitializationRequest(
            obj=obj,
            state=state,
            field_name=FlextConstants.Mixins.FIELD_STATE,
            validate_state=True,
            allowed_states=[
                FlextConstants.Mixins.STATE_ACTIVE,
                FlextConstants.Mixins.STATE_INACTIVE,
                FlextConstants.Mixins.STATE_PENDING,
                FlextConstants.Mixins.STATE_COMPLETED,
                FlextConstants.Mixins.STATE_FAILED,
            ],
        )

        # Delegate to the model-based implementation
        FlextMixins.initialize_state_with_config(request)

    @staticmethod
    def initialize_state_with_config(
        request: FlextModels.StateInitializationRequest,
    ) -> None:
        """Initialize state for object using StateInitializationRequest model.

        Args:
            request: StateInitializationRequest containing object and state settings

        """
        obj = request.obj

        # Validate state if object has the field
        if hasattr(obj, request.field_name):
            # Set the state value
            setattr(obj, request.field_name, request.state)

    @staticmethod
    def to_dict(obj: object) -> dict[str, object]:
        """Convert object to dictionary using default configuration.

        Args:
            obj: Object to convert to dictionary

        Returns:
            Dictionary representation of the object

        Note:
            This method maintains backward compatibility. New code should use
            to_dict_with_request() method with SerializationRequest model.

        """
        # Import here to avoid circular imports
        from flext_core.config import FlextConfig
        from flext_core.models import FlextModels

        # Get configuration settings
        config = FlextConfig()

        # Create SerializationRequest model for validation
        request = FlextModels.SerializationRequest(
            obj=obj,
            use_model_dump=True,
            indent=config.json_indent,
            sort_keys=config.json_sort_keys,
            ensure_ascii=not config.ensure_json_serializable,
            encoding=config.serialization_encoding,
        )

        # Delegate to the model-based implementation
        return FlextMixins.to_dict_with_request(request)

    @staticmethod
    def to_dict_with_request(
        request: FlextModels.SerializationRequest,
    ) -> dict[str, object]:
        """Convert object to dictionary using SerializationRequest model.

        Args:
            request: SerializationRequest containing object and options

        Returns:
            Dictionary representation of the object

        """
        obj = request.obj

        # Try model_dump method first if requested
        if request.use_model_dump and hasattr(obj, "model_dump"):
            model_dump_method = getattr(obj, "model_dump", None)
            if model_dump_method is not None and callable(model_dump_method):
                result = model_dump_method()
                return result if isinstance(result, dict) else {"model_dump": result}

        # Try __dict__ method with proper type casting
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
            return dict(obj_dict) if isinstance(obj_dict, dict) else {"__dict__": obj_dict}

        # Fallback to type and value representation
        return {"type": type(obj).__name__, "value": str(obj)}
