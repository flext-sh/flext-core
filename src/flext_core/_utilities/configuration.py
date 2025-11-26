"""Utilities module - FlextUtilitiesConfiguration.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# TypeVar for generic Pydantic model type
T_Model = TypeVar("T_Model", bound=BaseModel)

_logger = logging.getLogger(__name__)


class FlextUtilitiesConfiguration:
    """Configuration parameter access and manipulation utilities."""

    @staticmethod
    def get_parameter(obj: object, parameter: str) -> FlextTypes.ParameterValueType:
        """Get parameter value from a Pydantic configuration object.

        Simplified implementation using Pydantic's model_dump for safe access.

        Args:
            obj: The configuration object (must have model_dump method or dict-like access)
            parameter: The parameter name to retrieve (must exist in model)

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not defined in the model

        """
        # Check for dict-like access first
        if FlextRuntime.is_dict_like(obj):
            if parameter not in obj:
                msg = f"Parameter '{parameter}' is not defined"
                raise FlextExceptions.NotFoundError(msg, resource_id=parameter)
            return obj[parameter]

        # Check for Pydantic model with model_dump method
        model_dump_method = getattr(obj, "model_dump", None)
        if model_dump_method is not None and callable(model_dump_method):
            try:
                # obj has model_dump method, call it directly
                model_data_raw = model_dump_method()
                if not isinstance(model_data_raw, dict):
                    msg = f"model_dump() must return dict, got {type(model_data_raw).__name__}"
                    raise TypeError(msg)
                model_data: dict[str, object] = model_data_raw
                if parameter not in model_data:
                    msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
                    raise FlextExceptions.NotFoundError(msg, resource_id=parameter)
                return model_data[parameter]
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                # Log and continue to fallback - object may not be Pydantic model
                _logger.debug("Failed to get parameter from model_dump: %s", e)

        # Fallback for non-Pydantic objects - direct attribute access
        if not hasattr(obj, parameter):
            msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
            raise FlextExceptions.NotFoundError(
                msg,
                resource_type=f"parameter '{parameter}'",
            )
        return getattr(obj, parameter)

    @staticmethod
    def set_parameter(
        obj: object,
        parameter: str,
        value: FlextTypes.ParameterValueType,
    ) -> bool:
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
            if isinstance(obj, FlextProtocols.HasModelFields):
                # Access model_fields from class, not instance (Pydantic 2.11+ compatibility)
                model_fields_dict = getattr(type(obj), "model_fields", {})
                if (
                    not isinstance(model_fields_dict, dict)
                    or parameter not in model_fields_dict
                ):
                    return False

            # Use setattr which triggers Pydantic validation if applicable
            setattr(obj, parameter, value)
            return True

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            # Validation error or attribute error returns False
            return False

    @staticmethod
    def get_singleton(
        singleton_class: type,
        parameter: str,
    ) -> FlextTypes.ParameterValueType:
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
            get_global_instance_method = singleton_class.get_global_instance
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                if isinstance(instance, FlextProtocols.HasModelDump):
                    return FlextUtilitiesConfiguration.get_parameter(
                        instance,
                        parameter,
                    )

        msg = (
            f"Class {singleton_class.__name__} does not have get_global_instance method"
        )
        raise FlextExceptions.ValidationError(msg)

    @staticmethod
    def set_singleton(
        singleton_class: type,
        parameter: str,
        value: FlextTypes.ParameterValueType,
    ) -> FlextResult[bool]:
        """Set parameter on a singleton configuration instance with validation.

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            FlextResult[bool] indicating success or failure

        """
        if not hasattr(singleton_class, "get_global_instance"):
            return FlextResult[bool].fail(
                f"Class {singleton_class.__name__} does not have get_global_instance method",
            )

        get_global_instance_method = singleton_class.get_global_instance
        if not callable(get_global_instance_method):
            return FlextResult[bool].fail(
                f"get_global_instance is not callable on {singleton_class.__name__}",
            )

        instance = get_global_instance_method()
        if not isinstance(instance, FlextProtocols.HasModelDump):
            return FlextResult[bool].fail(
                "Instance does not implement HasModelDump protocol",
            )

        success = FlextUtilitiesConfiguration.set_parameter(instance, parameter, value)
        if success:
            return FlextResult[bool].ok(True)
        return FlextResult[bool].fail(
            f"Failed to set parameter '{parameter}' on {singleton_class.__name__}",
        )

    @staticmethod
    def validate_config_class(config_class: object) -> tuple[bool, str | None]:
        """Validate that a configuration class is properly configured.

        Checks that the class:
        - Is a type (not an instance)
        - Has proper model_config for environment binding
        - Can be instantiated without errors

        Args:
            config_class: Configuration class to validate

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if valid
                - (False, error_message) if invalid

        """
        try:
            # Check that it's a class
            if not isinstance(config_class, type):
                return (False, "config_class must be a class, not an instance")

            # Check model_config existence
            class_name = getattr(config_class, "__name__", "UnknownClass")
            if not hasattr(config_class, "model_config"):
                return (False, f"{class_name} must define model_config")

            # Try to instantiate to ensure it's valid
            _ = config_class()

            return (True, None)

        except Exception as e:
            return (False, f"Configuration class validation failed: {e!s}")

    @staticmethod
    def create_settings_config(
        env_prefix: str,
        env_file: str | None = None,
        env_nested_delimiter: str = "__",
    ) -> dict[str, object]:
        """Create a SettingsConfigDict for environment binding.

        Helper method for creating proper Pydantic v2 SettingsConfigDict
        that enables automatic environment variable binding.

        Args:
            env_prefix: Environment variable prefix (e.g., "MYAPP_")
                       All env vars matching this prefix will be loaded
            env_file: Optional path to .env file
            env_nested_delimiter: Delimiter for nested configs (default: "__")
                                 Example: MYAPP_DB__HOST → nested config binding

        Returns:
            dict: Pydantic v2 settings configuration dictionary

        """
        return {
            "env_prefix": env_prefix,
            "env_file": env_file,
            "env_nested_delimiter": env_nested_delimiter,
            "case_sensitive": False,
            "extra": "ignore",
            "validate_default": True,
        }

    @staticmethod
    def build_options_from_kwargs(
        model_class: type[T_Model],
        explicit_options: T_Model | None,
        default_factory: Callable[[], T_Model],
        **kwargs: object,
    ) -> FlextResult[T_Model]:
        """Build Pydantic options model from explicit options or kwargs.

        Generic utility for the Options+Config+kwargs pattern. Handles three cases:
        1. If explicit_options provided: use it (with kwargs overrides)
        2. Otherwise: get defaults from config via default_factory()
        3. Apply kwargs overrides to either case

        Architecture:
            - WriteFormatOptions/ParseFormatOptions remain as Pydantic Models
            - Config provides defaults via to_write_options() / to_parse_options()
            - Public methods accept **kwargs for convenience
            - This method converts kwargs → validated Pydantic model

        Example Usage:
            def write(
                self,
                entries: list[Entry],
                format_options: WriteFormatOptions | None = None,
                **format_kwargs: object,
            ) -> FlextResult[str]:
                options_result = FlextUtilities.Configuration.build_options_from_kwargs(
                    model_class=WriteFormatOptions,
                    explicit_options=format_options,
                    default_factory=lambda: self.config.ldif.to_write_options(),
                    **format_kwargs,
                )
                if options_result.is_failure:
                    return FlextResult[str].fail(options_result.error)
                options = options_result.unwrap()
                # ... use options

        Args:
            model_class: The Pydantic model class (e.g., WriteFormatOptions)
            explicit_options: Explicitly provided options instance, or None
            default_factory: Callable that returns default options from config
            **kwargs: Individual option overrides (snake_case field names)

        Returns:
            FlextResult[T_Model]: Validated options model or error

        """
        try:
            # Step 1: Get base options (explicit or from config defaults)
            if explicit_options is not None:
                base_options = explicit_options
            else:
                base_options = default_factory()

            # Step 2: If no kwargs, return base options directly
            if not kwargs:
                return FlextResult[T_Model].ok(base_options)

            # Step 3: Get valid field names from model class
            # Access model_fields as class attribute for type safety
            model_fields_attr = getattr(model_class, "model_fields", {})
            model_fields: dict[str, object] = (
                model_fields_attr if isinstance(model_fields_attr, dict) else {}
            )
            valid_field_names = set(model_fields.keys())

            # Step 4: Filter kwargs to only valid field names
            valid_kwargs: dict[str, object] = {}
            invalid_kwargs: list[str] = []

            for key, value in kwargs.items():
                if key in valid_field_names:
                    valid_kwargs[key] = value
                else:
                    invalid_kwargs.append(key)

            # Get class name for logging
            class_name = getattr(model_class, "__name__", "UnknownModel")

            # Step 5: Log warning for invalid kwargs (don't fail)
            if invalid_kwargs:
                _logger.warning(
                    "Ignored invalid kwargs for %s: %s. Valid fields: %s",
                    class_name,
                    invalid_kwargs,
                    sorted(valid_field_names),
                )

            # Step 6: If no valid overrides, return base options
            if not valid_kwargs:
                return FlextResult[T_Model].ok(base_options)

            # Step 7: Create new model with base values + kwargs overrides
            # Use model_dump to get base values, then update with kwargs
            base_dict = base_options.model_dump()
            base_dict.update(valid_kwargs)

            # Step 8: Validate and create new model instance
            merged_options = model_class(**base_dict)

            return FlextResult[T_Model].ok(merged_options)

        except (TypeError, ValueError) as e:
            # Pydantic validation error
            class_name = getattr(model_class, "__name__", "UnknownModel")
            return FlextResult[T_Model].fail(
                f"Failed to build {class_name}: {e}",
            )
        except Exception as e:
            # Unexpected error
            class_name = getattr(model_class, "__name__", "UnknownModel")
            _logger.exception("Unexpected error building options model")
            return FlextResult[T_Model].fail(
                f"Unexpected error building {class_name}: {e}",
            )


__all__ = ["FlextUtilitiesConfiguration", "T_Model"]
