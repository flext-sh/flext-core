"""Utilities module - FlextUtilitiesConfiguration.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from typing import cast

from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

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
                # Cast to protocol with model_dump for type safety
                pydantic_obj = cast("FlextProtocols.HasModelDump", obj)
                model_data: dict[str, object] = pydantic_obj.model_dump()
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
                model_fields = type(obj).model_fields
                if parameter not in model_fields:
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


__all__ = ["FlextUtilitiesConfiguration"]
