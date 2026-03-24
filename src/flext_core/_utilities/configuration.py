"""Configuration helpers for parameter access and manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, MutableSequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from flext_core import FlextRuntime, T_Model, c, e, p, r, t


class FlextUtilitiesConfiguration:
    """Configuration utilities for parameter access and manipulation."""

    @staticmethod
    def _duck_dump_get_parameter(
        obj: p.HasModelDump | BaseModel | p.Base,
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        """Get parameter from duck-typed model_dump() result.

        Instead of iterating the dump dict (which has Unknown types from getattr),
        directly look up the parameter using subscript access.
        """
        model_dump_fn = getattr(obj, "model_dump", None)
        if model_dump_fn is None or not callable(model_dump_fn):
            return (False, None)
        raw = model_dump_fn()
        get_fn = getattr(raw, "get", None)
        if get_fn is None or not callable(get_fn):
            return (False, None)

        class _Sentinel:
            pass

        sentinel = _Sentinel()
        val = get_fn(parameter, sentinel)
        if val is sentinel:
            return (False, None)
        if val is None or isinstance(val, (str, int, float, bool, datetime, Path)):
            return (True, val)
        if isinstance(val, BaseModel):
            return (True, val)
        return (True, str(val))

    @staticmethod
    def _get_logger() -> p.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def get_log_level_from_config() -> int:
        """Get log level from default constant (avoids circular import with config.py).

        Returns:
            int: Numeric logging level (e.g., logging.INFO = 20)

        """
        default_log_level = c.DEFAULT_LEVEL.upper()
        return getattr(logging, default_log_level, logging.INFO)

    @staticmethod
    def resolve_env_file() -> str:
        """Resolve .env file path from FLEXT_ENV_FILE environment variable.

        Returns:
            str: Path to .env file (custom, discovered, or default ".env")

        """
        custom_env_file = os.environ.get(c.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        default_path = Path.cwd() / c.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return c.ENV_FILE_DEFAULT

    _NOT_FOUND: tuple[bool, None] = (False, None)

    @staticmethod
    def _try_get_attr(
        obj: p.HasModelDump | BaseModel | p.Base,
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get attribute value from t.NormalizedValue via direct attribute access.

        Args:
            obj: Object instance that might have the parameter as attribute
            parameter: Attribute name to retrieve

        Returns:
            (True, value) if attribute exists, (False, None) if not

        """
        obj_vars: t.ContainerMapping = vars(obj) if hasattr(obj, "__dict__") else {}
        if parameter not in obj_vars:
            return FlextUtilitiesConfiguration._NOT_FOUND
        value = obj_vars[parameter]
        return (True, value)

    @staticmethod
    def _try_get_from_dict_like(
        obj: Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get parameter from dict-like t.NormalizedValue.

        Args:
            obj: Potentially dict-like t.NormalizedValue
            parameter: Key to retrieve

        Returns:
            (True, value) if key exists, (False, None) if not dict-like or missing

        """
        if parameter in obj:
            return (True, obj[parameter])
        return FlextUtilitiesConfiguration._NOT_FOUND

    @staticmethod
    def _try_get_from_duck_model_dump(
        obj: p.HasModelDump | BaseModel | p.Base,
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        try:
            return FlextUtilitiesConfiguration._duck_dump_get_parameter(obj, parameter)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        return FlextUtilitiesConfiguration._NOT_FOUND

    @staticmethod
    def _try_get_from_model_dump(
        obj: p.HasModelDump,
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get parameter from HasModelDump protocol t.NormalizedValue.

        Args:
            obj: Object implementing HasModelDump protocol
            parameter: Field name to retrieve from dumped dict

        Returns:
            (True, value) if found in model_dump(), (False, None) if not

        """
        try:
            obj_dict = obj.model_dump()
            if parameter in obj_dict:
                return (True, obj_dict[parameter])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        return FlextUtilitiesConfiguration._NOT_FOUND

    @staticmethod
    def build_options_from_kwargs(
        model_class: type[T_Model],
        explicit_options: T_Model | None,
        default_factory: Callable[[], T_Model],
        **kwargs: t.Scalar,
    ) -> FlextRuntime.RuntimeResult[T_Model]:
        """Build Pydantic options model from explicit options or kwargs.

        Args:
            model_class: The Pydantic model class (e.g., WriteFormatOptions)
            explicit_options: Explicitly provided options instance, or None
            default_factory: Callable that returns default options from config
            **kwargs: Individual option overrides (snake_case field names)

        Returns:
            r[T_Model]: ok(validated_model) or fail(error_msg)

        """
        try:
            if explicit_options is not None:
                base_options = explicit_options
            else:
                base_options = default_factory()
            if not kwargs:
                return r[T_Model].ok(base_options)
            base_class: type[BaseModel] = model_class
            valid_field_names: set[str] = set(base_class.model_fields.keys())
            valid_kwargs = t.ConfigMap(root={})
            invalid_kwargs: MutableSequence[str] = []
            for key, value in kwargs.items():
                if key in valid_field_names:
                    valid_kwargs[key] = value
                else:
                    invalid_kwargs.append(key)
            class_name = getattr(model_class, "__name__", "UnknownModel")
            if invalid_kwargs:
                FlextUtilitiesConfiguration._get_logger().warning(
                    "Ignored invalid kwargs for %s: %s. Valid fields: %s",
                    class_name,
                    str(invalid_kwargs),
                    str(sorted(valid_field_names)),
                )
            if not valid_kwargs:
                return r[T_Model].ok(base_options)
            base_dict = base_options.model_dump()
            base_dict.update(valid_kwargs)
            merged_options = model_class(**base_dict)
            return r[T_Model].ok(merged_options)
        except (TypeError, ValueError) as e:
            class_name = getattr(model_class, "__name__", "UnknownModel")
            return r[T_Model].fail(f"Failed to build {class_name}: {e}")
        except (AttributeError, RuntimeError, KeyError) as e:
            class_name = getattr(model_class, "__name__", "UnknownModel")
            FlextUtilitiesConfiguration._get_logger().exception(
                "Unexpected error building options model",
            )
            return r[T_Model].fail(f"Unexpected error building {class_name}: {e}")

    @staticmethod
    def bulk_register(
        container: p.Container,
        registrations: Mapping[str, t.Scalar | t.ConfigMap | t.Dict],
    ) -> r[int]:
        """Register multiple services at once.

        Args:
            container: Container to register in (must implement DI protocol).
            registrations: Mapping of name to service instance or factory.

        Returns:
            r[int]: Success with count of registered services, or failure.

        """
        count = 0
        for name, value in registrations.items():
            try:
                register_result = container.register(name, value)
                if not isinstance(register_result, p.ResultLike):
                    return r[int].fail(
                        f"Bulk registration failed at {name}: register returned non-result",
                    )
                if register_result.is_failure:
                    return r[int].fail(
                        f"Bulk registration failed at {name}: {register_result.error}",
                    )
                count += 1
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[int].fail(f"Bulk registration failed at {name}: {e}")
        return r[int].ok(count)

    @staticmethod
    def create_settings_config(
        env_prefix: str,
        env_file: str = c.ENV_FILE_DEFAULT,
        env_nested_delimiter: str = "__",
    ) -> t.ConfigurationMapping:
        """Create a SettingsConfigDict for environment binding.

        Args:
            env_prefix: Environment variable prefix (e.g., "FLEXT_LDAP_")
            env_file: Optional path to .env file
            env_nested_delimiter: Delimiter for nested configs (default: "__")

        Returns:
            dict: Configuration compatible with Pydantic v2 SettingsConfigDict

        """
        return {
            "env_prefix": env_prefix,
            "env_file": env_file,
            "env_nested_delimiter": env_nested_delimiter,
            "case_sensitive": False,
            "extra": c.EXTRA_IGNORE,
            "validate_default": True,
        }

    @staticmethod
    def get_parameter(
        obj: p.HasModelDump | BaseModel | p.Base | Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> t.ValueOrModel:
        """Get parameter value from a configuration t.NormalizedValue.

        Args:
            obj: Configuration t.NormalizedValue (HasModelDump, dict-like, or with attributes)
            parameter: Parameter name to retrieve

        Returns:
            The parameter value (can be None if that's the stored value)

        Raises:
            e.NotFoundError: If parameter is not defined

        """
        return FlextUtilitiesConfiguration._resolve_parameter(obj, parameter)

    @staticmethod
    def _resolve_from_mapping(
        obj: Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> tuple[bool, t.ValueOrModel]:
        return FlextUtilitiesConfiguration._try_get_from_dict_like(obj, parameter)

    @staticmethod
    def _resolve_from_obj(
        obj: p.HasModelDump | BaseModel | p.Base,
        parameter: str,
    ) -> t.ValueOrModel:
        if isinstance(obj, p.HasModelDump):
            found, value = FlextUtilitiesConfiguration._try_get_from_model_dump(
                obj,
                parameter,
            )
            if found:
                return value
        found_d, duck_v = FlextUtilitiesConfiguration._try_get_from_duck_model_dump(
            obj,
            parameter,
        )
        if found_d:
            return duck_v
        found_a, attr_v = FlextUtilitiesConfiguration._try_get_attr(obj, parameter)
        if found_a:
            return attr_v
        class_name = obj.__class__.__name__
        msg = f"Parameter '{parameter}' is not defined in {class_name}"
        raise e.NotFoundError(msg)

    @staticmethod
    def _resolve_parameter(
        obj: p.HasModelDump | BaseModel | p.Base | Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> t.ValueOrModel:
        if isinstance(obj, BaseModel):
            return FlextUtilitiesConfiguration._resolve_from_obj(obj, parameter)
        if isinstance(obj, p.HasModelDump):
            return FlextUtilitiesConfiguration._resolve_from_obj(obj, parameter)
        # Mapping branch: obj is p.Base | Mapping[str, NormalizedValue | BaseModel]
        # Use _resolve_from_mapping for Mapping, fallback to _resolve_from_obj
        # Attempt to resolve from mapping-like objects using attribute access

        def _default_get(_k: str) -> t.ValueOrModel | None:
            return None

        contains_method = getattr(obj, "__contains__", None)
        if callable(contains_method) and contains_method(parameter):
            get_method: Callable[[str], t.ValueOrModel | None] = getattr(
                obj,
                "get",
                _default_get,
            )
            raw_val: t.ValueOrModel | None = get_method(parameter)
            if raw_val is None:
                return raw_val
            if isinstance(raw_val, (str, int, float, bool, datetime, Path)):
                return raw_val
            if isinstance(raw_val, BaseModel):
                return raw_val
            return str(raw_val)
        # p.Base fallback via duck model_dump or attr
        base_obj: p.HasModelDump | BaseModel | p.Base = obj
        if isinstance(base_obj, BaseModel):
            return FlextUtilitiesConfiguration._resolve_from_obj(base_obj, parameter)
        if isinstance(base_obj, p.HasModelDump):
            return FlextUtilitiesConfiguration._resolve_from_obj(base_obj, parameter)
        # Final fallback: direct attr access
        found_a, attr_v = FlextUtilitiesConfiguration._try_get_attr(base_obj, parameter)
        if found_a:
            return attr_v
        class_name = str(type(obj).__name__)
        msg = f"Parameter '{parameter}' is not defined in {class_name}"
        raise e.NotFoundError(msg)

    @staticmethod
    def get_singleton(singleton_class: type, parameter: str) -> t.ValueOrModel:
        """Get parameter from a singleton configuration instance.

        Args:
            singleton_class: The singleton class (e.g., FlextSettings)
            parameter: The parameter name to retrieve

        Returns:
            The parameter value

        Raises:
            e.ValidationError: If class doesn't have get_global
            e.NotFoundError: If parameter is not defined

        """
        if hasattr(singleton_class, "get_global"):
            get_global_attr = getattr(singleton_class, "get_global", None)
            if get_global_attr is not None and callable(get_global_attr):
                instance = get_global_attr()
                found, value = (
                    FlextUtilitiesConfiguration._try_get_from_duck_model_dump(
                        instance,
                        parameter,
                    )
                )
                if found:
                    return value
                found, value = FlextUtilitiesConfiguration._try_get_attr(
                    instance,
                    parameter,
                )
                if found:
                    return value
                msg = f"Parameter '{parameter}' is not defined"
                raise e.NotFoundError(msg)
        msg = f"Class {singleton_class.__name__} does not have get_global method"
        raise e.ValidationError(msg)

    @staticmethod
    def register_factory(
        container: p.Container,
        name: str,
        factory: Callable[[], t.Scalar | t.ConfigMap | t.Dict],
        *,
        _cache: bool = False,
    ) -> r[bool]:
        """Register factory with optional caching.

        Args:
            container: Container to register in (must implement DI protocol).
            name: Factory name.
            factory: Factory function to register.
            _cache: Reserved for future implementation of cached factory pattern.

        Returns:
            r[bool]: Success(true) if registration succeeds, failure otherwise.

        """
        try:
            _ = _cache
            register_result = container.register(name, factory, kind="factory")
            if not isinstance(register_result, p.ResultLike):
                return r[bool].fail("Factory registration failed")
            if register_result.is_failure:
                return r[bool].fail(
                    register_result.error or "Factory registration failed",
                )
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Factory registration failed for {name}: {e}")

    @staticmethod
    def register_singleton(
        container: p.Container,
        name: str,
        instance: t.Scalar | t.ConfigMap | t.Dict,
    ) -> r[bool]:
        """Register singleton with standard error handling.

        Args:
            container: Container to register in (must implement DI protocol).
            name: Service name.
            instance: Service instance to register.

        Returns:
            r[bool]: Success(true) if registration succeeds, failure otherwise.

        """
        try:
            register_result = container.register(name, instance)
            if not isinstance(register_result, p.ResultLike):
                return r[bool].fail("Registration failed")
            if register_result.is_failure:
                return r[bool].fail(register_result.error or "Registration failed")
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Registration failed for {name}: {e}")

    @staticmethod
    def set_parameter(
        obj: p.HasModelDump | BaseModel | p.Base,
        parameter: str,
        value: t.Scalar | t.ConfigMap,
    ) -> bool:
        """Set parameter value on a configuration t.NormalizedValue with validation.

        Args:
            obj: The configuration t.NormalizedValue (Pydantic BaseSettings instance)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            True if successful, False if validation failed or parameter doesn't exist

        """
        try:
            obj_class = obj.__class__
            if hasattr(obj_class, "model_fields"):
                model_fields_dict = getattr(obj_class, "model_fields", {})
                if not FlextRuntime.is_dict_like(model_fields_dict):
                    return False
                if parameter not in model_fields_dict:
                    return False
            setattr(obj, parameter, value)
            return True
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return False

    @staticmethod
    def set_singleton(
        singleton_class: type,
        parameter: str,
        value: t.Scalar | t.ConfigMap,
    ) -> FlextRuntime.RuntimeResult[bool]:
        """Set parameter on a singleton configuration instance with validation.

        Args:
            singleton_class: The singleton class (e.g., FlextSettings)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            r[bool] - ok(True) on success, fail(error_msg) on failure

        """
        if not hasattr(singleton_class, "get_global"):
            return r[bool].fail(
                f"Class {singleton_class.__name__} does not have get_global method",
            )
        get_global_attr = getattr(singleton_class, "get_global", None)
        if get_global_attr is None or not callable(get_global_attr):
            return r[bool].fail(
                f"get_global is not callable on {singleton_class.__name__}",
            )
        instance = get_global_attr()
        model_dump_attr = getattr(instance, "model_dump", None)
        if model_dump_attr is None or not callable(model_dump_attr):
            return r[bool].fail("Instance does not implement model_dump() method")
        success = FlextUtilitiesConfiguration.set_parameter(instance, parameter, value)
        if success:
            return r[bool].ok(value=True)
        return r[bool].fail(
            f"Failed to set parameter '{parameter}' on {singleton_class.__name__}",
        )

    @staticmethod
    def validate_config_class(config_class: type) -> r[bool]:
        """Validate that a configuration class is properly configured.

        Args:
            config_class: Configuration class to validate (Pydantic BaseSettings)

        Returns:
            r[bool]: ok(True) if valid, fail(error_message) if invalid

        """
        try:
            class_name = getattr(config_class, "__name__", "UnknownClass")
            if not hasattr(config_class, "model_config"):
                return r[bool].fail(f"{class_name} must define model_config")
            _ = config_class()
            return r[bool].ok(True)
        except (TypeError, ValueError, AttributeError) as e:
            return r[bool].fail(f"Configuration class validation failed: {e!s}")

    @staticmethod
    def resolve_effective_log_level(
        *,
        trace: bool,
        debug: bool,
        log_level: c.LogLevel,
    ) -> c.LogLevel:
        """Resolve effective log level based on debug/trace flags.

        Pure function extracted from FlextSettings.effective_log_level computed field.

        Args:
            trace: Whether trace mode is enabled.
            debug: Whether debug mode is enabled.
            log_level: Base log level when neither flag is active.

        Returns:
            Resolved log level: DEBUG if trace, INFO if debug, else log_level.

        """
        if trace:
            return c.LogLevel.DEBUG
        if debug:
            return c.LogLevel.INFO
        return log_level

    @staticmethod
    def normalize_env_log_level() -> None:
        """Normalize FLEXT_LOG_LEVEL environment variable to uppercase.

        Ensures case-insensitive env var values are uppercased before
        Pydantic reads them (Pydantic enums are case-sensitive).
        """
        log_level = os.environ.get("FLEXT_LOG_LEVEL")
        if log_level and log_level.islower():
            os.environ["FLEXT_LOG_LEVEL"] = log_level.upper()

    @staticmethod
    def validate_database_url_scheme(url: str) -> None:
        """Validate database URL has a supported scheme.

        Args:
            url: Database URL to validate.

        Raises:
            ValueError: If URL scheme is not postgresql://, mysql://, or sqlite://.

        """
        if url and not url.startswith(("postgresql://", "mysql://", "sqlite://")):
            msg = "Invalid database URL scheme"
            raise ValueError(msg)

    @staticmethod
    def validate_trace_requires_debug(*, trace: bool, debug: bool) -> None:
        """Validate that trace mode requires debug mode.

        Args:
            trace: Whether trace mode is enabled.
            debug: Whether debug mode is enabled.

        Raises:
            ValueError: If trace is True but debug is False.

        """
        if trace and not debug:
            msg = "Trace mode requires debug mode"
            raise ValueError(msg)


__all__ = ["FlextUtilitiesConfiguration"]
