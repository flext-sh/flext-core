"""Configuration helpers for parameter access and manipulation.

Business Rules & Architecture:
============================

1. **Parameter Access Precedence** (get_parameter method):
   - HasModelDump protocol → model_dump() dict access (Pydantic models)
   - Dict-like objects → direct key access (Mapping types)
   - Objects with model_dump method → duck-typed call (compatibility)
   - Direct attribute access → final fallback (plain objects)

   This chain ensures consistent parameter retrieval across Pydantic models,
   dicts, and arbitrary objects. The sentinel tuple pattern (found: bool, value)
   distinguishes "not found" from "value is None".

2. **Singleton Pattern Integration** (get_singleton/set_singleton):
   - Expects classes with `get_global()` method (FlextSettings pattern)
   - Returns r for set operations (railway-oriented error handling)
   - Raises specific exceptions for get operations (fail-fast behavior)

3. **Pydantic v2 Configuration** (create_settings_config):
   - env_prefix: Namespace isolation (e.g., "FLEXT_LDAP_")
   - env_nested_delimiter: "__" for nested config (FLEXT_LDAP__HOST)
   - case_sensitive: False (environment variables are case-insensitive)
   - extra: "ignore" (unknown env vars don't cause errors)
   - validate_default: True (always validate even default values)

4. **Options+Config+kwargs Pattern** (build_options_from_kwargs):
   - Explicit options take precedence over kwargs
   - kwargs override individual fields on base options
   - Invalid kwargs are logged (warning) but don't fail operation
   - Pydantic validation ensures type safety on merged options

Validation Context:
- Python 3.13+: Uses collections.abc.Mapping/Sequence (not typing)
- Pydantic v2: model_dump() replaces dict(), model_fields replaces __fields__
- r: All operations that can fail return r[T]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from flext_core import FlextRuntime, T_Model, c, e, p, r, t


class FlextUtilitiesConfiguration:
    """Configuration utilities for parameter access and manipulation.

    Business Rules:
    ==============
    This class provides a unified interface for accessing and manipulating
    configuration parameters across different object types in the FLEXT ecosystem.

    1. **Type Coercion Strategy**:
       - NO automatic type coercion (Pydantic handles validation)
       - Values are returned as-is from source objects
       - Type validation happens at Pydantic model boundaries

    2. **Error Handling Strategy**:
       - get_parameter: Raises NotFoundError (fail-fast for missing config)
       - set_parameter: Returns bool (graceful handling for write failures)
       - get_singleton/set_singleton: Raises/Returns based on operation type

    3. **Thread Safety**:
       - Read operations are thread-safe (no shared state mutation)
       - Write operations assume external synchronization when needed
       - Singleton access uses class-level get_global pattern

    4. **Protocol-Based Dispatch**:
       - HasModelDump: Pydantic models with model_dump() method
       - HasModelFields: Pydantic models with model_fields class attribute
       - Duck typing fallback for model_dump method on non-protocol objects
    """

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
        sentinel = object()
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

        Business Rule: Log Level Resolution
        ===================================
        This method resolves the default log level from constants to avoid circular
        imports between configuration and logging modules. It provides a safe way
        to get the default log level without importing the full settings hierarchy.

        Process:
        1. Get default log level name from constants (e.g., "INFO")
        2. Convert string to actual logging level constant
        3. Return numeric logging level or fallback to INFO

        Returns:
            int: Numeric logging level (e.g., logging.INFO = 20)

        """
        default_log_level = c.Logging.DEFAULT_LEVEL.upper()
        return getattr(logging, default_log_level, logging.INFO)

    @staticmethod
    def resolve_env_file() -> str:
        """Resolve .env file path from FLEXT_ENV_FILE environment variable.

        Business Rule: Environment File Resolution
        ==========================================
        All FLEXT ecosystem configurations share the same .env file resolution
        logic to ensure consistent behavior across all namespace configurations.

        Precedence Chain (highest to lowest):
        1. FLEXT_ENV_FILE environment variable → custom path (user override)
        2. .env file in current working directory → standard location
        3. Default ".env" string → Pydantic handles gracefully if missing

        Implications:
        - Custom path is ALWAYS used if FLEXT_ENV_FILE is set (even if invalid)
        - Invalid custom paths are returned as-is (Pydantic handles gracefully)
        - Current directory .env takes precedence over hardcoded defaults
        - Returns string (not Path) for Pydantic SettingsConfigDict compatibility

        Python 3.13+ / Pydantic v2 Context:
        - Path.resolve() returns absolute Path (converted to str)
        - Pydantic v2 SettingsConfigDict accepts str | None for env_file
        - Pydantic ignores missing env_file gracefully (no error raised)

        Returns:
            str: Path to .env file (custom, discovered, or default ".env")

        Example:
            # In namespace config classes (e.g., FlextLdapSettings)
            model_config = SettingsConfigDict(
                env_prefix="FLEXT_LDAP_",
                env_file=u.resolve_env_file(),
                ...
            )

        """
        custom_env_file = os.environ.get(c.Platform.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        default_path = Path.cwd() / c.Platform.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return c.Platform.ENV_FILE_DEFAULT

    _NOT_FOUND: tuple[bool, None] = (False, None)

    @staticmethod
    def _try_get_attr(
        obj: p.HasModelDump | BaseModel | p.Base, parameter: str
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get attribute value from object via direct attribute access.

        Business Rule: Direct Attribute Access (Fallback Strategy)
        ==========================================================
        This is the FINAL fallback in the parameter access chain. Used when:
        - Object is not a Pydantic model (no model_dump)
        - Object is not dict-like (no __getitem__)
        - Object has simple attributes (plain Python classes)

        Type Safety:
        - Uses direct attribute access with AttributeError handling
        - Cast to object preserves union type safety
        - Returns sentinel tuple to distinguish "not found" from "None value"

        Args:
            obj: Object instance that might have the parameter as attribute
            parameter: Attribute name to retrieve

        Returns:
            (True, value) if attribute exists, (False, None) if not

        """
        obj_vars: dict[str, t.NormalizedValue] = (
            vars(obj) if hasattr(obj, "__dict__") else {}
        )
        if parameter not in obj_vars:
            return FlextUtilitiesConfiguration._NOT_FOUND
        value = obj_vars[parameter]
        return (True, value)

    @staticmethod
    def _try_get_from_dict_like(
        obj: Mapping[str, t.ValueOrModel], parameter: str
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get parameter from dict-like object.

        Business Rule: Dict-Like Access (Secondary Strategy)
        ===================================================
        This strategy handles objects implementing Mapping protocol:
        - dict instances
        - MappingProxyType (frozen dicts)
        - Custom Mapping implementations

        FlextRuntime.is_dict_like() Check:
        - Uses collections.abc.Mapping for type check
        - Ensures 'in' operator and __getitem__ are available
        - Returns False for sequences (list/tuple) even though they support []

        Type Safety:
        - obj[parameter] returns the exact stored type
        - No type coercion (preserves None vs missing distinction)

        Args:
            obj: Potentially dict-like object
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
        obj: p.HasModelDump, parameter: str
    ) -> tuple[bool, t.ValueOrModel]:
        """Try to get parameter from HasModelDump protocol object.

        Business Rule: Pydantic Model Access (Primary Strategy)
        ======================================================
        This is the FIRST strategy in the parameter access chain for
        objects implementing the HasModelDump protocol (Pydantic models).

        Why model_dump() over direct attribute access?
        - Includes computed fields and validators applied
        - Handles field aliases (alias vs field name)
        - Consistent with Pydantic serialization semantics
        - Works with both BaseModel and BaseSettings

        Error Handling:
        - AttributeError: model_dump() not properly implemented
        - TypeError: Invalid argument to model_dump()
        - ValueError: Validation error during dump

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
        '''Build Pydantic options model from explicit options or kwargs.

        Business Rule: Options+Config+kwargs Pattern (FLEXT Convention)
        ===============================================================
        This is a core pattern used throughout the FLEXT ecosystem for
        flexible configuration with type safety:

        Priority Chain:
        1. explicit_options (if provided) → Use as base
        2. default_factory() → Get defaults from config singleton
        3. kwargs → Override individual fields on base

        Pattern Rationale:
        - Pydantic models ensure type safety and validation
        - Config singleton provides consistent defaults
        - **kwargs enables convenient API for simple cases
        - explicit_options enables complex/reusable configuration

        Invalid Kwargs Handling:
        - Invalid kwargs are logged as WARNING (not error)
        - Operation continues with valid kwargs only
        - This is intentional: typos shouldn't crash the application
        - Callers can check logs for debugging

        Type Variable T_Model:
        - Bound to Pydantic BaseModel subclass
        - Ensures model_dump() and model_fields are available
        - Enables type inference for return value

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
        **format_kwargs: t.Scalar,
            ) -> "FlextRuntime.RuntimeResult[str]":
                # Get ldif config using get_namespace_config (no __getattr__)
                def get_ldif_config_default() -> WriteFormatOptions:
                    """Get default options from ldif config namespace."""
                    config_class = self.config.get_namespace_config("ldif")
                    if config_class is None:
                        msg = "ldif namespace not registered in config"
                        raise ValueError(msg)
                    ldif_config = config_class()
                    # Use getattr for known method - config classes have to_write_options()
                    to_write_options = getattr(ldif_config, "to_write_options", None)
                    if to_write_options is None:
                        msg = "ldif config does not have to_write_options() method"
                        raise AttributeError(msg)
                    return to_write_options()

                options_result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                    model_class=WriteFormatOptions,
                    explicit_options=format_options,
                    default_factory=get_ldif_config_default,
                    **format_kwargs,
                )
                if options_result.is_failure:
                    return r[T_Model].fail(options_result.error or "Failed to get options")
                # Use .value directly - r never returns None on success
                options = options_result.value

        Args:
            model_class: The Pydantic model class (e.g., WriteFormatOptions)
            explicit_options: Explicitly provided options instance, or None
            default_factory: Callable that returns default options from config
            **kwargs: Individual option overrides (snake_case field names)

        Returns:
            r[T_Model]: ok(validated_model) or fail(error_msg)

        '''
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
            invalid_kwargs: list[str] = []
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
                "Unexpected error building options model"
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
                        f"Bulk registration failed at {name}: register returned non-result"
                    )
                if register_result.is_failure:
                    return r[int].fail(
                        f"Bulk registration failed at {name}: {register_result.error}"
                    )
                count += 1
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[int].fail(f"Bulk registration failed at {name}: {e}")
        return r[int].ok(count)

    @staticmethod
    def create_settings_config(
        env_prefix: str,
        env_file: str = c.Platform.ENV_FILE_DEFAULT,
        env_nested_delimiter: str = "__",
    ) -> Mapping[str, t.Scalar]:
        """Create a SettingsConfigDict for environment binding.

        Business Rule: Pydantic v2 Environment Binding Configuration
        ============================================================
        This method creates a standardized configuration dictionary for
        Pydantic v2 BaseSettings classes in the FLEXT ecosystem.

        Configuration Options Explained:
        - env_prefix: Namespace isolation (FLEXT_LDAP_, FLEXT_API_, etc.)
          Prevents conflicts between different FLEXT libraries
        - env_file: Path to .env file (use resolve_env_file() for standard resolution)
        - env_nested_delimiter: "__" enables FLEXT_DB__HOST → config.db.host mapping
        - case_sensitive: False (ENV_VAR, env_var, Env_Var all work)
        - extra: "ignore" (unknown env vars don't cause ValidationError)
        - validate_default: True (validates default values at class definition)

        FLEXT Ecosystem Convention:
        - All FLEXT libraries use env_prefix pattern: "FLEXT_{LIBRARY}_"
        - Example: FLEXT_LDAP_, FLEXT_API_, FLEXT_CLI_
        - This ensures namespace isolation and consistent configuration

        Why dict instead of SettingsConfigDict?
        - Returns dict for flexibility and type compatibility
        - Caller can cast to SettingsConfigDict if needed
        - Avoids importing pydantic_settings in this module

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
            "extra": c.ModelConfig.EXTRA_IGNORE,
            "validate_default": True,
        }

    @staticmethod
    def get_parameter(
        obj: p.HasModelDump | BaseModel | p.Base | Mapping[str, t.ValueOrModel],
        parameter: str,
    ) -> t.ValueOrModel:
        """Get parameter value from a configuration object.

        Business Rule: Parameter Access Precedence Chain
        ================================================
        This method implements a deterministic precedence chain for parameter
        retrieval that handles diverse object types consistently:

        1. HasModelDump protocol → model_dump() dict access
           - Highest priority for Pydantic models
           - Ensures computed fields and validation are included

        2. Dict-like objects → direct key access
           - For Mapping implementations (dict, MappingProxyType)
           - Efficient O(1) key lookup

        3. Objects with model_dump method → duck-typed call
           - Compatibility for third-party Pydantic-like objects
           - Fallback when protocol check fails

        4. Direct attribute access → final fallback
           - Plain Python objects with attributes
           - Uses hasattr/getattr pattern

        Fail-Fast vs Graceful Handling:
        - This method uses FAIL-FAST semantics (raises NotFoundError)
        - Rationale: Missing configuration is a programming error
        - Callers should ensure parameters exist or catch the exception

        None Value Handling:
        - None is a VALID configuration value and is returned correctly
        - Only raises when parameter doesn't exist at all
        - Sentinel tuple pattern in helpers distinguishes "None value" from "not found"

        Args:
            obj: Configuration object (HasModelDump, dict-like, or with attributes)
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
                obj, parameter
            )
            if found:
                return value
        found_d, duck_v = FlextUtilitiesConfiguration._try_get_from_duck_model_dump(
            obj, parameter
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
                obj, "get", _default_get
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

        Business Rule: Singleton Configuration Access (FLEXT Pattern)
        ============================================================
        The FLEXT ecosystem uses a singleton pattern for global configuration
        via `get_global()` class method. This enables:

        - Consistent configuration across all services
        - Lazy initialization (instance created on first access)
        - Thread-safe singleton access (handled by FlextSettings implementation)

        Expected Interface:
        - singleton_class.get_global() → Returns singleton instance
        - Instance must implement HasModelDump protocol
        - Parameters accessed via get_parameter (precedence chain applies)

        Fail-Fast Semantics:
        - Raises ValidationError if class lacks get_global
        - Raises NotFoundError if parameter not found (from get_parameter)
        - This is intentional: missing config is a programming error

        Type Safety:
        - Type narrowing ensures HasModelDump protocol before access
        - Explicit local variable for type checker compatibility

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
                        instance, parameter
                    )
                )
                if found:
                    return value
                found, value = FlextUtilitiesConfiguration._try_get_attr(
                    instance, parameter
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
                    register_result.error or "Factory registration failed"
                )
            return r[bool].ok(value=True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[bool].fail(f"Factory registration failed for {name}: {e}")

    @staticmethod
    def register_singleton(
        container: p.Container, name: str, instance: t.Scalar | t.ConfigMap | t.Dict
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
        """Set parameter value on a configuration object with validation.

        Business Rule: Graceful Write with Pydantic Validation
        =====================================================
        This method uses GRACEFUL semantics (returns bool) unlike get_parameter
        which uses fail-fast. Rationale:

        - Write failures are often recoverable (use default, retry, etc.)
        - Pydantic validation errors should not crash the application
        - Callers can check return value and handle appropriately

        Pydantic v2.11+ Compatibility:
        - model_fields is a CLASS attribute, not instance attribute
        - Uses getattr(obj.__class__, "model_fields", {}) for correct access
        - This avoids deprecation warnings in newer Pydantic versions

        Validation Flow:
        1. Check if object implements HasModelFields protocol
        2. Verify parameter exists in model_fields (prevents adding new fields)
        3. Use setattr which triggers Pydantic's validate_assignment
        4. Pydantic validates the value against field type

        Error Handling:
        - AttributeError: Object doesn't support attribute assignment
        - TypeError: Value type incompatible with field type
        - ValueError: Pydantic validation failure
        - RuntimeError: Model frozen/immutable
        - KeyError: Field not found (shouldn't happen after model_fields check)

        Args:
            obj: The configuration object (Pydantic BaseSettings instance)
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
        singleton_class: type, parameter: str, value: t.Scalar | t.ConfigMap
    ) -> FlextRuntime.RuntimeResult[bool]:
        """Set parameter on a singleton configuration instance with validation.

        Business Rule: Railway-Oriented Singleton Mutation
        =================================================
        Unlike get_singleton (fail-fast), this method uses r for
        graceful error handling. Rationale:

        - Configuration mutation is often optional (fallback to defaults)
        - Runtime errors shouldn't crash the application
        - Callers can decide how to handle failures

        Validation Chain:
        1. Check get_global method exists (r.fail if not)
        2. Check method is callable (r.fail if not)
        3. Check instance implements HasModelDump (r.fail if not)
        4. Delegate to set_parameter for actual mutation
        5. set_parameter returns bool, converted to r

        Thread Safety:
        - Singleton access is thread-safe (FlextSettings guarantees this)
        - Individual parameter mutation is NOT atomic
        - External synchronization needed for concurrent writes

        Args:
            singleton_class: The singleton class (e.g., FlextSettings)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            r[bool] - ok(True) on success, fail(error_msg) on failure

        """
        if not hasattr(singleton_class, "get_global"):
            return r[bool].fail(
                f"Class {singleton_class.__name__} does not have get_global method"
            )
        get_global_attr = getattr(singleton_class, "get_global", None)
        if get_global_attr is None or not callable(get_global_attr):
            return r[bool].fail(
                f"get_global is not callable on {singleton_class.__name__}"
            )
        instance = get_global_attr()
        model_dump_attr = getattr(instance, "model_dump", None)
        if model_dump_attr is None or not callable(model_dump_attr):
            return r[bool].fail("Instance does not implement model_dump() method")
        success = FlextUtilitiesConfiguration.set_parameter(instance, parameter, value)
        if success:
            return r[bool].ok(value=True)
        return r[bool].fail(
            f"Failed to set parameter '{parameter}' on {singleton_class.__name__}"
        )

    @staticmethod
    def validate_config_class(config_class: type) -> r[bool]:
        """Validate that a configuration class is properly configured.

        Business Rule: Pydantic v2 Configuration Class Validation
        =========================================================
        This method validates that a class follows FLEXT ecosystem patterns
        for Pydantic v2 BaseSettings configuration classes.

        Required Attributes:
        - model_config: Dict or SettingsConfigDict with env binding configuration
          This is MANDATORY for all FLEXT configuration classes

        Validation Steps:
        1. Type check implicit (config_class: type in signature)
        2. Check model_config attribute exists
        3. Attempt instantiation to verify default values work

        Why Instantiation Test?
        - Catches missing required fields
        - Catches invalid default values
        - Catches Pydantic validation errors early
        - Prevents runtime failures in production

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
        log_level: c.Settings.LogLevel,
    ) -> c.Settings.LogLevel:
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
            return c.Settings.LogLevel.DEBUG
        if debug:
            return c.Settings.LogLevel.INFO
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
