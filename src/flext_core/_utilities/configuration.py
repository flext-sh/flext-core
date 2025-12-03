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
   - Expects classes with `get_global_instance()` method (FlextConfig pattern)
   - Returns FlextResult for set operations (railway-oriented error handling)
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
- FlextResult: All operations that can fail return r[T]

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.exceptions import e
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import T_Model, t


class FlextConfiguration:
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
       - Singleton access uses class-level get_global_instance pattern

    4. **Protocol-Based Dispatch**:
       - HasModelDump: Pydantic models with model_dump() method
       - HasModelFields: Pydantic models with model_fields class attribute
       - Duck typing fallback for model_dump method on non-protocol objects
    """

    @staticmethod
    def _get_logger() -> p.StructlogLogger:
        """Get logger instance using FlextRuntime.

        Business Rule: Logger access through FlextRuntime avoids circular
        imports between configuration and logging modules.

        Returns:
            Structlog logger instance with all standard logging methods.

        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def resolve_env_file() -> str | None:
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
            str | None: Path to .env file or None if not found

        Example:
            # In namespace config classes (e.g., FlextLdapConfig)
            model_config = SettingsConfigDict(
                env_prefix="FLEXT_LDAP_",
                env_file=u.Configuration.resolve_env_file(),
                ...
            )

        """
        # Check for custom env file path
        custom_env_file = os.environ.get(FlextConstants.Platform.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            # If custom path doesn't exist, return it anyway (Pydantic will handle gracefully)
            return custom_env_file

        # Default: use .env from current directory
        default_path = Path.cwd() / FlextConstants.Platform.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())

        # Fallback: use default value (Pydantic handles missing file gracefully)
        return FlextConstants.Platform.ENV_FILE_DEFAULT

    # =========================================================================
    # Sentinel Pattern for Parameter Access
    # =========================================================================
    # Business Rule: Distinguish "not found" from "value is None"
    #
    # Problem: When get_parameter returns None, callers can't distinguish:
    # - Parameter exists with None value (valid configuration)
    # - Parameter doesn't exist at all (missing configuration)
    #
    # Solution: Sentinel tuple pattern (found: bool, value)
    # - (True, value) → Parameter found, value may be None or any type
    # - (False, None) → Parameter not found in this access strategy
    #
    # This pattern is internal (private helper methods) and converted to
    # either return value or NotFoundError at the public API boundary.
    # =========================================================================

    _NOT_FOUND: tuple[bool, None] = (False, None)

    @staticmethod
    def _try_get_attr(
        obj: t.GeneralValueType | p.HasModelDump,
        parameter: str,
    ) -> tuple[bool, t.GeneralValueType | None]:
        """Try to get attribute value from object via hasattr/getattr.

        Business Rule: Direct Attribute Access (Fallback Strategy)
        ==========================================================
        This is the FINAL fallback in the parameter access chain. Used when:
        - Object is not a Pydantic model (no model_dump)
        - Object is not dict-like (no __getitem__)
        - Object has simple attributes (plain Python classes)

        Type Safety:
        - Uses hasattr() before getattr() to avoid AttributeError
        - Cast to GeneralValueType preserves union type safety
        - Returns sentinel tuple to distinguish "not found" from "None value"

        Args:
            obj: Any object that might have the parameter as attribute
            parameter: Attribute name to retrieve

        Returns:
            (True, value) if attribute exists, (False, None) if not

        """
        if hasattr(obj, parameter):
            return (True, cast("t.GeneralValueType", getattr(obj, parameter)))
        return FlextConfiguration._NOT_FOUND

    @staticmethod
    def _try_get_from_model_dump(
        obj: p.HasModelDump,
        parameter: str,
    ) -> tuple[bool, t.GeneralValueType | None]:
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
            if isinstance(obj_dict, dict) and parameter in obj_dict:
                return (True, cast("t.GeneralValueType", obj_dict[parameter]))
        except (AttributeError, TypeError, ValueError):
            pass
        return FlextConfiguration._NOT_FOUND

    @staticmethod
    def _try_get_from_dict_like(
        obj: t.GeneralValueType,
        parameter: str,
    ) -> tuple[bool, t.GeneralValueType | None]:
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
        if FlextRuntime.is_dict_like(obj) and parameter in obj:
            return (True, obj[parameter])
        return FlextConfiguration._NOT_FOUND

    @staticmethod
    def _try_get_from_duck_model_dump(
        obj: t.GeneralValueType | p.HasModelDump,
        parameter: str,
    ) -> tuple[bool, t.GeneralValueType | None]:
        """Try to get parameter via duck-typed model_dump method.

        Business Rule: Duck-Typed Pydantic Access (Tertiary Strategy)
        =============================================================
        This strategy handles objects that have model_dump() method but don't
        formally implement the HasModelDump protocol. Use cases:
        - Third-party Pydantic models not in our type system
        - Proxy objects wrapping Pydantic models
        - Objects mimicking Pydantic interface

        Why Duck Typing?
        - isinstance() check for HasModelDump might miss some valid objects
        - Dynamic nature of Python allows model_dump without protocol
        - Backwards compatibility with pre-protocol code

        Safety Checks:
        - getattr with None default (no AttributeError)
        - callable() check prevents calling non-callables
        - Exception handling for any runtime errors

        Args:
            obj: Object that might have model_dump() method
            parameter: Field name to retrieve

        Returns:
            (True, value) if found via model_dump(), (False, None) otherwise

        """
        model_dump_fn = getattr(obj, "model_dump", None)
        if model_dump_fn is None or not callable(model_dump_fn):
            return FlextConfiguration._NOT_FOUND
        try:
            model_data = model_dump_fn()
            if isinstance(model_data, dict) and parameter in model_data:
                return (
                    True,
                    cast("t.GeneralValueType", model_data[parameter]),
                )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        return FlextConfiguration._NOT_FOUND

    @staticmethod
    def get_parameter(
        obj: t.GeneralValueType | p.HasModelDump,
        parameter: str,
    ) -> t.GeneralValueType:
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
        # Strategy 1: HasModelDump protocol
        if isinstance(obj, p.HasModelDump):
            found, value = FlextConfiguration._try_get_from_model_dump(obj, parameter)
            if found:
                # Type narrowing: when found is True, value is GeneralValueType (not None)
                return value

        # Strategy 2: Dict-like GeneralValueType
        if isinstance(obj, (str, int, float, bool, type(None), Sequence, Mapping)):
            obj_general = cast("t.GeneralValueType", obj)
            found, value = FlextConfiguration._try_get_from_dict_like(
                obj_general, parameter
            )
            if found:
                # Type narrowing: when found is True, value is GeneralValueType (not None)
                return value

        # Strategy 3: Object with model_dump method (duck typing)
        found, value = FlextConfiguration._try_get_from_duck_model_dump(obj, parameter)
        if found:
            # Type narrowing: when found is True, value is GeneralValueType (not None)
            return value

        # Strategy 4: Direct attribute access (final fallback)
        found, attr_val = FlextConfiguration._try_get_attr(obj, parameter)
        if found:
            # Type narrowing: when found is True, attr_val is GeneralValueType (not None)
            return attr_val

        msg = f"Parameter '{parameter}' is not defined in {obj.__class__.__name__}"
        raise e.NotFoundError(msg)

    @staticmethod
    def set_parameter(
        obj: t.GeneralValueType | p.HasModelDump,
        parameter: str,
        value: t.GeneralValueType,
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
        - Uses getattr(type(obj), "model_fields", {}) for correct access
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
            # Check if parameter exists in model fields for Pydantic objects
            if isinstance(obj, p.HasModelFields):
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
    ) -> t.GeneralValueType:
        """Get parameter from a singleton configuration instance.

        Business Rule: Singleton Configuration Access (FLEXT Pattern)
        ============================================================
        The FLEXT ecosystem uses a singleton pattern for global configuration
        via `get_global_instance()` class method. This enables:

        - Consistent configuration across all services
        - Lazy initialization (instance created on first access)
        - Thread-safe singleton access (handled by FlextConfig implementation)

        Expected Interface:
        - singleton_class.get_global_instance() → Returns singleton instance
        - Instance must implement HasModelDump protocol
        - Parameters accessed via get_parameter (precedence chain applies)

        Fail-Fast Semantics:
        - Raises ValidationError if class lacks get_global_instance
        - Raises NotFoundError if parameter not found (from get_parameter)
        - This is intentional: missing config is a programming error

        Type Safety:
        - Type narrowing ensures HasModelDump protocol before access
        - Explicit local variable for type checker compatibility

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to retrieve

        Returns:
            The parameter value

        Raises:
            e.ValidationError: If class doesn't have get_global_instance
            e.NotFoundError: If parameter is not defined

        """
        if hasattr(singleton_class, "get_global_instance"):
            get_global_instance_method = singleton_class.get_global_instance
            if callable(get_global_instance_method):
                instance = get_global_instance_method()
                if isinstance(instance, p.HasModelDump):
                    # Type narrowing: instance is HasModelDump
                    has_model_dump_instance: p.HasModelDump = instance
                    return FlextConfiguration.get_parameter(
                        has_model_dump_instance,
                        parameter,
                    )

        msg = (
            f"Class {singleton_class.__name__} does not have get_global_instance method"
        )
        raise e.ValidationError(msg)

    @staticmethod
    def set_singleton(
        singleton_class: type,
        parameter: str,
        value: t.GeneralValueType,
    ) -> r[bool]:
        """Set parameter on a singleton configuration instance with validation.

        Business Rule: Railway-Oriented Singleton Mutation
        =================================================
        Unlike get_singleton (fail-fast), this method uses FlextResult for
        graceful error handling. Rationale:

        - Configuration mutation is often optional (fallback to defaults)
        - Runtime errors shouldn't crash the application
        - Callers can decide how to handle failures

        Validation Chain:
        1. Check get_global_instance method exists (FlextResult.fail if not)
        2. Check method is callable (FlextResult.fail if not)
        3. Check instance implements HasModelDump (FlextResult.fail if not)
        4. Delegate to set_parameter for actual mutation
        5. set_parameter returns bool, converted to FlextResult

        Thread Safety:
        - Singleton access is thread-safe (FlextConfig guarantees this)
        - Individual parameter mutation is NOT atomic
        - External synchronization needed for concurrent writes

        Args:
            singleton_class: The singleton class (e.g., FlextConfig)
            parameter: The parameter name to set
            value: The new value to set (will be validated by Pydantic)

        Returns:
            r[bool] - ok(True) on success, fail(error_msg) on failure

        """
        if not hasattr(singleton_class, "get_global_instance"):
            return r[bool].fail(
                f"Class {singleton_class.__name__} does not have get_global_instance method",
            )

        get_global_instance_method = singleton_class.get_global_instance
        if not callable(get_global_instance_method):
            return r[bool].fail(
                f"get_global_instance is not callable on {singleton_class.__name__}",
            )

        instance = get_global_instance_method()
        if not isinstance(instance, p.HasModelDump):
            return r[bool].fail(
                "Instance does not implement HasModelDump protocol",
            )

        # Type narrowing: instance is HasModelDump
        has_model_dump_instance: p.HasModelDump = instance
        success = FlextConfiguration.set_parameter(
            has_model_dump_instance,
            parameter,
            value,
        )
        if success:
            return r[bool].ok(True)
        return r[bool].fail(
            f"Failed to set parameter '{parameter}' on {singleton_class.__name__}",
        )

    @staticmethod
    def validate_config_class(config_class: type[object]) -> tuple[bool, str | None]:
        """Validate that a configuration class is properly configured.

        Business Rule: Pydantic v2 Configuration Class Validation
        =========================================================
        This method validates that a class follows FLEXT ecosystem patterns
        for Pydantic v2 BaseSettings configuration classes.

        Required Attributes:
        - model_config: Dict or SettingsConfigDict with env binding configuration
          This is MANDATORY for all FLEXT configuration classes

        Validation Steps:
        1. Type check implicit (config_class: type[object] in signature)
        2. Check model_config attribute exists
        3. Attempt instantiation to verify default values work

        Why Instantiation Test?
        - Catches missing required fields
        - Catches invalid default values
        - Catches Pydantic validation errors early
        - Prevents runtime failures in production

        Return Pattern:
        - Returns tuple (bool, str | None) instead of FlextResult
        - This is intentional for simple yes/no validation
        - Error message provides context for debugging

        Args:
            config_class: Configuration class to validate (Pydantic BaseSettings)

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if valid
                - (False, error_message) if invalid

        """
        try:
            # config_class: type[object] already guarantees it's a type
            # No need to check isinstance(config_class, type)

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
    ) -> dict[str, t.GeneralValueType]:
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
            "extra": "ignore",
            "validate_default": True,
        }

    @staticmethod
    def build_options_from_kwargs(
        model_class: type[T_Model],
        explicit_options: T_Model | None,
        default_factory: Callable[[], T_Model],
        **kwargs: t.GeneralValueType,
    ) -> r[T_Model]:
        """Build Pydantic options model from explicit options or kwargs.

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
                **format_kwargs: t.GeneralValueType,
            ) -> r[str]:
                options_result = u.Configuration.build_options_from_kwargs(
                    model_class=WriteFormatOptions,
                    explicit_options=format_options,
                    default_factory=lambda: self.config.ldif.to_write_options(),
                    **format_kwargs,
                )
                if options_result.is_failure:
                    return r[str].fail(options_result.error)
                options = options_result.unwrap()
                # ... use options

        Args:
            model_class: The Pydantic model class (e.g., WriteFormatOptions)
            explicit_options: Explicitly provided options instance, or None
            default_factory: Callable that returns default options from config
            **kwargs: Individual option overrides (snake_case field names)

        Returns:
            r[T_Model]: ok(validated_model) or fail(error_msg)

        """
        try:
            # Step 1: Get base options (explicit or from config defaults)
            if explicit_options is not None:
                base_options = explicit_options
            else:
                base_options = default_factory()

            # Step 2: If no kwargs, return base options directly
            if not kwargs:
                return r[T_Model].ok(base_options)

            # Step 3: Get valid field names from model class
            # Access model_fields as class attribute for type safety
            model_fields_attr = getattr(model_class, "model_fields", {})
            model_fields: dict[str, t.GeneralValueType] = (
                model_fields_attr if isinstance(model_fields_attr, dict) else {}
            )
            valid_field_names = set(model_fields.keys())

            # Step 4: Filter kwargs to only valid field names
            valid_kwargs: dict[str, t.GeneralValueType] = {}
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
                FlextConfiguration._get_logger().warning(
                    "Ignored invalid kwargs for %s: %s. Valid fields: %s",
                    class_name,
                    invalid_kwargs,
                    sorted(valid_field_names),
                )

            # Step 6: If no valid overrides, return base options
            if not valid_kwargs:
                return r[T_Model].ok(base_options)

            # Step 7: Create new model with base values + kwargs overrides
            # Use model_dump to get base values, then update with kwargs
            # NOTE: Cannot use u.merge() here due to circular import
            # (utilities.py imports from _utilities/configuration.py)
            base_dict = base_options.model_dump()
            base_dict.update(valid_kwargs)

            # Step 8: Validate and create new model instance
            merged_options = model_class(**base_dict)

            return r[T_Model].ok(merged_options)

        except (TypeError, ValueError) as e:
            # Pydantic validation error
            class_name = getattr(model_class, "__name__", "UnknownModel")
            return r[T_Model].fail(
                f"Failed to build {class_name}: {e}",
            )
        except Exception as e:
            # Unexpected error
            class_name = getattr(model_class, "__name__", "UnknownModel")
            FlextConfiguration._get_logger().exception(
                "Unexpected error building options model",
            )
            return r[T_Model].fail(
                f"Unexpected error building {class_name}: {e}",
            )


uConfiguration = FlextConfiguration  # noqa: N816

__all__ = [
    "FlextConfiguration",
    "uConfiguration",
]
