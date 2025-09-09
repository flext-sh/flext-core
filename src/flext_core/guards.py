"""Type guards and data integrity enforcement.

Provides FlextGuards with type guards, validation decorators, and
assert-style validation using FlextResult integration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Generic, TypeVar, cast

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Type variables for guards system
R = TypeVar("R")  # Return type variable
T = TypeVar("T")  # Generic type variable


class FlextGuards:
    """Validation and guard system with pure functions, immutability, and type guards."""

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class PureWrapper(Generic[R]):
        """Wrapper class for pure function enforcement with memoization."""

        def __init__(self, func: Callable[[object], R] | Callable[[], R]) -> None:
            """Initialize pure function wrapper."""
            self.func = func
            self.cache: dict[object, R] = {}
            self.__pure__ = True
            # Copy function metadata safely
            if hasattr(func, "__name__"):
                self.__name__ = func.__name__
            if hasattr(func, "__doc__"):
                self.__doc__ = func.__doc__

        def __call__(self, *args: object, **kwargs: object) -> R:
            """Invoke wrapped function with memoization caching.

            Returns:
                R: The result of the wrapped pure function, possibly cached.

            """
            try:
                cache_key = (args, tuple(sorted(kwargs.items())))
                if cache_key in self.cache:
                    return self.cache[cache_key]
                result = self.func(*args, **kwargs)
                self.cache[cache_key] = result
                return result
            except TypeError:
                return self.func(*args, **kwargs)

        def __cache_size__(self) -> int:
            """Return current memoization cache size.

            Returns:
                int: Number of cached call results.

            """
            return len(self.cache)

        def __get__(self, instance: object, owner: type | None = None) -> object:
            """Descriptor protocol for method binding support.

            Returns:
                object: A bound callable that preserves purity metadata.

            """
            if instance is None:
                return self

            # Create a bound method-like callable that preserves __pure__ attribute
            def bound_method(*args: object, **kwargs: object) -> R:
                return self(instance, *args, **kwargs)

            # Safely add the __pure__ attribute to the function using setattr
            with contextlib.suppress(AttributeError, TypeError):
                setattr(bound_method, "__pure__", True)
            return bound_method

    class ValidationUtils:
        """Assertion-style validation utilities with structured error handling."""

        @staticmethod
        def require_not_none(
            value: object,
            message: str = "Value cannot be None",
        ) -> object:
            """Validate that a value is not None.

            Returns:
                object: The original value when it is not None.

            Raises:
                FlextExceptions.ValidationError: If the value is None.

            """
            if value is None:
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_positive(
            value: object,
            message: str = "Value must be positive",
        ) -> object:
            """Validate that a value is a positive integer.

            Returns:
                object: The original value when it is a positive integer.

            Raises:
                FlextExceptions.ValidationError: If the value is not a positive int.

            """
            if not (isinstance(value, int) and value > 0):
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_in_range(
            value: object,
            min_val: int,
            max_val: int,
            message: str | None = None,
        ) -> object:
            """Validate that a numeric value falls within inclusive bounds.

            Returns:
                object: The original value when it is within the bounds.

            Raises:
                FlextExceptions.ValidationError: If the value is out of bounds or not numeric.

            """
            if not (isinstance(value, (int, float)) and min_val <= value <= max_val):
                if not message:
                    message = f"Value must be between {min_val} and {max_val}"
                raise FlextExceptions.ValidationError(message)
            return value

        @staticmethod
        def require_non_empty(
            value: object,
            message: str = "Value cannot be empty",
        ) -> object:
            """Validate that a string value is non-empty.

            Returns:
                object: The original value when it is a non-empty string.

            Raises:
                FlextExceptions.ValidationError: If the value is empty or not a string.

            """
            if not isinstance(value, str) or not value.strip():
                raise FlextExceptions.ValidationError(message)
            return value

    # ==========================================================================
    # MAIN GUARD FUNCTIONALITY
    # ==========================================================================

    @staticmethod
    def is_dict_of(obj: object, value_type: type) -> bool:
        """Type guard to validate dictionary with homogeneous value types.

        Returns:
            bool: True when all values match the given type.

        """
        if not isinstance(obj, dict):
            return False
        # After isinstance check, obj is narrowed to dict type
        dict_obj = cast("dict[object, object]", obj)
        return all(isinstance(value, value_type) for value in dict_obj.values())

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Type guard to validate list with homogeneous item types.

        Returns:
            bool: True when all items match the given type.

        """
        if not isinstance(obj, list):
            return False
        # After isinstance check, obj is narrowed to list type
        list_obj = cast("FlextTypes.Core.List", obj)
        return all(isinstance(item, item_type) for item in list_obj)

    @staticmethod
    def immutable(target_class: type) -> type:
        """Transform class into immutable with frozen state and validation.

        Returns:
            type: A new immutable subclass wrapper around the target class.

        """

        def _init(self: object, *args: object, **kwargs: object) -> None:
            # Call original class initialization first (safely)
            init_method = getattr(target_class, "__init__", None)
            with contextlib.suppress(Exception):
                if init_method is not None:
                    init_method(self, *args, **kwargs)
            # Mark as initialized to prevent further modifications
            object.__setattr__(self, "_initialized", True)

        def _setattr(self: object, name: str, value: object) -> None:
            if hasattr(self, "_initialized"):
                msg = "Cannot modify immutable object attribute '" + name + "'"
                raise AttributeError(msg)
            object.__setattr__(self, name, value)

        def _hash(self: object) -> int:
            try:
                attrs = tuple(
                    getattr(self, attr)
                    for attr in dir(self)
                    if not attr.startswith("_") and not callable(getattr(self, attr))
                )
                return hash((self.__class__.__name__, attrs))
            except TypeError:
                return hash(id(self))

        # Create wrapper class
        return type(
            target_class.__name__,
            (target_class,),
            {
                "__init__": _init,
                "__setattr__": _setattr,
                "__hash__": _hash,
                "__module__": getattr(target_class, "__module__", __name__),
                "__qualname__": getattr(
                    target_class,
                    "__qualname__",
                    target_class.__name__,
                ),
            },
        )

    @staticmethod
    def pure(
        func: Callable[[object], R] | Callable[[], R],
    ) -> Callable[[object], R] | Callable[[], R]:
        """Transform function into pure function with automatic memoization caching.

        Returns:
            Callable: Wrapped pure function with memoization.

        """
        return FlextGuards.PureWrapper(func)

    # =========================================================================
    # CONFIGURATION MANAGEMENT - FLEXT TYPES INTEGRATION
    # =========================================================================

    @classmethod
    def configure_guards_system(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure guards system via Settings → SystemConfigs bridge.

        - Valida environment/log_level/validation_level com GuardsConfig.
        - Mantém compatibilidade adicionando as chaves derivadas esperadas.

        Returns:
            FlextResult[FlextTypes.Config.ConfigDict]: Validated configuration dictionary.

        """
        try:
            # Constrói Settings a partir do input e valida com GuardsConfig
            settings_res = FlextConfig.create_from_environment(
                extra_settings=cast("FlextTypes.Core.Dict", config)
                if isinstance(config, dict)
                else None,
            )
            if settings_res.is_failure:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    settings_res.error or "Failed to create GuardsSettings",
                )
            # Convert FlextConfig to dict for backwards compatibility
            cfg_res = FlextResult[FlextTypes.Config.ConfigDict].ok(
                cast("FlextTypes.Config.ConfigDict", settings_res.value.to_dict())
            )
            if cfg_res.is_failure:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    cfg_res.error or "Failed to validate GuardsConfig",
                )

            base = cfg_res.value

            # Defaults compatíveis com implementação anterior
            if "log_level" not in config:
                base["log_level"] = FlextConstants.Config.LogLevel.DEBUG.value
            if "validation_level" not in config:
                base["validation_level"] = (
                    FlextConstants.Config.ValidationLevel.NORMAL.value
                )

            # Campos adicionais esperados
            base["enable_pure_function_caching"] = bool(
                config.get("enable_pure_function_caching", True)
            )
            base["enable_immutable_classes"] = bool(
                config.get("enable_immutable_classes", True)
            )
            base["enable_validation_guards"] = bool(
                config.get("enable_validation_guards", True)
            )
            max_cache_value = config.get("max_cache_size", 1000)
            base["max_cache_size"] = (
                int(max_cache_value)
                if isinstance(max_cache_value, (int, str))
                else 1000
            )
            base["enable_strict_validation"] = bool(
                config.get("enable_strict_validation", False)
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(base)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Guards system configuration failed: {e}",
            )

    @classmethod
    def get_guards_system_config(cls) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current guards system configuration with runtime information.

        Returns:
            FlextResult[FlextTypes.Config.ConfigDict]: A snapshot of current config and metrics.

        """
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Current system state
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                # Guards-specific settings
                "enable_pure_function_caching": True,
                "enable_immutable_classes": True,
                "enable_validation_guards": True,
                "max_cache_size": 1000,
                "enable_strict_validation": False,
                # Runtime metrics
                "active_pure_functions": 0,  # Would be tracked in registry
                "cache_hit_ratio": 0.0,  # Would be calculated from cache stats
                "validation_errors_count": 0,  # Would be tracked in validation
                "immutable_classes_created": 0,  # Would be tracked in factory
                # Available guard types
                "available_guard_types": [
                    "pure_functions",
                    "immutable_classes",
                    "validation_utils",
                    "factory_methods",
                    "builder_patterns",
                    "type_guards",
                ],
                # Performance settings
                "cache_cleanup_interval": 300,  # 5 minutes
                "enable_performance_monitoring": False,
                "enable_debug_logging": False,
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get guards system configuration: {e}",
            )

    @classmethod
    def create_environment_guards_config(
        cls,
        environment: FlextTypes.Config.Environment,
        validation_level: str | None = None,
        *,
        cache_enabled: bool | None = None,
        **kwargs: object,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Create environment-specific guards system configuration.

        Returns:
            FlextResult[FlextTypes.Config.ConfigDict]: Environment-tailored configuration.

        """
        try:
            # Validate environment
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if environment not in valid_environments:
                return FlextResult[FlextTypes.Config.ConfigDict].fail(
                    f"Invalid environment '{environment}'. Valid options: {valid_environments}",
                )

            # Base configuration
            config: FlextTypes.Config.ConfigDict = {
                "environment": environment,
            }

            # Environment-specific settings
            if environment == "production":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.WARNING.value,
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "enable_pure_function_caching": True,  # Caching for performance
                        "enable_immutable_classes": True,  # Immutability for safety
                        "enable_validation_guards": True,  # Strict validation in production
                        "max_cache_size": 2000,  # Larger cache for production
                        "enable_strict_validation": True,  # Strict validation in production
                        "enable_performance_monitoring": True,  # Performance monitoring
                        "enable_debug_logging": False,  # No debug logging in production
                    },
                )
            elif environment == "development":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.DEBUG.value,
                        "validation_level": FlextConstants.Config.ValidationLevel.LOOSE.value,
                        "enable_pure_function_caching": False,  # No caching for fresh results
                        "enable_immutable_classes": True,  # Immutability for consistency
                        "enable_validation_guards": True,  # Validation for catching issues
                        "max_cache_size": 100,  # Small cache for development
                        "enable_strict_validation": False,  # Flexible validation in development
                        "enable_performance_monitoring": False,  # No performance monitoring
                        "enable_debug_logging": True,  # Full debug logging for development
                    },
                )
            elif environment == "test":
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.ERROR.value,
                        "validation_level": FlextConstants.Config.ValidationLevel.STRICT.value,
                        "enable_pure_function_caching": False,  # No caching in tests
                        "enable_immutable_classes": True,  # Immutability for test consistency
                        "enable_validation_guards": True,  # Validation for test accuracy
                        "max_cache_size": 50,  # Minimal cache for tests
                        "enable_strict_validation": True,  # Strict validation for tests
                        "enable_performance_monitoring": False,  # No performance monitoring in tests
                        "enable_test_utilities": True,  # Special test utilities
                    },
                )
            else:  # staging, local, etc.
                config.update(
                    {
                        "log_level": FlextConstants.Config.LogLevel.INFO.value,
                        "validation_level": FlextConstants.Config.ValidationLevel.NORMAL.value,
                        "enable_pure_function_caching": True,  # Caching for performance
                        "enable_immutable_classes": True,  # Immutability for safety
                        "enable_validation_guards": True,  # Standard validation
                        "max_cache_size": 1000,  # Standard cache size
                        "enable_strict_validation": False,  # Balanced validation
                    },
                )

            # Apply custom overrides if provided
            if validation_level is not None:
                config["validation_level"] = validation_level
            if cache_enabled is not None:
                config["enable_pure_function_caching"] = cache_enabled

            # Apply any additional kwargs (filter to compatible config types)
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    config[key] = value
                elif isinstance(value, list):
                    config[key] = list(value)  # Ensure it's a proper list
                elif isinstance(value, dict):
                    config[key] = dict(value)  # Ensure it's a proper dict

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to create environment guards config: {e}",
            )

    @classmethod
    def optimize_guards_performance(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Optimize guards system performance based on configuration.

        Returns:
            FlextResult[FlextTypes.Config.ConfigDict]: Optimized configuration.

        """
        try:
            # Extract performance level or determine from config
            performance_level = config.get("performance_level", "medium")

            # Base optimization settings
            optimized_config: FlextTypes.Config.ConfigDict = {
                "performance_level": performance_level,
                "optimization_enabled": True,
            }

            # Performance level specific optimizations
            if performance_level == "high":
                optimized_config.update(
                    {
                        "max_cache_size": config.get("max_cache_size", 5000),
                        "enable_pure_function_caching": True,
                        "cache_cleanup_interval": 600,  # 10 minutes
                        "enable_lazy_validation": True,
                        "batch_validation_size": 100,
                        "enable_concurrent_guards": True,
                        "memory_optimization": "aggressive",
                        "enable_cache_prewarming": True,
                    },
                )
            elif performance_level == "medium":
                optimized_config.update(
                    {
                        "max_cache_size": config.get("max_cache_size", 2000),
                        "enable_pure_function_caching": True,
                        "cache_cleanup_interval": 300,  # 5 minutes
                        "enable_lazy_validation": False,
                        "batch_validation_size": 50,
                        "enable_concurrent_guards": False,
                        "memory_optimization": "balanced",
                        "enable_cache_prewarming": False,
                    },
                )
            elif performance_level == "low":
                optimized_config.update(
                    {
                        "max_cache_size": config.get("max_cache_size", 500),
                        "enable_pure_function_caching": False,
                        "cache_cleanup_interval": 60,  # 1 minute
                        "enable_lazy_validation": False,
                        "batch_validation_size": 10,
                        "enable_concurrent_guards": False,
                        "memory_optimization": "conservative",
                        "enable_cache_prewarming": False,
                    },
                )
            else:
                # Default/custom performance level
                optimized_config.update(
                    {
                        "max_cache_size": config.get("max_cache_size", 1000),
                        "enable_pure_function_caching": config.get(
                            "enable_pure_function_caching",
                            True,
                        ),
                        "cache_cleanup_interval": 300,
                        "memory_optimization": "balanced",
                    },
                )

            # Merge with original config
            optimized_config.update(
                {
                    key: value
                    for key, value in config.items()
                    if key not in optimized_config
                },
            )

            return FlextResult[FlextTypes.Config.ConfigDict].ok(optimized_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Guards performance optimization failed: {e}",
            )

    # Factory and builder methods removed - use direct class construction
    @staticmethod
    def make_factory(name: str, defaults: FlextTypes.Core.Dict) -> FlextResult[object]:
        """Create simple factory with name and defaults.

        Returns:
            FlextResult[object]: A factory instance exposing `create(**overrides)`.

        """
        try:

            class Factory:
                def __init__(self) -> None:
                    self.defaults = defaults
                    self.name = name

                def create(self, **overrides: object) -> object:
                    kwargs = {**self.defaults, **overrides}
                    return cast("object", type(self.name, (), kwargs)())

            return FlextResult[object].ok(Factory())
        except Exception as e:
            return FlextResult[object].fail(f"Failed to create factory: {e!s}")

    @staticmethod
    def make_builder(name: str, fields: dict[str, type]) -> FlextResult[object]:
        """Create simple builder with name and fields.

        Returns:
            FlextResult[object]: A builder instance exposing `set(key, value)` and `build()`.

        """
        try:

            class Builder:
                def __init__(self) -> None:
                    self._kwargs: FlextTypes.Core.Dict = {}
                    self.name = name
                    self.fields = fields

                def set(self, key: str, value: object) -> Builder:
                    self._kwargs[key] = value
                    return self

                def build(self) -> object:
                    return cast("object", type(self.name, (), self._kwargs)())

            return FlextResult[object].ok(Builder())
        except Exception as e:
            return FlextResult[object].fail(f"Failed to create builder: {e!s}")


__all__: FlextTypes.Core.StringList = [
    "FlextGuards",
]
