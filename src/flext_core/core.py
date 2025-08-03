"""FLEXT Core Main - Extension Layer Central Orchestration.

Central orchestration point providing unified access to all FLEXT Core functionality
through a single, cohesive interface across the 32-project FLEXT ecosystem. Foundation
for application-wide consistency, subsystem coordination, and enterprise integration
patterns in distributed data integration platforms.

Module Role in Architecture:
    Extension Layer → Central Orchestration → Unified System Access

    This module provides unified system access patterns used throughout FLEXT projects:
    - Singleton pattern for application-wide consistency and state management
    - Facade pattern hiding complex subsystem interactions
    - Type-safe operations with comprehensive dependency injection
    - Railway-oriented programming utilities for functional composition

Core Architecture Patterns:
    Singleton Pattern: Application-wide consistency with global state management
    Facade Pattern: Subsystem complexity hiding through unified interface
    Dependency Injection: Type-safe service registration and retrieval
    Railway Programming: Functional composition with FlextResult integration

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Singleton orchestration, facade interface, DI integration
    🚧 Active Development: Advanced orchestration patterns (Enhancement 5 - Low)
    📋 TODO Integration: Plugin system orchestration (Priority 3)

Core Orchestration Features:
    FlextCore: Main singleton providing unified system access
    Container Integration: Type-safe dependency injection with service management
    Logging Configuration: Centralized logging setup and context management
    Result Pattern Integration: Railway-oriented programming for error handling
    Configuration Management: Settings caching with environment variable support

Ecosystem Usage Patterns:
    # FLEXT Service Applications
    from flext_core import FlextCore

    core = FlextCore.get_instance()
    container = core.get_container()
    logger = core.get_logger(__name__)

    # Service registration
    container.register("user_service", user_service_instance)

    # FLEXT Applications
    class FlextApplication:
        def __init__(self):
            self.core = FlextCore.get_instance()
            self.setup_dependencies()

        def setup_dependencies(self):
            self.core.configure_logging(level="INFO")
            self.core.register_services()

    # Singer Tap/Target Integration
    core = FlextCore.get_instance()
    oracle_service = core.get_service("oracle_client")
    result = oracle_service.extract_data("users")

Enterprise Integration Benefits:
    - Application-wide consistency through singleton pattern
    - Subsystem coordination through facade interface
    - Type-safe dependency management with compile-time validation
    - Functional programming support for data processing pipelines

Quality Standards:
    - Singleton instance must be thread-safe for concurrent access
    - Facade interface must remain stable across version updates
    - Service registration must be type-safe with proper error handling
    - Configuration management must support environment-specific settings

See Also:
    docs/TODO.md: Enhancement 5 - Advanced orchestration pattern development
    container.py: Dependency injection implementation details
    result.py: Railway-oriented programming pattern implementation

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.constants import FlextConstants
from flext_core.container import (
    FlextContainer,
    ServiceKey,
    get_flext_container,
    get_typed,
    register_typed,
)
from flext_core.guards import ValidatedModel, immutable, is_dict_of, pure
from flext_core.loggings import FlextLogger, FlextLoggerFactory
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextCore:
    """Central orchestration class providing unified access to FLEXT Core functionality.

    Implements facade pattern to provide simple, unified interface to complex subsystems
    while maintaining type safety and error handling. Serves as the
    primary entry point for all FLEXT Core operations.

    Architecture:
        - Singleton pattern ensuring single global instance
        - Facade pattern hiding complex subsystem interactions
        - Type-safe service registration and retrieval
        - Railway-oriented programming utilities for functional composition
        - Settings caching for performance optimization

    Core Subsystem Integration:
        - Container: Type-safe dependency injection with FlextContainer
        - Logging: Structured logging with FlextLogger integration
        - Result: Railway-oriented programming with FlextResult patterns
        - Configuration: Settings management with caching
        - Constants: System-wide constants access through FlextConstants

    Enterprise Features:
        - Type-safe service registration using ServiceKey[T] pattern
        - Railway programming utilities for functional data processing
        - Structured logging with automatic context management
        - Configuration caching for performance optimization
        - Error handling without exception propagation

    Usage Patterns:
        # Singleton access
        flext = FlextCore.get_instance()

        # Type-safe service management
        USER_SERVICE_KEY = ServiceKey[UserService]("user_service")
        flext.register_service(USER_SERVICE_KEY, UserService())
        user_service_result = flext.get_service(USER_SERVICE_KEY)

        # Structured logging
        logger = flext.get_logger("myapp.service")
        logger.info("Service started", service="user", version="1.0")

        # Railway programming
        pipeline = flext.pipe(
            validate_input,
            transform_data,
            save_to_database
        )
        result = pipeline(input_data)

        # Conditional processing
        conditional_processor = flext.when(
            lambda x: x > 0,
            lambda x: flext.ok(x * 2),
            lambda x: flext.fail("Negative value")
        )

        # Side effects in pipelines
        logged_pipeline = flext.pipe(
            validate_input,
            flext.tap(lambda x: logger.info("Validated", data=x)),
            process_data,
            flext.tap(lambda x: logger.info("Processed", result=x))
        )
    """

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize FLEXT Core with all subsystems."""
        # Core container
        self._container = get_flext_container()

        # Logging system - FlextLogger configuration
        # Note: Logging configuration will be implemented when
        # FlextLogger.configure is available

        # Settings - Use object instead of Any
        self._settings_cache: dict[type[object], object] = {}

    @classmethod
    def get_instance(cls) -> FlextCore:
        """Get singleton instance of FlextCore."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # CONTAINER ACCESS
    # =========================================================================

    @property
    def container(self) -> FlextContainer:
        """Access dependency injection container."""
        return self._container

    def register_service[S](
        self,
        key: ServiceKey[S],
        service: S,
    ) -> FlextResult[None]:
        """Register typed service in container.

        Args:
            key: Type-safe service key
            service: Service instance

        Returns:
            Result of registration

        """
        return register_typed(self._container, key, service)

    def get_service[S](self, key: ServiceKey[S]) -> FlextResult[S]:
        """Get typed service from container.

        Args:
            key: Type-safe service key

        Returns:
            Result containing service or error

        """
        return get_typed(self._container, key)

    # =========================================================================
    # LOGGING ACCESS
    # =========================================================================

    def get_logger(self, name: str) -> FlextLogger:
        """Get configured logger instance.

        Args:
            name: Logger name (typically module name)

        Returns:
            Configured logger

        """
        return FlextLoggerFactory.get_logger(name)

    def configure_logging(
        self,
        *,
        log_level: str = "INFO",
        _json_output: bool | None = None,
    ) -> None:
        """Configure logging system.

        Args:
            log_level: Minimum log level
            _json_output: Force JSON output (not yet implemented)

        """
        FlextLoggerFactory.set_global_level(log_level)
        # Note: FlextLogger.configure not yet implemented

    # =========================================================================
    # RESULT PATTERN ACCESS
    # =========================================================================

    @staticmethod
    def ok[V](value: V) -> FlextResult[V]:
        """Create successful Result.

        Args:
            value: Success value

        Returns:
            Success Result

        """
        return FlextResult.ok(value)

    @staticmethod
    def fail[V](error: str) -> FlextResult[V]:
        """Create failed Result.

        Args:
            error: Error message

        Returns:
            Failed Result

        """
        return FlextResult.fail(error)

    # =========================================================================
    # RAILWAY PROGRAMMING
    # =========================================================================

    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create pipeline of Result-returning functions."""

        def pipeline(value: object) -> FlextResult[object]:
            result = FlextResult.ok(value)
            for func in funcs:
                if result.is_failure:
                    break
                result = func(result.unwrap())
            return result

        return pipeline

    @staticmethod
    def compose(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Compose Result-returning functions (right to left)."""
        return FlextCore.pipe(*reversed(funcs))

    @staticmethod
    def when[V](
        predicate: Callable[[V], bool],
        then_func: Callable[[V], FlextResult[V]],
        else_func: Callable[[V], FlextResult[V]] | None = None,
    ) -> Callable[[V], FlextResult[V]]:
        """Conditional Result execution."""

        def conditional(value: V) -> FlextResult[V]:
            if predicate(value):
                return then_func(value)
            if else_func:
                return else_func(value)
            return FlextResult.ok(value)

        return conditional

    @staticmethod
    def tap[V](
        side_effect: Callable[[V], None],
    ) -> Callable[[V], FlextResult[V]]:
        """Execute side effect in pipeline."""

        def side_effect_wrapper(value: V) -> FlextResult[V]:
            side_effect(value)
            return FlextResult.ok(value)

        return side_effect_wrapper

    # =========================================================================
    # CONFIGURATION ACCESS
    # =========================================================================

    def get_settings(
        self,
        settings_class: type[object],
    ) -> object:
        """Get settings instance with caching.

        Args:
            settings_class: Settings model class

        Returns:
            Configured settings instance

        """
        if settings_class not in self._settings_cache:
            # Note: FlextCoreSettings implementation pending in config module
            self._settings_cache[settings_class] = settings_class()
        return self._settings_cache[settings_class]

    @property
    def constants(self) -> type[FlextConstants]:
        """Access FLEXT constants."""
        return FlextConstants

    # =========================================================================
    # VALIDATION & GUARDS - SOLID Implementation with DIP
    # =========================================================================

    def validate_type(
        self,
        obj: object,
        expected_type: type[object],
    ) -> FlextResult[object]:
        """Validate object type using dependency injection pattern.

        Args:
            obj: Object to validate
            expected_type: Expected type

        Returns:
            FlextResult with validation outcome

        """
        if not isinstance(obj, expected_type):
            return FlextResult.fail(
                f"Expected {expected_type.__name__}, got {type(obj).__name__}",
            )
        return FlextResult.ok(obj)

    def validate_dict_structure(
        self,
        obj: object,
        value_type: type[object],
    ) -> FlextResult[dict[str, object]]:
        """Validate dictionary structure using guards module.

        Args:
            obj: Object to validate as dictionary
            value_type: Expected type of dictionary values

        Returns:
            FlextResult with validated dictionary or error

        """
        if not isinstance(obj, dict):
            return FlextResult.fail("Expected dictionary")

        if not is_dict_of(obj, value_type):
            return FlextResult.fail(
                f"Dictionary values must be of type {value_type.__name__}",
            )

        return FlextResult.ok(obj)

    def create_validated_model[T: ValidatedModel](
        self,
        model_class: type[T],
        **data: object,
    ) -> FlextResult[T]:
        """Create validated model using guards module integration.

        Args:
            model_class: ValidatedModel subclass
            **data: Model data

        Returns:
            FlextResult with validated model or error

        """
        # Use the safe creation method from ValidatedModel
        return model_class.create(**data)  # type: ignore[return-value]

    def make_immutable[T](self, cls: type[T]) -> type[T]:
        """Make class immutable using guards module.

        Args:
            cls: Class to make immutable

        Returns:
            Immutable version of the class

        """
        return immutable(cls)

    def make_pure[T](self, func: T) -> T:
        """Make function pure using guards module.

        Args:
            func: Function to make pure

        Returns:
            Pure version of the function

        """
        return pure(func)  # type: ignore[return-value]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation

        """
        service_count = self._container.get_service_count()
        return f"FlextCore(services={service_count})"


# Convenience function for global access
def flext_core() -> FlextCore:
    """Get global FlextCore instance with convenient access pattern.

    Convenience function providing direct access to the global FlextCore singleton
    instance without requiring explicit class method calls. Maintains singleton
    pattern while providing simpler access syntax.

    Returns:
        Global FlextCore singleton instance

    Usage:
        # Convenient access
        flext = flext_core()

        # Equivalent to
        flext = FlextCore.get_instance()

        # Direct usage in functional style
        result = flext_core().ok("success").map(str.upper)

    """
    return FlextCore.get_instance()


# Export API
__all__ = [
    "FlextCore",
    "flext_core",
]
