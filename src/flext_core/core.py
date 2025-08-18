"""Central orchestration for unified FLEXT Core system access."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from flext_core.constants import FlextConstants
from flext_core.container import (
    FlextContainer,
    FlextServiceKey,
    get_flext_container,
)
from flext_core.guards import ValidatedModel, immutable, is_dict_of, pure
from flext_core.loggings import FlextLogger, FlextLoggerFactory, FlextLogLevel
from flext_core.result import FlextResult

# Type variables for function signatures
P = ParamSpec("P")
R = TypeVar("R")


class FlextCore:
    """Singleton facade providing unified access to FLEXT functionality.

    Manages container, logging, configuration, and validation subsystems.
    Thread-safe with lazy initialization of components.
    """

    _instance: FlextCore | None = None

    def __init__(self) -> None:
        """Initialize FLEXT Core with all subsystems."""
        # Core container
        self._container = get_flext_container()

        # Logging system - FlextLogger configuration
        # Note: Logging configuration will be implemented when
        # FlextLogger.configure is available

        # Settings - Use an object instead of Any
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
        key: FlextServiceKey[S],
        service: S,
    ) -> FlextResult[None]:
        """Register typed service in container.

        Args:
            key: Type-safe service key
            service: Service instance

        Returns:
            Result of registration

        """
        return self._container.register(
            str(key),
            service,
        )

    def get_service[S](self, key: FlextServiceKey[S]) -> FlextResult[S]:
        """Get typed service from container.

        Args:
            key: Type-safe service key

        Returns:
            Result containing service or error

        """
        key_str = str(key)
        result = self._container.get(key_str)
        if result.is_failure:
            return FlextResult.fail(result.error or "Service not found")
        return FlextResult[S].ok(result.data)  # type: ignore[arg-type]

    # =========================================================================
    # LOGGING ACCESS
    # =========================================================================

    @staticmethod
    def get_logger(name: str) -> FlextLogger:
        """Get configured logger instance.

        Args:
            name: Logger name (typically module name)

        Returns:
            Configured logger

        """
        return FlextLoggerFactory.get_logger(name)

    @staticmethod
    def configure_logging(
        *,
        log_level: str = "INFO",
        _json_output: bool | None = None,
    ) -> None:
        """Configure logging system.

        Args:
            log_level: Minimum log level
            _json_output: Force JSON output (implemented via FlextLogger)

        """
        # Convert string log level to FlextLogLevel enum
        log_level_enum = FlextLogLevel.INFO
        try:
            log_level_enum = FlextLogLevel(log_level.upper())
        except (ValueError, AttributeError):
            # Fallback to INFO if invalid level
            log_level_enum = FlextLogLevel.INFO

        # Set global level via factory
        FlextLoggerFactory.set_global_level(log_level)

        # Configure JSON output if specified
        if _json_output is not None:
            FlextLogger.configure(
                log_level=log_level_enum,
                json_output=_json_output,
                add_timestamp=True,  # Always include timestamps
                add_caller=False,  # Keep caller info optional for performance
            )

    # =========================================================================
    # RESULT PATTERN ACCESS
    # =========================================================================

    @staticmethod
    def ok[T](value: T) -> FlextResult[T]:
        """Create successful Result.

        Args:
            value: Success value

        Returns:
            Success Result

        """
        return FlextResult[T].ok(value)

    @staticmethod
    def fail(error: str) -> FlextResult[object]:
        """Create failed Result.

        Args:
            error: Error message

        Returns:
            Failed Result

        """
        return FlextResult[object].fail(error)

    # =========================================================================
    # RAILWAY PROGRAMMING
    # =========================================================================

    @staticmethod
    def pipe(
        *funcs: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Create a pipeline of Result-returning functions."""

        def pipeline(value: object) -> FlextResult[object]:
            result: FlextResult[object] = FlextResult[object].ok(value)
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
            return FlextResult[V].ok(value)

        return conditional

    @staticmethod
    def tap[V](
        side_effect: Callable[[V], None],
    ) -> Callable[[V], FlextResult[V]]:
        """Execute side effect in pipeline."""

        def side_effect_wrapper(value: V) -> FlextResult[V]:
            side_effect(value)
            return FlextResult[V].ok(value)

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

    @staticmethod
    def validate_type[T](
        obj: object,
        expected_type: type[T],
    ) -> FlextResult[T]:
        """Validate an object type using a dependency injection pattern.

        Args:
            obj: Object to validate
            expected_type: Expected type

        Returns:
            FlextResult with a validation outcome

        """
        if not isinstance(obj, expected_type):
            return FlextResult[T].fail(
                f"Expected {expected_type.__name__}, got {type(obj).__name__}",
            )
        return FlextResult[T].ok(obj)

    @staticmethod
    def validate_dict_structure[V](
        obj: object,
        value_type: type[V],
    ) -> FlextResult[dict[str, V]]:
        """Validate dictionary structure using guards module.

        Args:
            obj: Object to validate as dictionary
            value_type: Expected type of dictionary values

        Returns:
            FlextResult with validated dictionary or error

        """
        # First check if obj is a dictionary at all
        if not isinstance(obj, dict):
            return FlextResult[dict[str, V]].fail(
                "Expected dictionary",
            )

        # Then check if all values are of the expected type
        if not is_dict_of(obj, value_type):
            return FlextResult[dict[str, V]].fail(
                f"Dictionary values must be of type {value_type.__name__}",
            )

        return FlextResult[dict[str, V]].ok(obj)

    @staticmethod
    def create_validated_model[T: ValidatedModel](
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
        return model_class.create(**data)

    @staticmethod
    def make_immutable[T](target_class: type[T]) -> type[T]:
        """Make class immutable using guards module.

        Args:
            target_class: Class to make immutable

        Returns:
            Immutable version of the class

        """
        return immutable(target_class)

    @staticmethod
    def make_pure(func: Callable[P, R]) -> Callable[P, R]:
        """Make function pure using guards module.

        Args:
            func: Function to make pure

        Returns:
            Pure version of the function

        """
        return pure(func)

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
    """Get global FlextCore instance with a convenient access pattern.

    Convenience function providing direct access to the global FlextCore singleton
    instance without requiring explicit class method calls. Maintains a singleton
    pattern while providing simpler access syntax.

    Returns:
      Global FlextCore singleton instance

    """
    return FlextCore.get_instance()


# Export API
__all__: list[str] = [
    "FlextCore",
    "flext_core",
]
