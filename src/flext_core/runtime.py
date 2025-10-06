"""Layer 0.5 runtime connectors exposing external libraries without circular imports.

This module isolates direct imports of third-party libraries that are required by
lower foundation layers (constants, config, loggings, models, exceptions,
typings, utilities). By centralizing the integration points here we eliminate
circular import risks while still providing direct access to the original APIs,
in line with FLEXT's *direct API* rule.

**CRITICAL**: This module maintains ZERO imports from other flext_core modules
to break circular dependencies. All runtime primitives, validation patterns,
and type guards are defined here without dependencies.

Dependency Layer: 0.5 (external runtime connectors only)
Dependencies: structlog, dependency_injector, Python stdlib only
Used by: config, container, loggings, dispatcher and any lower-layer module
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, TypeGuard, cast

import structlog
from dependency_injector import containers, providers


class FlextRuntime:
    """Expose structlog and dependency-injector primitives to foundation layers.

    The class intentionally keeps ZERO imports from other ``flext_core`` modules
    so that any layer (including ``FlextConstants`` and ``FlextTypes``) can
    access runtime primitives without creating circular dependencies. All
    methods return the concrete library modules or perform direct configuration
    using the official APIs—no wrappers, aliases or indirection beyond this
    centralized access point.

    **Enhanced Features**:
    - Configuration primitives and defaults without FlextConfig dependency
    - Validation patterns (email, URL, phone) without FlextConstants dependency
    - Logging level constants without FlextLogger dependency
    - Type guard utilities without FlextTypes dependency
    - Serialization utilities without FlextModels dependency
    - Runtime primitives without any flext_core imports
    """

    _structlog_configured: bool = False

    # =========================================================================
    # CONFIGURATION PRIMITIVES (No flext_core imports)
    # =========================================================================
    # NOTE: These provide default configuration values that can be used by
    # FlextConstants and FlextConfig without creating circular dependencies.

    # Core application defaults
    DEFAULT_APP_NAME: str = "FLEXT Application"
    DEFAULT_ENVIRONMENT: str = "development"
    DEFAULT_ENCODING: str = "utf-8"

    # Logging defaults
    DEFAULT_LOG_LEVEL: str = "INFO"
    DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Logging level constants
    LOG_LEVEL_DEBUG: str = "DEBUG"
    LOG_LEVEL_INFO: str = "INFO"
    LOG_LEVEL_WARNING: str = "WARNING"
    LOG_LEVEL_ERROR: str = "ERROR"
    LOG_LEVEL_CRITICAL: str = "CRITICAL"

    # Numeric logging levels for structlog integration
    LOG_LEVEL_NUM_DEBUG: int = logging.DEBUG
    LOG_LEVEL_NUM_INFO: int = logging.INFO
    LOG_LEVEL_NUM_WARNING: int = logging.WARNING
    LOG_LEVEL_NUM_ERROR: int = logging.ERROR
    LOG_LEVEL_NUM_CRITICAL: int = logging.CRITICAL

    # Performance and timeout defaults
    DEFAULT_TIMEOUT: int = 30
    DEFAULT_MAX_WORKERS: int = 10
    DEFAULT_PAGE_SIZE: int = 50
    DEFAULT_BATCH_SIZE: int = 100
    DEFAULT_RETRY_ATTEMPTS: int = 3

    # Network defaults
    DEFAULT_PORT: int = 8000
    DEFAULT_HOST: str = "localhost"
    DEFAULT_CONNECTION_POOL_SIZE: int = 10

    # Cache defaults
    DEFAULT_CACHE_TTL: int = 300
    DEFAULT_CACHE_MAX_SIZE: int = 1000

    # Security defaults
    DEFAULT_JWT_EXPIRY_MINUTES: int = 15
    DEFAULT_BCRYPT_ROUNDS: int = 12

    # Environment variable prefix
    ENV_PREFIX: str = "FLEXT_"
    ENV_FILE_DEFAULT: str = ".env"
    ENV_NESTED_DELIMITER: str = "__"

    # =========================================================================
    # TYPE GUARD UTILITIES (No flext_core imports, uses regex directly)
    # =========================================================================
    # NOTE: These use compiled regex patterns for runtime validation without
    # depending on FlextConstants to avoid circular imports.

    @staticmethod
    def is_valid_email(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid email string.

        Uses RFC 5322 simplified pattern for validation.
        FlextConstants.Platform.PATTERN_EMAIL contains the same pattern.

        Args:
            value: Value to check

        Returns:
            True if value is a valid email string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(
            r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}"
            r"[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
        )
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_url(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid HTTP/HTTPS URL string.

        FlextConstants.Platform.URL_PATTERN contains similar validation.

        Args:
            value: Value to check

        Returns:
            True if value is a valid URL string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_phone(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid phone number string.

        Uses international format validation.
        FlextConstants.Platform.PATTERN_PHONE_NUMBER contains similar pattern.

        Args:
            value: Value to check

        Returns:
            True if value is a valid phone number string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(r"^\+?1?\d{9,15}$")
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_uuid(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid UUID string.

        Supports both hyphenated and non-hyphenated UUID formats.
        FlextConstants.Platform.PATTERN_UUID contains similar validation.

        Args:
            value: Value to check

        Returns:
            True if value is a valid UUID string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(
            r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
            r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
        )
        return pattern.match(value) is not None

    @staticmethod
    def is_dict_like(value: object) -> TypeGuard[dict[str, Any]]:
        """Type guard to check if value is dict-like.

        Args:
            value: Value to check

        Returns:
            True if value is a dict or dict-like object, False otherwise

        """
        return isinstance(value, dict)

    @staticmethod
    def is_list_like(value: object) -> TypeGuard[list[Any]]:
        """Type guard to check if value is list-like.

        Args:
            value: Value to check

        Returns:
            True if value is a list or list-like sequence, False otherwise

        """
        return isinstance(value, list)

    @staticmethod
    def is_valid_json(value: object) -> TypeGuard[str]:
        """Type guard to check if value is valid JSON string.

        Args:
            value: Value to check

        Returns:
            True if value is a valid JSON string, False otherwise

        """
        if not isinstance(value, str):
            return False
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    @staticmethod
    def is_valid_path(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid file/directory path.

        Args:
            value: Value to check

        Returns:
            True if value is a valid path string, False otherwise

        """
        if not isinstance(value, str):
            return False
        # Basic path validation - checks for invalid characters
        pattern = re.compile(r'^[^<>:"|?*\x00-\x1F]+$')
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_identifier(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid Python identifier.

        Args:
            value: Value to check

        Returns:
            True if value is a valid Python identifier, False otherwise

        """
        if not isinstance(value, str):
            return False
        return value.isidentifier()

    # =========================================================================
    # SERIALIZATION UTILITIES (No flext_core imports)
    # =========================================================================

    @staticmethod
    def safe_get_attribute(obj: object, attr: str, default: object = None) -> object:
        """Safe attribute access without raising AttributeError.

        Args:
            obj: Object to get attribute from
            attr: Attribute name
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default

        """
        return getattr(obj, attr, default)

    @staticmethod
    def safe_serialize_to_dict(obj: object) -> dict[str, Any] | None:
        """Serialize object to dictionary without dependencies.

        Attempts multiple serialization strategies without importing
        FlextModels or other flext_core modules.

        Args:
            obj: Object to serialize

        Returns:
            Dictionary representation or None if serialization fails

        """
        # Strategy 1: Check for model_dump method (Pydantic)
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                result = obj.model_dump()  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                import logging

                logging.getLogger(__name__).debug(
                    f"model_dump() serialization strategy failed: {e}"
                )

        # Strategy 2: Check for dict method
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                result = obj.dict()  # type: ignore[attr-defined]
                if isinstance(result, dict):
                    return result
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                import logging

                logging.getLogger(__name__).debug(
                    f"dict() serialization strategy failed: {e}"
                )

        # Strategy 3: Check for __dict__ attribute
        if hasattr(obj, "__dict__"):
            try:
                result = obj.__dict__
                if isinstance(result, dict):
                    return dict(result)
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                import logging

                logging.getLogger(__name__).debug(
                    f"__dict__ serialization strategy failed: {e}"
                )

        # Strategy 4: Check if already dict
        if isinstance(obj, dict):
            return obj

        return None

    @staticmethod
    def is_optional_type(type_hint: object) -> bool:
        """Check if type hint represents Optional[T] (Union[T, None]).

        Args:
            type_hint: Type hint to check

        Returns:
            True if type hint is Optional, False otherwise

        """
        try:
            import typing

            # Get the origin (e.g., Union for Union[T, None])
            origin = typing.get_origin(type_hint)
            if origin is typing.Union:
                # Get the args (e.g., (T, NoneType) for Union[T, None])
                args = typing.get_args(type_hint)
                # Check if None is one of the args
                return type(None) in args
            return False
        except Exception:
            return False

    @staticmethod
    def extract_generic_args(type_hint: object) -> tuple[Any, ...]:
        """Extract generic type arguments from a type hint.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            import typing

            return typing.get_args(type_hint)
        except Exception:
            return ()

    @staticmethod
    def is_sequence_type(type_hint: object) -> bool:
        """Check if type hint represents a sequence type (list, tuple, etc.).

        Args:
            type_hint: Type hint to check

        Returns:
            True if type hint is a sequence type, False otherwise

        """
        try:
            import typing
            from collections.abc import Sequence

            origin = typing.get_origin(type_hint)
            if origin is not None:
                return issubclass(origin, Sequence)
            return False
        except Exception:
            return False

    @staticmethod
    def structlog() -> ModuleType:
        """Return the imported structlog module."""
        return structlog

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @classmethod
    def configure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[Callable[..., object]] | None = None,
        wrapper_class_factory: object | None = None,
        logger_factory: object | None = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults.

        Args:
            log_level: Numeric log level. Defaults to ``logging.INFO``.
            console_renderer: When ``True`` use the console renderer, otherwise
                JSON renderer.
            additional_processors: Optional extra processors appended after the
                standard FLEXT processors.
            wrapper_class_factory: Custom wrapper factory passed to structlog.
                Falls back to :func:`structlog.make_filtering_bound_logger`.
            logger_factory: Custom logger factory. Defaults to
                :class:`structlog.PrintLoggerFactory`.
            cache_logger_on_first_use: Forwarded to structlog configuration.

        """
        if cls._structlog_configured:
            return

        module = structlog
        if module.is_configured():
            cls._structlog_configured = True
            return

        level_to_use = log_level if log_level is not None else logging.INFO

        processors: list[Any] = [
            module.contextvars.merge_contextvars,
            module.processors.add_log_level,
            module.processors.TimeStamper(fmt="iso"),
            module.processors.StackInfoRenderer(),
        ]
        if additional_processors:
            processors.extend(additional_processors)

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(module.processors.JSONRenderer())

        module.configure(
            processors=processors,
            wrapper_class=cast(
                "type[structlog.BoundLoggerBase] | None",
                wrapper_class_factory
                if wrapper_class_factory is not None
                else module.make_filtering_bound_logger(level_to_use),
            ),
            logger_factory=cast(
                "Callable[[], structlog.BoundLoggerBase] | None",
                logger_factory
                if logger_factory is not None
                else module.PrintLoggerFactory(),
            ),
            cache_logger_on_first_use=cache_logger_on_first_use,
        )

        cls._structlog_configured = True


__all__ = ["FlextRuntime"]
