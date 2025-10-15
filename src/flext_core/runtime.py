"""Runtime utilities and external library connectors.

This module provides runtime utilities that consume patterns from FlextConstants
and expose external library APIs to higher-level modules, maintaining proper
dependency hierarchy while eliminating code duplication.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
import logging
import re
import typing
import uuid
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TypeGuard, cast

import structlog
from dependency_injector import containers, providers

from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes


class FlextRuntime:
    """Runtime utilities and external library connectors.

    Provides runtime utilities that consume patterns from FlextConstants
    and expose external library APIs to higher-level modules, maintaining
    proper dependency hierarchy while eliminating code duplication.

    Features:
    - Type guard utilities using FlextConstants validation patterns
    - Serialization utilities for Pydantic and dict[str, object] conversion
    - Optional type introspection utilities
    - Sequence type checking utilities
    - Direct access to external library modules (structlog, dependency_injector)
    - Structured logging configuration with FLEXT defaults

    Usage:
        >>> from flext_core.runtime import FlextRuntime
        >>>
        >>> # Type guards using constants patterns
        >>> if FlextRuntime.is_valid_email("user@example.com"):
        ...     print("Valid email")
        >>>
        >>> # Access external libraries
        >>> structlog = FlextRuntime.structlog()
        >>> providers = FlextRuntime.dependency_providers()
    """

    _structlog_configured: bool = False

    # Log level constants for compatibility with examples and tests
    LOG_LEVEL_DEBUG: str = "DEBUG"
    LOG_LEVEL_INFO: str = "INFO"
    LOG_LEVEL_WARNING: str = "WARNING"
    LOG_LEVEL_ERROR: str = "ERROR"
    LOG_LEVEL_CRITICAL: str = "CRITICAL"
    LOG_LEVEL_NUM_INFO: int = 20
    LOG_LEVEL_NUM_ERROR: int = 40

    # =========================================================================
    # TYPE GUARD UTILITIES (Uses regex patterns from FlextConstants)
    # =========================================================================

    @staticmethod
    def is_valid_email(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid email string.

        Uses RFC 5322 simplified pattern from FlextConstants.Platform.PATTERN_EMAIL.

        Args:
            value: Value to check

        Returns:
            True if value is a valid email string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_EMAIL)
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_url(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid HTTP/HTTPS URL string.

        Uses URL pattern from FlextConstants.Platform.PATTERN_URL.

        Args:
            value: Value to check

        Returns:
            True if value is a valid URL string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_URL, re.IGNORECASE)
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_phone(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid phone number string.

        Uses international format pattern from FlextConstants.Platform.PATTERN_PHONE_NUMBER.

        Args:
            value: Value to check

        Returns:
            True if value is a valid phone number string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_PHONE_NUMBER)
        return pattern.match(value) is not None

    @staticmethod
    def is_valid_uuid(value: object) -> TypeGuard[str]:
        """Type guard to check if value is a valid UUID string.

        Supports both hyphenated and non-hyphenated formats.
        Uses pattern from FlextConstants.Platform.PATTERN_UUID.

        Args:
            value: Value to check

        Returns:
            True if value is a valid UUID string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_UUID)
        return pattern.match(value) is not None

    @staticmethod
    def is_dict_like(value: object) -> TypeGuard[FlextTypes.Dict]:
        """Type guard to check if value is dict-like.

        Args:
            value: Value to check

        Returns:
            True if value is a dict[str, object] or dict-like object, False otherwise

        """
        return isinstance(value, dict)

    @staticmethod
    def is_list_like(value: object) -> TypeGuard[FlextTypes.List]:
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

        Uses pattern from FlextConstants.Platform.PATTERN_PATH.

        Args:
            value: Value to check

        Returns:
            True if value is a valid path string, False otherwise

        """
        if not isinstance(value, str):
            return False
        pattern = re.compile(FlextConstants.Platform.PATTERN_PATH)
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
    def safe_serialize_to_dict(obj: object) -> FlextTypes.Dict | None:
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
                model_dump_method = getattr(obj, "model_dump")
                result = model_dump_method()
                if isinstance(result, dict):
                    return cast("FlextTypes.Dict", result)
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                logging.getLogger(__name__).debug(
                    f"model_dump() serialization strategy failed: {e}"
                )

        # Strategy 2: Check for dict[str, object] method
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                dict_method = getattr(obj, "dict")
                result = dict_method()
                if isinstance(result, dict):
                    return cast("FlextTypes.Dict", result)
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                logging.getLogger(__name__).debug(
                    f"dict() serialization strategy failed: {e}"
                )

        # Strategy 3: Check for __dict__ attribute
        if hasattr(obj, "__dict__"):
            try:
                result = obj.__dict__
                return dict[str, object](result)
            except Exception as e:  # pragma: no cover
                # Silent fallback for serialization strategy - log at debug level
                # Extremely rare: __dict__ exists but dict[str, object]() conversion fails
                logging.getLogger(__name__).debug(
                    f"__dict__ serialization strategy failed: {e}"
                )

        # Strategy 4: Check if already dict
        if isinstance(obj, dict):
            return cast("FlextTypes.Dict", obj)

        return None

    @staticmethod
    def is_optional_type(type_hint: object) -> bool:
        """Check if type hint represents Optional[T] (Union[T, None] or T | None).

        Supports both Union[T, None] and Python 3.10+ syntax T | None.

        Args:
            type_hint: Type hint to check

        Returns:
            True if type hint is Optional, False otherwise

        """
        try:
            # Get the origin (e.g., Union for Union[T, None] or T | None)
            origin = typing.get_origin(type_hint)
            # Python 3.10+ uses types.UnionType for X | Y syntax
            # typing.Union is used for Union[X, Y] syntax
            if origin is typing.Union or str(type(type_hint).__name__) == "UnionType":
                # Get the args (e.g., (T, NoneType) for Union[T, None])
                args = typing.get_args(type_hint)
                # Check if None is one of the args
                return type(None) in args
            return False
        except Exception:  # pragma: no cover
            # Defensive: typing module failures are extremely rare
            return False

    @staticmethod
    def extract_generic_args(type_hint: object) -> tuple[object, ...]:
        """Extract generic type arguments from a type hint.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            # First try the standard typing.get_args
            args = typing.get_args(type_hint)
            if args:
                return args

            # Fallback for type aliases: check if it's a known type alias
            if hasattr(type_hint, "__name__"):
                type_name = getattr(type_hint, "__name__", "")
                # Handle common type aliases
                type_mapping: dict[str, tuple[type, ...]] = {
                    "StringList": (str,),
                    "IntList": (int,),
                    "FloatList": (float,),
                    "BoolList": (bool,),
                    "Dict": (str, object),
                    "List": (object,),
                    "StringDict": (str, str),
                    "IntDict": (str, int),
                    "FloatDict": (str, float),
                    "BoolDict": (str, bool),
                    "NestedDict": (str, object),  # Simplified
                }
                if type_name in type_mapping:
                    return type_mapping[type_name]

            return ()
        except Exception:  # pragma: no cover
            # Defensive: typing module failures are extremely rare
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
            origin = typing.get_origin(type_hint)
            if origin is not None:
                return issubclass(origin, Sequence)

            # Check if the type itself is a sequence subclass (for type aliases)
            if isinstance(type_hint, type) and issubclass(type_hint, Sequence):
                return True

            # Check __name__ for type aliases like StringList
            if hasattr(type_hint, "__name__"):
                type_name = getattr(type_hint, "__name__", "")
                # Common sequence type aliases
                if type_name in {
                    "StringList",
                    "IntList",
                    "FloatList",
                    "BoolList",
                    "List",
                }:
                    return True

            return False
        except Exception:  # pragma: no cover
            # Defensive: typing/issubclass failures are extremely rare
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

        processors: FlextTypes.List = [
            module.contextvars.merge_contextvars,
            module.processors.add_log_level,
            module.processors.TimeStamper(fmt="iso"),
            module.processors.StackInfoRenderer(),
        ]
        if additional_processors:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.extend(additional_processors)

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:  # pragma: no cover
            # Tested but not covered: structlog configures once per process
            processors.append(module.processors.JSONRenderer())

        module.configure(
            processors=cast("list[Callable[..., FlextTypes.Dict]]", processors),
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

    # =========================================================================
    # APPLICATION LAYER INTEGRATION (Using structlog directly - Layer 0.5)
    # =========================================================================
    # DESIGN: Integration uses structlog directly without importing from
    # Infrastructure layer (FlextContext, FlextLogger), avoiding circular imports.
    # USAGE: Opt-in helpers for APPLICATION/SERVICE layer only.
    # =========================================================================

    class Integration:
        """Application-layer integration helpers using structlog directly (Layer 0.5).

        **DESIGN**: These methods use structlog directly without importing from
        higher layers (FlextContext, FlextLogger), avoiding all circular imports.

        **USAGE**: Opt-in helpers for application/service layer to integrate
        foundation components with context tracking.

        **CORRECT USAGE** (Application Layer):
            ```python
            from flext_core import FlextContainer
            from flext_core.runtime import FlextRuntime

            container = FlextContainer.get_global()
            result = container.get("database")

            # Opt-in integration at application layer
            FlextRuntime.Integration.track_service_resolution(
                "database", resolved=result.is_success
            )
            ```

        **NOTES**:
            - Uses structlog directly (single source of truth for context)
            - No imports from Infrastructure layer (context.py, loggings.py)
            - Pure Layer 0.5 implementation
        """

        @staticmethod
        def track_service_resolution(
            service_name: str,
            *,
            resolved: bool = True,
            error_message: str | None = None,
        ) -> None:
            """Track service resolution with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service being resolved
                resolved: Whether resolution was successful
                error_message: Error message if resolution failed

            """
            # Get correlation_id directly from structlog (single source of truth)
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly (no FlextLogger wrapper needed)
            logger = structlog.get_logger(__name__)

            if resolved:
                logger.info(
                    "Service resolved",
                    service_name=service_name,
                    correlation_id=correlation_id,
                )
            else:
                logger.error(
                    "Service resolution failed",
                    service_name=service_name,
                    error=error_message,
                    correlation_id=correlation_id,
                )

        @staticmethod
        def log_config_access(
            key: str,
            value: object | None = None,
            *,
            masked: bool = False,
        ) -> None:
            """Log configuration access with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                key: Configuration key being accessed
                value: Configuration value (will be masked if sensitive)
                masked: Whether to mask the value in logs (for secrets)

            """
            # Get correlation_id directly from structlog
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly
            logger = structlog.get_logger(__name__)
            log_value = "***MASKED***" if masked else value

            logger.debug(
                "Configuration accessed",
                config_key=key,
                config_value=log_value,
                correlation_id=correlation_id,
            )

        @staticmethod
        def attach_context_to_result(
            result: object,
            *,
            attach_correlation: bool = True,
            attach_service_name: bool = False,
        ) -> object:
            """Attach context metadata to FlextResult (future-proofing).

            Uses structlog directly to avoid circular imports.

            Args:
                result: FlextResult instance
                attach_correlation: Whether to read correlation ID
                attach_service_name: Whether to read service name

            Returns:
                Result (currently unchanged, context available via structlog)

            """
            # Read from structlog contextvars directly
            context_vars = structlog.contextvars.get_contextvars()
            _ = context_vars.get("correlation_id") if attach_correlation else None
            _ = context_vars.get("service_name") if attach_service_name else None
            return result

        @staticmethod
        def track_domain_event(
            event_name: str,
            aggregate_id: str | None = None,
            event_data: FlextTypes.Dict | None = None,
        ) -> None:
            """Track domain event with context correlation.

            Uses structlog directly to avoid circular imports.

            Args:
                event_name: Name of the domain event
                aggregate_id: ID of the aggregate root
                event_data: Additional event data

            """
            # Get correlation_id directly from structlog
            context_vars = structlog.contextvars.get_contextvars()
            correlation_id = context_vars.get("correlation_id")

            # Use structlog directly
            logger = structlog.get_logger(__name__)

            logger.info(
                "Domain event emitted",
                event_name=event_name,
                aggregate_id=aggregate_id,
                event_data=event_data,
                correlation_id=correlation_id,
            )

        @staticmethod
        def setup_service_infrastructure(
            *,
            service_name: str,
            service_version: str | None = None,
            enable_context_correlation: bool = True,
        ) -> None:
            """Setup complete service infrastructure.

            Uses structlog directly to avoid circular imports.

            Args:
                service_name: Name of the service
                service_version: Version of the service
                enable_context_correlation: Whether to enable correlation

            """
            # Set service context directly in structlog contextvars
            structlog.contextvars.bind_contextvars(service_name=service_name)
            if service_version:
                structlog.contextvars.bind_contextvars(service_version=service_version)

            # Generate correlation ID if enabled
            if enable_context_correlation:
                correlation_id = f"flext-{uuid.uuid4().hex[:12]}"
                structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

            # Use structlog directly
            logger = structlog.get_logger(__name__)
            logger.info(
                "Service infrastructure initialized",
                service_name=service_name,
                service_version=service_version,
                correlation_enabled=enable_context_correlation,
            )


__all__ = ["FlextRuntime"]
