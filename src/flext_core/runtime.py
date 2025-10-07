"""Layer 0.5: Runtime utilities and external library connectors.

This module provides runtime utilities and exposes external libraries that are required by
higher-level modules. It imports from Layer 0 (constants.py) to use validation patterns,
eliminating duplication while maintaining the correct dependency hierarchy.

**ARCHITECTURE HIERARCHY**:
- Layer 0: constants.py, typings.py (pure Python, no flext_core imports)
- Layer 0.5: runtime.py (imports Layer 0, exposes external libraries)
- Layer 1+: All other modules (import Layer 0 and 0.5)

**KEY FEATURES**:
- Type guard utilities using patterns from FlextConstants.Platform
- Serialization utilities for object conversion
- Direct access to external library modules (structlog, dependency_injector)
- Structured logging configuration with FLEXT defaults
- Optional type introspection utilities
- Sequence type checking utilities

**DEPENDENCIES**: FlextConstants (Layer 0), structlog, dependency_injector, Python stdlib
**USED BY**: ALL higher-layer modules (loggings, config, container, models, exceptions)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import logging
import re
import typing
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, TypeGuard, cast

import structlog
from dependency_injector import containers, providers

from flext_core.constants import FlextConstants


class FlextRuntime:
    """Runtime utilities and external library connectors for higher layers (Layer 0.5).

    FlextRuntime provides runtime utilities that consume patterns from FlextConstants
    (Layer 0) and expose external library APIs to higher-level modules (Layer 1+).
    This eliminates code duplication while maintaining proper dependency hierarchy.

    **ARCHITECTURE ROLE**: Layer 0.5 - Runtime Utilities
        - Imports FlextConstants (Layer 0) for validation patterns
        - NO imports from other flext_core modules (loggings, config, models, etc.)
        - Imported by ALL higher-level modules for runtime utilities
        - Exposes structlog and dependency-injector APIs

    **PROVIDES**:
        - Type guard utilities using FlextConstants.Platform patterns
          (is_valid_email, is_valid_url, is_valid_uuid, etc.)
        - Serialization utilities for Pydantic and dict conversion
        - Optional type introspection (is_optional_type, extract_generic_args)
        - Sequence type checking (is_sequence_type)
        - Direct access to external library modules (structlog(), dependency_providers())
        - Structured logging configuration (configure_structlog())

    **USAGE**: Higher-level modules use FlextRuntime for utilities
        ```python
        from flext_core.runtime import FlextRuntime

        # Type guards using FlextConstants patterns
        if FlextRuntime.is_valid_email(user_input):
            process_email(user_input)

        # Access external libraries
        structlog_module = FlextRuntime.structlog()
        providers = FlextRuntime.dependency_providers()
        ```
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
                model_dump_method = getattr(obj, "model_dump")
                result = model_dump_method()
                if isinstance(result, dict):
                    return result
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                logging.getLogger(__name__).debug(
                    f"model_dump() serialization strategy failed: {e}"
                )

        # Strategy 2: Check for dict method
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                dict_method = getattr(obj, "dict")
                result = dict_method()
                if isinstance(result, dict):
                    return result
            except Exception as e:
                # Silent fallback for serialization strategy - log at debug level
                logging.getLogger(__name__).debug(
                    f"dict() serialization strategy failed: {e}"
                )

        # Strategy 3: Check for __dict__ attribute
        if hasattr(obj, "__dict__"):
            try:
                result = obj.__dict__
                return dict(result)
            except Exception as e:  # pragma: no cover
                # Silent fallback for serialization strategy - log at debug level
                # Extremely rare: __dict__ exists but dict() conversion fails
                logging.getLogger(__name__).debug(
                    f"__dict__ serialization strategy failed: {e}"
                )

        # Strategy 4: Check if already dict
        if isinstance(obj, dict):
            return obj

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
    def extract_generic_args(type_hint: object) -> tuple[Any, ...]:
        """Extract generic type arguments from a type hint.

        Args:
            type_hint: Type hint to extract args from

        Returns:
            Tuple of type arguments, empty tuple if no args

        """
        try:
            return typing.get_args(type_hint)
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

        processors: list[Any] = [
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
