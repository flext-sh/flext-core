"""FLEXT Core Constants - Single Unified Constants Class.

All FLEXT constants organized in one clean, structured class.
No backward compatibility layer - direct usage only.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final
from typing import NewType

# ==================================================================
# ENUMS - Core system enums
# ==================================================================


class FlextEnvironment(str, Enum):
    """Environment types for FLEXT applications."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class FlextLogLevel(str, Enum):
    """Log levels following Python logging standards."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class FlextResultStatus(str, Enum):
    """Result status types for FlextResult pattern."""

    SUCCESS = "success"
    FAILURE = "failure"


# ==================================================================
# TYPE ALIASES - Type-safe string aliases
# ==================================================================

ServiceName = NewType("ServiceName", str)
ConfigKey = NewType("ConfigKey", str)
EntityId = NewType("EntityId", str)
TraceId = NewType("TraceId", str)


# ==================================================================
# UNIFIED CONSTANTS CLASS - All constants in one place
# ==================================================================


class FlextConstants:
    """Unified constants class containing all FLEXT configuration.

    All constants are organized as class attributes within this class.
    This class should NEVER be instantiated - use class attributes.

    Usage:
        from flext_core.constants import FlextConstants

        # Access constants directly
        version = FlextConstants.VERSION
        env = FlextConstants.DEFAULT_ENVIRONMENT
        timeout = FlextConstants.DEFAULT_TIMEOUT
    """

    # ================================================================
    # PROJECT METADATA
    # ================================================================

    VERSION: Final[str] = "0.8.0"
    NAME: Final[str] = "flext-core"
    DESCRIPTION: Final[str] = "Enterprise Foundation Framework"

    # ================================================================
    # ENVIRONMENT CONSTANTS
    # ================================================================

    # Environment enum values
    ENV_DEVELOPMENT = FlextEnvironment.DEVELOPMENT
    ENV_TESTING = FlextEnvironment.TESTING
    ENV_STAGING = FlextEnvironment.STAGING
    ENV_PRODUCTION = FlextEnvironment.PRODUCTION

    # Default environment
    DEFAULT_ENVIRONMENT: Final[FlextEnvironment] = FlextEnvironment.DEVELOPMENT

    # ================================================================
    # LOGGING CONSTANTS
    # ================================================================

    # Log level enum values
    LOG_CRITICAL = FlextLogLevel.CRITICAL
    LOG_ERROR = FlextLogLevel.ERROR
    LOG_WARNING = FlextLogLevel.WARNING
    LOG_INFO = FlextLogLevel.INFO
    LOG_DEBUG = FlextLogLevel.DEBUG
    LOG_TRACE = FlextLogLevel.TRACE

    # Default log level
    DEFAULT_LOG_LEVEL: Final[FlextLogLevel] = FlextLogLevel.INFO

    # ================================================================
    # SYSTEM DEFAULTS
    # ================================================================

    DEFAULT_ENCODING: Final[str] = "utf-8"
    DEFAULT_TIMEOUT: Final[int] = 30
    DEFAULT_RETRY_COUNT: Final[int] = 3

    # ================================================================
    # SERVICE CONTAINER CONFIGURATION
    # ================================================================

    MAX_CONTAINER_SERVICES: Final[int] = 10000
    MAX_SERVICE_NAME_LENGTH: Final[int] = 255
    MIN_SERVICE_NAME_LENGTH: Final[int] = 1

    RESERVED_SERVICE_NAMES: Final[frozenset[str]] = frozenset(
        {
            "admin",
            "cache",
            "config",
            "container",
            "core",
            "flext",
            "health",
            "internal",
            "logger",
            "metrics",
            "system",
        },
    )

    # ================================================================
    # VALIDATION PATTERNS
    # ================================================================

    VALID_IDENTIFIER_PATTERN: Final[str] = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    VALID_SERVICE_NAME_PATTERN: Final[str] = r"^[a-zA-Z0-9_.-]+$"

    # Compiled regex patterns for performance
    IDENTIFIER_REGEX: Final[re.Pattern[str]] = re.compile(
        VALID_IDENTIFIER_PATTERN,
    )
    SERVICE_NAME_REGEX: Final[re.Pattern[str]] = re.compile(
        VALID_SERVICE_NAME_PATTERN,
    )

    # ================================================================
    # PERFORMANCE LIMITS
    # ================================================================

    MAX_NESTING_DEPTH: Final[int] = 100
    CACHE_TTL_SECONDS: Final[int] = 3600
    MAX_RETRY_ATTEMPTS: Final[int] = 5
    CONNECTION_POOL_SIZE: Final[int] = 20

    # ================================================================
    # RESULT STATUS CONSTANTS
    # ================================================================

    RESULT_SUCCESS = FlextResultStatus.SUCCESS
    RESULT_FAILURE = FlextResultStatus.FAILURE

    # ================================================================
    # VALIDATION METHODS
    # ================================================================

    @classmethod
    def is_valid_service_name(cls, name: str) -> bool:
        """Check if a service name is valid.

        Args:
            name: Service name to validate

        Returns:
            True if valid, False otherwise

        """
        if not (
            cls.MIN_SERVICE_NAME_LENGTH
            <= len(name)
            <= cls.MAX_SERVICE_NAME_LENGTH
        ):
            return False
        if name in cls.RESERVED_SERVICE_NAMES:
            return False
        return cls.SERVICE_NAME_REGEX.match(name) is not None

    @classmethod
    def is_valid_identifier(cls, identifier: str) -> bool:
        """Check if an identifier is valid Python identifier.

        Args:
            identifier: Identifier to validate

        Returns:
            True if valid, False otherwise

        """
        return cls.IDENTIFIER_REGEX.match(identifier) is not None

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    @classmethod
    def get_all_environments(cls) -> list[FlextEnvironment]:
        """Get all available environments."""
        return [
            cls.ENV_DEVELOPMENT,
            cls.ENV_TESTING,
            cls.ENV_STAGING,
            cls.ENV_PRODUCTION,
        ]

    @classmethod
    def get_all_log_levels(cls) -> list[FlextLogLevel]:
        """Get all available log levels."""
        return [
            cls.LOG_CRITICAL,
            cls.LOG_ERROR,
            cls.LOG_WARNING,
            cls.LOG_INFO,
            cls.LOG_DEBUG,
            cls.LOG_TRACE,
        ]

    # ================================================================
    # PREVENT INSTANTIATION - Use class attributes directly
    # ================================================================

    def __init__(self) -> None:
        """Prevent instantiation of FlextConstants.

        This class should not be instantiated. Use class attributes directly.

        Raises:
            TypeError: Always raised to prevent instantiation

        """
        msg = (
            "FlextConstants should not be instantiated - use class attributes"
        )
        raise TypeError(msg)


# ==================================================================
# EXPORTS - Clean public API
# ==================================================================

__all__ = [
    "ConfigKey",
    "EntityId",
    # Main constants class - PRIMARY INTERFACE
    "FlextConstants",
    # Enums
    "FlextEnvironment",
    "FlextLogLevel",
    "FlextResultStatus",
    # Type aliases
    "ServiceName",
    "TraceId",
]
