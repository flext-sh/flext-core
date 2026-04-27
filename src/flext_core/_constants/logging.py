"""FlextConstantsLogging - log level + async + log message constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsLogging:
    """SSOT for logging level enumeration and log message templates."""

    @unique
    class LogLevel(StrEnum):
        """Standard log levels."""

        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    @unique
    class LoggingOperation(StrEnum):
        """Canonical operation identifiers used in logging context failure reports."""

        BIND_CONTEXT = "bind context for scope"
        BIND_GLOBAL = "bind global context"
        CLEAR_GLOBAL = "clear global context"
        CLEAR_SCOPE = "clear scope"
        UNBIND_GLOBAL = "unbind global context"
        GET_CALLER_SOURCE = "get_caller_source_path"
        SHOULD_INCLUDE_STACK = "should_include_stack_trace"

    ASYNC_ENABLED: Final[bool] = True
    ASYNC_BLOCK_ON_FULL: Final[bool] = False

    MAX_FILE_SIZE: Final[int] = 10485760
    BACKUP_COUNT: Final[int] = 5

    # Workspace / path detection
    WORKSPACE_ROOT_MARKERS: Final[frozenset[str]] = frozenset({
        "pyproject.toml",
        ".git",
        "poetry.lock",
    })
    VENV_DIR_NAME: Final[str] = ".venv"
    MODULE_FRAME_NAME: Final[str] = "<module>"
    FRAME_SELF_KEY: Final[str] = "self"

    # Logger identity
    LOGGER_NAME_FLEXT_CORE: Final[str] = "flext_core"

    # Internal logging machinery path fragments — used to skip the logging stack
    LOGGING_INTERNAL_PATH_FRAGMENTS: Final[frozenset[str]] = frozenset({
        "flext_core/loggings.py",
        "flext_core/_utilities/logging_context.py",
        "flext_core/_utilities/logging_config.py",
        "flext_core/_utilities/logging_processors.py",
        "flext_core/_utilities/logging_observability.py",
    })

    # Exceptions caught by context binding / unbinding helpers
    CONTEXT_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
    )

    _TEMPLATE_REGISTERED: Final[str] = "Registered {subject}"
    LOG_REGISTERED_AUTO_DISCOVERY_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="auto-discovery handler"
    )
    LOG_REGISTERED_EVENT_SUBSCRIBER: Final[str] = _TEMPLATE_REGISTERED.format(
        subject="event subscriber"
    )
    LOG_REGISTERED_HANDLER: Final[str] = _TEMPLATE_REGISTERED.format(subject="handler")
    LOG_HANDLER_EXECUTION_FAILED: Final[str] = "Handler execution failed"
    LOG_HANDLER_PIPELINE_FAILURE: Final[str] = "Critical handler pipeline failure"
    LOG_TRACKED_OPERATION_EXPECTED_EXCEPTION: Final[str] = (
        "Tracked operation raised expected exception"
    )
    LOG_SERVICE_REGISTRATION_FAILED: Final[str] = "Service registration failed"
    LOG_INTERNAL_OPERATION_FAILED: Final[str] = "Internal logger operation failed"
    LOG_CONTEXT_REMOVAL_FAILED: Final[str] = "Failed to validate context after removal"
