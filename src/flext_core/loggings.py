"""Structured logging utilities enabling the context-first pillar for 1.0.0.

The module mirrors the expectations in ``README.md`` and
``docs/architecture.md`` by binding log records to ``FlextContext`` metadata
and providing the default processors shared across FLEXT packages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import logging
import os
import platform
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import ClassVar, Literal, Self, TypedDict, cast

import structlog
from structlog.typing import EventDict, Processor

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextLogger(FlextProtocols.Infrastructure.LoggerProtocol):
    """High-performance structured logger with comprehensive context management.

    Optimized implementation with Pydantic validation, centralized context management,
    FlextConfig integration, and explicit protocol compliance for 1.0.0 stability.

    Key optimizations:
    - Centralized _ContextManager for all context operations
    - Pydantic models for parameter validation
    - FlextConfig singleton as single source of truth
    - FlextConstants for all default values
    - Explicit LoggerProtocol implementation
    - FlextResult patterns for error handling
    """

    # Type declarations for enhanced static analysis
    _local: threading.local
    _instances: ClassVar[dict[str, FlextLogger]] = {}
    _configured: ClassVar[bool] = False
    _global_correlation_id: ClassVar[str | None] = None
    _service_info: ClassVar[FlextTypes.Core.Dict] = {}
    _request_context: ClassVar[FlextTypes.Core.Dict] = {}

    # Context manager for optimized operations
    _context_manager: _ContextManager

    # Thread-local storage for request context
    _local = threading.local()

    # Instance attributes type declarations
    _name: str
    _level: FlextTypes.Config.LogLevel
    _environment: FlextTypes.Config.Environment

    def __new__(
        cls,
        name: str,
        _level: FlextTypes.Config.LogLevel | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        *,
        _force_new: bool = False,
    ) -> Self:
        """Create or return cached logger instance."""
        # Check if this is a bind() call that needs a new instance
        force_new = _force_new

        if not force_new and name in cls._instances:
            # Type cast needed for proper Self return type
            return cast("Self", cls._instances[name])

        instance = super().__new__(cls)
        if not force_new:  # Only cache if not forced new
            cls._instances[name] = instance
        return instance

    def __init__(
        self,
        name: str,
        _level: FlextTypes.Config.LogLevel | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        *,
        _force_new: bool = False,  # Accept but ignore this parameter
    ) -> None:
        """Initialize structured logger instance using Pydantic validation and FlextConfig singleton."""
        # Validate initialization parameters using Pydantic model
        try:
            init_model = FlextModels.LoggerInitializationModel(
                name=name,
                log_level=_level or "INFO",
            )
        except Exception as e:
            warnings.warn(
                f"Logger initialization validation failed: {e}. Using provided values as-is.",
                UserWarning,
                stacklevel=2,
            )
            # Use parameters directly if validation fails
            init_model = None

        if not self._is_configured():
            # Always (re)configure structlog to ensure processors reflect stored config
            global_config = FlextConfig.get_global_instance()
            # Extract configuration from FlextConfig singleton - single source of truth
            config_kwargs = {
                "log_level": global_config.log_level,
                "json_output": getattr(global_config, "json_output", None),
                "include_source": getattr(
                    global_config,
                    "include_source",
                    FlextConstants.Logging.INCLUDE_SOURCE,
                ),
                "structured_output": getattr(
                    global_config,
                    "structured_output",
                    FlextConstants.Logging.STRUCTURED_OUTPUT,
                ),
            }

            # Call configure with proper typed arguments
            log_level = str(
                config_kwargs.get("log_level", FlextConstants.Logging.DEFAULT_LEVEL)
            )
            json_output = config_kwargs.get("json_output")
            include_source = bool(
                config_kwargs.get(
                    "include_source", FlextConstants.Logging.INCLUDE_SOURCE
                )
            )
            structured_output = bool(
                config_kwargs.get(
                    "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
                )
            )

            # Type-safe configure call
            json_output_typed: bool | None = (
                None if json_output is None else bool(json_output)
            )
            type(self).configure(
                log_level=log_level,
                json_output=json_output_typed,
                include_source=include_source,
                structured_output=structured_output,
            )

        # Use validated model values if available, otherwise use original parameters
        validated_name = init_model.name if init_model else name
        validated_level = init_model.log_level if init_model else _level
        validated_service_name = _service_name  # Not in model, use parameter directly
        validated_service_version = (
            _service_version  # Not in model, use parameter directly
        )
        validated_correlation_id = (
            _correlation_id  # Not in model, use parameter directly
        )

        self._name = validated_name
        # Load configuration early so .env and FLEXT_* vars are available
        config = FlextConfig.get_global_instance()

        # Resolve log level with strict precedence and deterministic behavior
        valid_levels = FlextConstants.Logging.VALID_LEVELS

        # 0) Detect test environment deterministically (pytest session or explicit config)
        is_pytest = (
            os.getenv(FlextConstants.Platform.ENV_PYTEST_CURRENT_TEST) is not None
            or "pytest" in sys.modules
        )
        is_test_env = (str(config.environment).lower() == "test") or is_pytest

        resolved_level: str | None = None

        # 1) Explicit parameter takes precedence when valid
        if isinstance(validated_level, str) and validated_level:
            cand = validated_level.upper()
            if cand in valid_levels:
                resolved_level = cand

        # 2) Testing default: prefer WARNING in test sessions when no explicit level
        if resolved_level is None and is_test_env:
            resolved_level = FlextConstants.Logging.WARNING

        # 3) Environment variable override when valid (after loading .env via config)
        if resolved_level is None:
            # Check project-specific environment variable first
            project_specific_var = self._get_project_specific_env_var("LOG_LEVEL")
            if project_specific_var:
                env_level = os.getenv(project_specific_var)
                env_level_upper = (
                    env_level.upper() if isinstance(env_level, str) else None
                )
                if env_level_upper in valid_levels:
                    resolved_level = env_level_upper

            # Fallback to global FLEXT_LOG_LEVEL
            if resolved_level is None:
                env_level = os.getenv("FLEXT_LOG_LEVEL")
                env_level_upper = (
                    env_level.upper() if isinstance(env_level, str) else None
                )
                if env_level_upper in valid_levels:
                    resolved_level = env_level_upper

        # 4) Configuration/defaults from FlextConfig singleton
        if resolved_level is None:
            cfg_level = str(config.log_level).upper()
            resolved_level = (
                cfg_level
                if cfg_level in valid_levels
                else FlextConstants.Logging.DEFAULT_LEVEL
            )

        self._level = self._validated_log_level(resolved_level)

        # Use environment from configuration singleton for consistency
        # Environment is already typed correctly in FlextConfig
        self._environment = cast("FlextTypes.Config.Environment", config.environment)

        # Initialize service context using validated parameters
        self._service_name = validated_service_name or self._extract_service_name()
        self._service_version = validated_service_version or FlextConstants.Core.VERSION

        # Set up performance tracking
        self._start_time = time.time()

        # Instance-level correlation ID (can override global)
        context_id = FlextContext.Correlation.get_correlation_id()

        self._correlation_id = (
            validated_correlation_id
            or self._global_correlation_id
            or context_id  # Check global context
            or FlextUtilities.Generators.generate_correlation_id()
        )

        # Set up structured logger with enriched context (import locally)

        self._structlog_logger = structlog.get_logger(validated_name)

        # Initialize persistent context
        self._persistent_context: FlextTypes.Core.Dict = {
            "service": {
                "name": self._service_name,
                "version": self._service_version,
                "instance_id": f"{platform.node()}-{os.getpid()}",
                "environment": (
                    os.environ.get("ENVIRONMENT")
                    or os.environ.get("ENV")
                    or "development"
                ).lower(),
            },
            "system": {
                "hostname": platform.node(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
            },
        }

        # Initialize centralized context manager - optimized context operations
        self._context_manager = self._ContextManager(self)

    def __eq__(self, other: object) -> bool:
        """Allow comparison with both logger objects and their string repr.

        - If comparing to another FlextLogger, enforce identity semantics.
        - If comparing to a string, compare with str(self) for test compatibility.

        Args:
            other: Object to compare with

        Returns:
            bool: True if equal, False otherwise

        """
        if isinstance(other, FlextLogger):
            return self is other
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        """Provide a stable hash consistent with __eq__ semantics.

        Uses the hash of the string representation to ensure that when a
        string equals this logger via __eq__, their hashes also match.

        Returns:
            int: Hash value based on string representation

        """
        return hash(str(self))

    def _extract_service_name(self) -> str:
        """Extract service name from logger name or environment.

        Returns:
            str: Service name extracted from environment or logger name

        """
        if service_name := os.environ.get("SERVICE_NAME"):
            return service_name

        # Extract from module name
        min_parts = 2
        parts = self._name.split(".")
        if len(parts) >= min_parts and parts[0].startswith("flext_"):
            return parts[0].replace("_", "-")

        return "flext-core"

    def _get_project_specific_env_var(self, suffix: str) -> str | None:
        """Get project-specific environment variable name for the current service.

        Args:
            suffix: Suffix to append to the environment variable name

        Returns:
            str | None: Environment variable name or None if not applicable

        """
        service_name = self._extract_service_name()
        if service_name.startswith("flext-"):
            # Convert flext-core -> FLEXT_CORE, flext-ldap -> FLEXT_LDAP, etc.
            env_name = service_name.upper().replace("-", "_")
            return f"{env_name}_{suffix}"
        return None

    def _get_current_timestamp(self) -> str:
        """Get current ISO timestamp.

        Returns:
            str: Current timestamp in ISO format

        """
        return datetime.now(UTC).isoformat()

    def _sanitize_context(self, context: dict[str, object]) -> dict[str, object]:
        """Sanitize context by redacting sensitive data.

        Args:
            context: Context dictionary to sanitize

        Returns:
            dict[str, object]: Sanitized context dictionary

        """
        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "key",
            "auth",
            "authorization",
            "credential",
            "private",
            "api_key",
            "access_token",
            "refresh_token",
            "session_id",
            "cookie",
        }

        sanitized: dict[str, object] = {}
        for key, value in context.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(
                    cast("dict[str, object]", value)
                )
            else:
                sanitized[key] = value

        return sanitized

    def _build_log_entry(
        self,
        level: str,
        message: str,
        context: Mapping[str, object] | None = None,
        error: Exception | str | None = None,
        duration_ms: float | None = None,
    ) -> FlextLogger.LogEntry:
        """Build structured log entry.

        Args:
            level: Log level
            message: Log message
            context: Optional context dictionary
            error: Optional error information
            duration_ms: Optional duration in milliseconds

        Returns:
            FlextLogger.LogEntry: Structured log entry

        """
        # Start with timestamp and correlation
        entry: FlextLogger.LogEntry = {
            "timestamp": self._get_current_timestamp(),
            "level": level.upper(),
            "message": str(message),
            "logger": self._name,
            "correlation_id": self._correlation_id,
            # Always include these containers for type predictability in tests
            "request": {},
            "context": {},
            "service": {},
            "system": {},
            "execution": {},
        }

        # Add service and system context
        # Populate service and system from persistent context without broad update
        service_ctx = self._persistent_context.get("service")
        system_ctx = self._persistent_context.get("system")

        if isinstance(service_ctx, dict) and service_ctx:
            entry["service"] = dict(cast("dict[str, object]", service_ctx))
        if isinstance(system_ctx, dict) and system_ctx:
            entry["system"] = dict(cast("dict[str, object]", system_ctx))

        # Add request context if available
        request_context = cast(
            "dict[str, object] | None", getattr(self._local, "request_context", None)
        )
        # Always set request (may be empty)
        entry["request"] = (
            dict(request_context) if isinstance(request_context, dict) else {}
        )

        # Add permanent context if available
        permanent_context = getattr(self, "_persistent_context", None)
        if isinstance(permanent_context, dict) and permanent_context:
            entry["permanent"] = dict(cast("dict[str, object]", permanent_context))

        # Add performance metrics
        if duration_ms is not None:
            entry["performance"] = {
                "duration_ms": round(duration_ms, 3),
                "timestamp": self._get_current_timestamp(),
            }

        # Add error details if present
        if error is not None:
            if isinstance(error, Exception):
                entry["error"] = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "details": getattr(error, "args", ()),
                }
            else:
                entry["error"] = {"message": str(error)}

        # Add additional context if provided
        if isinstance(context, dict) and context:
            entry["context"] = dict(context)

        return entry

    def _get_calling_function(self) -> str:
        """Get the name of the calling function.

        Returns:
            str: Name of the calling function

        """
        try:
            frame = inspect.currentframe()
            # Skip internal logging frames
            for _ in range(4):
                if frame is None:
                    break
                frame = frame.f_back
            if frame is not None:
                return frame.f_code.co_name
            return "unknown"
        except (AttributeError, ValueError):
            return "unknown"

    def _get_calling_line(self) -> int:
        """Get the line number of the calling code.

        Returns:
            int: Line number of the calling code

        """
        try:
            frame = inspect.currentframe()
            # Skip internal logging frames
            for _ in range(4):
                if frame is None:
                    break
                frame = frame.f_back
            if frame is not None:
                return frame.f_lineno
            return 0
        except (AttributeError, ValueError):
            return 0

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for request tracing - optimized through context manager."""
        result = self._context_manager.set_correlation_id(correlation_id)
        if result.is_failure:
            warnings.warn(
                f"Failed to set correlation ID: {result.error}",
                UserWarning,
                stacklevel=2,
            )

    def set_request_context(self, **context: object) -> None:
        """Set request-specific context - optimized through context manager."""
        # Extract known fields from context
        model_kwargs: dict[str, object] = {}
        if "request_id" in context:
            model_kwargs["request_id"] = str(context["request_id"])
        if "method" in context and context["method"] is not None:
            model_kwargs["method"] = str(context["method"])
        if "path" in context and context["path"] is not None:
            model_kwargs["path"] = str(context["path"])
        if "user_id" in context and context["user_id"] is not None:
            model_kwargs["user_id"] = str(context["user_id"])
        if "endpoint" in context and context["endpoint"] is not None:
            model_kwargs["endpoint"] = str(context["endpoint"])
        if "headers" in context and isinstance(context["headers"], dict):
            headers_dict = cast("dict[object, object]", context["headers"])
            model_kwargs["headers"] = {str(k): str(v) for k, v in headers_dict.items()}
        if "query_params" in context and isinstance(context["query_params"], dict):
            query_dict = cast("dict[object, object]", context["query_params"])
            model_kwargs["query_params"] = {
                str(k): str(v) for k, v in query_dict.items()
            }

        # Convert model_kwargs to proper types for LoggerRequestContextModel
        request_id = str(model_kwargs.get("request_id", ""))
        method = str(model_kwargs["method"]) if model_kwargs.get("method") else None
        path = str(model_kwargs["path"]) if model_kwargs.get("path") else None
        headers = cast("dict[str, str]", model_kwargs.get("headers", {}))
        query_params = cast("dict[str, str]", model_kwargs.get("query_params", {}))
        correlation_id = (
            str(model_kwargs["correlation_id"])
            if model_kwargs.get("correlation_id")
            else None
        )
        user_id = str(model_kwargs["user_id"]) if model_kwargs.get("user_id") else None
        endpoint = str(model_kwargs["endpoint"]) if model_kwargs.get("endpoint") else None

        model = FlextModels.LoggerRequestContextModel(
            request_id=request_id,
            method=method,
            path=path,
            headers=headers,
            query_params=query_params,
            correlation_id=correlation_id,
            user_id=user_id,
            endpoint=endpoint,
        )
        result = self._context_manager.set_request_context(model)
        if result.is_failure:
            warnings.warn(
                f"Failed to set request context: {result.error}",
                UserWarning,
                stacklevel=2,
            )

    def clear_request_context(self) -> None:
        """Clear request context - optimized through context manager."""
        result = self._context_manager.clear_request_context()
        if result.is_failure:
            warnings.warn(
                f"Failed to clear request context: {result.error}",
                UserWarning,
                stacklevel=2,
            )

    def bind(self, **context: object) -> FlextLogger:
        """Create logger instance with bound context - optimized through context manager.

        Args:
            **context: Context data to bind

        Returns:
            FlextLogger: New logger instance with bound context

        """
        model = FlextModels.LoggerContextBindingModel(
            logger_name=self._name,
            context_data=context,
            bind_type="temporary",
            clear_existing=False,
        )
        result = self._context_manager.bind_context(model)
        if result.is_failure:
            warnings.warn(
                f"Failed to bind context: {result.error}", UserWarning, stacklevel=2
            )
            # Fallback to simplified bind method
            bound_logger = FlextLogger(
                name=self._name,
                _level=self._level,
                _service_name=getattr(self, "_service_name", None),
                _service_version=getattr(self, "_service_version", None),
                _correlation_id=getattr(self, "_correlation_id", None),
                _force_new=True,
            )

            # Copy existing request context
            if hasattr(self._local, "request_context"):
                bound_logger.set_request_context(**self._local.request_context)

                # Copy existing persistent context
                if hasattr(self, "_persistent_context"):
                    bound_logger._persistent_context = self._persistent_context.copy()

            # Add new bound context
            bound_logger.set_request_context(**context)

            return bound_logger

        return result.unwrap()

    def set_context(
        self,
        context_dict: FlextTypes.Core.Dict | None = None,
        *,
        replace_existing: bool | None = None,
        merge_strategy: Literal["replace", "update", "merge_deep"] | None = None,
        **context: object,
    ) -> None:
        """Set persistent context data via the validated model and context manager."""
        # Merge context_dict and kwargs
        final_context: dict[str, object] = {}
        if context_dict is not None:
            final_context.update(context_dict)
        final_context.update(context)

        # Resolve merge behaviour while preserving historic defaults
        if replace_existing is not None:
            replace_flag = replace_existing
        elif merge_strategy is not None:
            replace_flag = merge_strategy == "replace"
        else:
            replace_flag = context_dict is not None

        resolved_merge_strategy = (
            merge_strategy if merge_strategy is not None else ("replace" if replace_flag else "update")
        )

        # Build permanent context model populated from configuration/attributes
        config = FlextConfig.get_global_instance()
        app_name = getattr(config, "app_name", None) or getattr(self, "_service_name", self._name)
        app_version = getattr(config, "version", None) or getattr(
            self, "_service_version", FlextConstants.Core.VERSION
        )
        environment_value = str(
            getattr(config, "environment", getattr(self, "_environment", "development"))
        )

        env_map = {
            "dev": "development",
            "development": "development",
            "local": "development",
            "test": "testing",
            "testing": "testing",
            "stage": "staging",
            "staging": "staging",
            "prod": "production",
            "production": "production",
        }
        normalized_environment = env_map.get(environment_value.lower(), environment_value.lower())
        if normalized_environment not in {"development", "testing", "staging", "production"}:
            normalized_environment = "development"

        try:
            model = FlextModels.LoggerPermanentContextModel(
                app_name=str(app_name),
                app_version=str(app_version),
                environment=normalized_environment,
                host=platform.node(),
                permanent_context=final_context,
                replace_existing=replace_flag,
                merge_strategy=resolved_merge_strategy,
            )
        except Exception as e:
            warnings.warn(
                f"Failed to validate persistent context: {e}",
                UserWarning,
                stacklevel=2,
            )
            return

        result = self._context_manager.set_persistent_context(model)
        if result.is_failure:
            warnings.warn(
                f"Failed to set persistent context: {result.error}",
                UserWarning,
                stacklevel=2,
            )

    def with_context(self, **context: object) -> FlextLogger:
        """Create logger instance with additional context - optimized through context manager.

        Args:
            **context: Context data to add

        Returns:
            FlextLogger: New logger instance with additional context

        """
        return self.bind(**context)

    def start_operation(self, operation_name: str, **context: object) -> str:
        """Start tracking operation with metrics.

        Args:
            operation_name: Name of the operation to track
            **context: Context data for the operation

        Returns:
            str: Operation ID for tracking

        """
        operation_id = f"op_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Store operation start time
        if not hasattr(self._local, "operations"):
            self._local.operations = {}
            self._local.operations[operation_id] = {
                "name": operation_name,
                "start_time": start_time,
                "context": context,
            }

        self.info(
            f"Operation started: {operation_name}",
            operation_id=operation_id,
            operation_name=operation_name,
            **context,
        )

        return operation_id

    def complete_operation(
        self,
        operation_id: str,
        *,
        success: bool = True,
        **context: object,
    ) -> None:
        """Complete operation tracking with metrics."""
        if not hasattr(self._local, "operations"):
            return

        operation_info = self._local.operations.get(operation_id)
        if not operation_info:
            return

        duration_ms = (time.time() - operation_info["start_time"]) * 1000

        log_context = {
            "operation_id": operation_id,
            "operation_name": operation_info["name"],
            "success": success,
            "duration_ms": round(duration_ms, 3),
            **operation_info["context"],
            **context,
        }

        if success:
            self.info(f"Operation completed: {operation_info['name']}", **log_context)
        else:
            self.error(f"Operation failed: {operation_info['name']}", **log_context)

        # Clean up
        del self._local.operations[operation_id]

    # LoggerProtocol implementation - Standard logging methods with enhanced context
    def trace(self, message: str, *args: object, **context: object) -> None:
        """Log trace message - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        entry = self._build_log_entry("TRACE", formatted_message, context)
        self._structlog_logger.debug(
            formatted_message,
            **entry,
        )  # Use debug since structlog doesn't have trace

    def debug(self, message: str, *args: object, **context: object) -> None:
        """Log debug message - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            self._structlog_logger.debug(formatted_message, **context)
        else:
            entry = self._build_log_entry(
                FlextConstants.Logging.DEBUG, formatted_message, context
            )
            self._structlog_logger.debug(formatted_message, **entry)

    def info(self, message: str, *args: object, **context: object) -> None:
        """Log info message - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            self._structlog_logger.info(formatted_message, **context)
        else:
            entry = self._build_log_entry(
                FlextConstants.Logging.INFO, formatted_message, context
            )
            self._structlog_logger.info(formatted_message, **entry)

    def warning(self, message: str, *args: object, **context: object) -> None:
        """Log warning message - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            self._structlog_logger.warning(formatted_message, **context)
        else:
            entry = self._build_log_entry(
                FlextConstants.Logging.WARNING, formatted_message, context
            )
            self._structlog_logger.warning(formatted_message, **entry)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message with context and error details - LoggerProtocol implementation."""
        # Extract special parameters from kwargs with proper typing
        args = kwargs.pop("args", ())
        error_obj = kwargs.pop("error", None)
        context = kwargs

        # Convert error to proper type
        error: Exception | str | None = None
        if error_obj is not None:
            error = error_obj if isinstance(error_obj, Exception) else str(error_obj)

        # Handle args formatting
        formatted_message = message % args if args else message

        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            if error:
                context["error"] = str(error)
            self._structlog_logger.error(formatted_message, **context)
        else:
            entry = self._build_log_entry(
                FlextConstants.Config.LogLevel.ERROR,
                formatted_message,
                context,
                error,
            )
            self._structlog_logger.error(formatted_message, **entry)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message with context and error details - LoggerProtocol implementation."""
        # Extract special parameters from kwargs with proper typing
        args = kwargs.pop("args", ())
        error_obj = kwargs.pop("error", None)
        context = kwargs

        # Convert error to proper type
        error: Exception | str | None = None
        if error_obj is not None:
            error = error_obj if isinstance(error_obj, Exception) else str(error_obj)

        # Handle args formatting
        formatted_message = message % args if args else message

        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            if error:
                context["error"] = str(error)
            self._structlog_logger.critical(formatted_message, **context)
        else:
            entry = self._build_log_entry(
                FlextConstants.Config.LogLevel.CRITICAL,
                formatted_message,
                context,
                error,
            )
            self._structlog_logger.critical(formatted_message, **entry)

    def exception(self, message: str, *args: object, **context: object) -> None:
        """Log exception with stack trace and context - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        exc_info = sys.exc_info()
        error = exc_info[1] if isinstance(exc_info[1], Exception) else None
        entry = self._build_log_entry(
            FlextConstants.Config.LogLevel.ERROR, formatted_message, context, error
        )
        self._structlog_logger.error(formatted_message, **entry)

    @classmethod
    def configure(
        cls,
        log_level: str | None = None,
        *,
        json_output: bool | None = None,
        include_source: bool | None = None,
        structured_output: bool | None = None,
        log_verbosity: str | None = None,
    ) -> FlextResult[None]:
        """Configure the FlextLogger globally with Pydantic validation.

        Uses LoggerConfigurationModel for parameter validation and FlextConfig/FlextConstants
        as source of truth for defaults.

        Args:
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_output: Whether to output JSON format
            include_source: Whether to include source code location
            structured_output: Whether to use structured logging
            log_verbosity: Console output verbosity (compact|detailed|full)

        Returns:
            FlextResult[None]: Success or failure with error details

        """
        # Create configuration model with defaults
        config_model = FlextModels.LoggerConfigurationModel(
            log_level=log_level or "INFO",
            json_output=json_output,
            include_source=include_source or FlextConstants.Logging.INCLUDE_SOURCE,
            structured_output=structured_output
            or FlextConstants.Logging.STRUCTURED_OUTPUT,
            log_verbosity=log_verbosity or FlextConstants.Logging.VERBOSITY,
        )

        # Reset if already configured (allow reconfiguration)
        if cls._configured and structlog.is_configured():
            structlog.reset_defaults()
            cls._configured = False

        try:
            # Configure structlog with validated parameters

            processors: list[Processor] = [
                structlog.stdlib.filter_by_level,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
            ]

            # Add source location if enabled
            if config_model.include_source:
                processors.append(
                    structlog.processors.CallsiteParameterAdder(
                        parameters=[
                            structlog.processors.CallsiteParameter.FILENAME,
                            structlog.processors.CallsiteParameter.FUNC_NAME,
                            structlog.processors.CallsiteParameter.LINENO,
                        ]
                    )
                )

            # Configure output format based on json_output setting
            if config_model.json_output:
                processors.append(structlog.processors.JSONRenderer())
            # Use structured console output with verbosity control
            elif config_model.log_verbosity == "compact":
                processors.append(
                    structlog.dev.ConsoleRenderer(colors=True, repr_native_str=False)
                )
            elif config_model.log_verbosity == "detailed":
                processors.append(
                    structlog.dev.ConsoleRenderer(
                        colors=True,
                        repr_native_str=False,
                        exception_formatter=structlog.dev.plain_traceback,
                    )
                )
            else:  # full verbosity
                processors.append(
                    structlog.dev.ConsoleRenderer(
                        colors=True,
                        repr_native_str=False,
                        exception_formatter=structlog.dev.rich_traceback,
                    )
                )

            # Configure structlog
            structlog.configure(
                processors=processors,
                wrapper_class=structlog.stdlib.BoundLogger,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

            # Set root logger level

            logging.basicConfig(
                level=getattr(logging, config_model.log_level.upper(), logging.INFO)
            )

            # Store configuration
            cls._configured = True

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Failed to configure structlog: {e}")

    @staticmethod
    def _add_correlation_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add correlation ID to log entries.

        Args:
            _logger: Logger instance (unused)
            _method_name: Method name (unused)
            event_dict: Event dictionary to modify

        Returns:
            EventDict: Modified event dictionary with correlation ID

        """
        if FlextLogger._global_correlation_id:
            event_dict["correlation_id"] = FlextLogger._global_correlation_id
        return event_dict

    @staticmethod
    def _add_performance_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add metadata to log entries.

        Args:
            _logger: Logger instance (unused)
            _method_name: Method name (unused)
            event_dict: Event dictionary to modify

        Returns:
            EventDict: Modified event dictionary with metadata

        """
        event_dict["@metadata"] = {
            "processor": "flext_logging",
            "version": FlextConstants.Core.VERSION,
            "processed_at": datetime.now(UTC).isoformat(),
        }
        return event_dict

    @staticmethod
    def _sanitize_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Sanitize sensitive data from log entries.

        Args:
            _logger: Logger instance (unused)
            _method_name: Method name (unused)
            event_dict: Event dictionary to sanitize

        Returns:
            EventDict: Sanitized event dictionary

        """
        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "key",
            "auth",
            "authorization",
            "credential",
            "private",
            "api_key",
        }

        for key in list(event_dict.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                event_dict[key] = "[REDACTED]"

        return event_dict

    @staticmethod
    def _create_enhanced_console_renderer(
        verbosity: str = FlextConstants.Logging.VERBOSITY,
    ) -> FlextLogger._ConsoleRenderer:
        """Create enhanced console renderer with configurable verbosity using FlextConstants.

        Args:
            verbosity: Verbosity level for the renderer

        Returns:
            FlextLogger._ConsoleRenderer: Enhanced console renderer instance

        """
        return FlextLogger._ConsoleRenderer(verbosity=verbosity)

    @classmethod
    def set_global_correlation_id(cls, correlation_id: str | None) -> None:
        """Set global correlation ID."""
        cls._global_correlation_id = correlation_id

    @classmethod
    def get_global_correlation_id(cls) -> str | None:
        """Get global correlation ID.

        Returns:
            str | None: Global correlation ID or None if not set

        """
        return cls._global_correlation_id

    def get_correlation_id(self) -> str | None:
        """Get instance correlation ID.

        Returns:
            str | None: Instance correlation ID or None if not set

        """
        return self._correlation_id

    def set_correlation_id_internal(self, correlation_id: str) -> None:
        """Set instance correlation ID (internal use).

        Args:
            correlation_id: Correlation ID to set

        """
        self._correlation_id = correlation_id

    def get_local_storage(self) -> threading.local:
        """Get thread-local storage.

        Returns:
            threading.local: Thread-local storage instance

        """
        return self._local

    def get_persistent_context(self) -> dict[str, object]:
        """Get persistent context dictionary.

        Returns:
            dict[str, object]: Persistent context dictionary

        """
        if not hasattr(self, "_persistent_context"):
            self._persistent_context = {}
        return self._persistent_context

    def set_persistent_context_dict(self, context: dict[str, object]) -> None:
        """Set persistent context dictionary (internal use).

        Args:
            context: Context dictionary to set

        """
        self._persistent_context = context

    def get_logger_attributes(self) -> dict[str, object]:
        """Get logger attributes for binding.

        Returns:
            dict[str, object]: Logger attributes dictionary

        """
        return {
            "name": self._name,
            "level": self._level,
            "service_name": getattr(self, "_service_name", None),
            "service_version": getattr(self, "_service_version", None),
            "correlation_id": getattr(self, "_correlation_id", None),
        }

    @classmethod
    def get_configuration(cls) -> dict[str, object]:
        """Get current logging configuration from FlextConfig singleton.

        Returns:
            Dictionary with current configuration settings from FlextConfig

        """
        global_config = FlextConfig.get_global_instance()
        return {
            "log_level": global_config.log_level,
            "json_output": getattr(global_config, "json_output", None),
            "include_source": getattr(
                global_config, "include_source", FlextConstants.Logging.INCLUDE_SOURCE
            ),
            "structured_output": getattr(
                global_config,
                "structured_output",
                FlextConstants.Logging.STRUCTURED_OUTPUT,
            ),
            "log_verbosity": getattr(
                global_config, "log_verbosity", FlextConstants.Logging.VERBOSITY
            ),
        }

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured.

        Returns:
            bool: True if logging is configured, False otherwise.

        """
        return cls._configured

    class LogEntry(TypedDict, total=False):
        """Typed structure for structured log entries.

        All fields are optional to support flexible logging scenarios.
        """

        # Core fields
        message: str
        level: str
        timestamp: str
        logger: str
        correlation_id: str | None
        context: FlextTypes.Core.Dict

        # Optional structured fields used in actual logging
        logger_name: str
        module: str
        function: str
        line_number: int
        request: FlextTypes.Core.Dict
        service: FlextTypes.Core.Dict
        system: FlextTypes.Core.Dict
        execution: FlextTypes.Core.Dict
        permanent: FlextTypes.Core.Dict
        performance: FlextTypes.Core.Dict
        error: FlextTypes.Core.Dict

    class _ConsoleRenderer:
        """Enhanced console renderer with hierarchical formatting - nested helper class."""

        def __init__(self, verbosity: str = FlextConstants.Logging.VERBOSITY) -> None:
            """Initialize renderer with specified verbosity level using FlextConstants."""
            self.verbosity = verbosity.lower()

            # Color codes for different log levels
            self._level_colors = {
                "critical": "\033[91;1m",  # Bright red, bold
                "error": "\033[91m",  # Red
                "warning": "\033[93m",  # Yellow
                "info": "\033[92m",  # Green
                "debug": "\033[94m",  # Blue
                "trace": "\033[95m",  # Magenta
            }

            # Color codes for different information types
            self._info_colors = {
                "service": "\033[36m",  # Cyan
                "context": "\033[95m",  # Magenta
                "execution": "\033[94m",  # Blue
                "system": "\033[90m",  # Dark gray
                "correlation": "\033[33m",  # Yellow
                "reset": "\033[0m",  # Reset
            }

        def __call__(
            self, _logger: logging.Logger, _method_name: str, event_dict: EventDict
        ) -> str:
            """Render log entry with hierarchical formatting.

            Args:
                _logger: Logger instance (unused)
                _method_name: Method name (unused)
                event_dict: Event dictionary to render

            Returns:
                str: Formatted log entry string

            """
            # Convert EventDict to dict[str, object] for type safety
            typed_event_dict: dict[str, object] = dict(event_dict)
            return self._format_event(typed_event_dict)

        def _format_event(self, event_dict: FlextTypes.Core.Dict) -> str:
            """Format event dictionary into hierarchical log output.

            Args:
                event_dict: Event dictionary to format

            Returns:
                str: Formatted log output string

            """
            # Extract core information with proper type casting
            timestamp = str(event_dict.get("@timestamp", ""))
            level = str(
                event_dict.get("level", FlextConstants.Config.LogLevel.INFO)
            ).upper()
            logger_name = str(event_dict.get("logger_name", ""))
            message = str(event_dict.get("event", ""))
            correlation_id = str(event_dict.get("correlation_id", ""))

            # Extract service info with type safety
            service_info = event_dict.get("service", {})
            if isinstance(service_info, dict):
                service_name = str(
                    cast("dict[str, object]", service_info).get("name", logger_name)
                )
            else:
                service_name = logger_name

            # Format main log line based on verbosity
            if self.verbosity == "compact":
                return self._format_compact(
                    timestamp,
                    level,
                    service_name,
                    message,
                    correlation_id=correlation_id,
                )
            if self.verbosity == "detailed":
                return self._format_detailed(
                    event_dict,
                    timestamp,
                    level,
                    service_name,
                    message,
                    correlation_id=correlation_id,
                )
            # full
            return self._format_full(
                event_dict,
                timestamp,
                level,
                service_name,
                message,
                correlation_id=correlation_id,
            )

        def _format_compact(
            self,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            *,
            correlation_id: str,
        ) -> str:
            """Format compact log entry.

            Args:
                timestamp: Timestamp string
                level: Log level string
                service_name: Service name string
                message: Log message string
                correlation_id: Correlation ID string

            Returns:
                str: Formatted compact log entry

            """
            # Clean timestamp (remove microseconds and timezone info for readability)
            clean_timestamp = (
                timestamp.split(".", maxsplit=1)[0] + "Z" if timestamp else ""
            )

            level_color = self._level_colors.get(level.lower(), "")
            reset = self._info_colors["reset"]
            correlation_color = self._info_colors["correlation"]

            # Format: timestamp [LEVEL] [service] message [correlation_id]
            correlation_part = (
                f" {correlation_color}[{correlation_id}]{reset}"
                if correlation_id
                else ""
            )

            return f"{clean_timestamp} {level_color}[{level}]{reset} [{service_name}] {message}{correlation_part}"

        def _format_detailed(
            self,
            event_dict: FlextTypes.Core.Dict,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            *,
            correlation_id: str,
        ) -> str:
            """Format detailed log entry with context tree.

            Args:
                event_dict: Event dictionary
                timestamp: Timestamp string
                level: Log level string
                service_name: Service name string
                message: Log message string
                correlation_id: Correlation ID string

            Returns:
                str: Formatted detailed log entry

            """
            lines: list[str] = []

            # Main line - fix the method call to use keyword argument
            main_line = self._format_compact(
                timestamp, level, service_name, message, correlation_id=correlation_id
            )
            lines.append(main_line)

            # Add context information if available
            context_raw = event_dict.get("context", {})
            execution_raw = event_dict.get("execution", {})

            context_color = self._info_colors["context"]
            execution_color = self._info_colors["execution"]
            reset = self._info_colors["reset"]

            if context_raw or execution_raw:
                # Format context info
                context_parts: list[str] = []
                if isinstance(context_raw, dict) and context_raw:
                    context = cast("dict[str, object]", context_raw)
                    # Extract meaningful context data
                    extra_dict = context
                    extra = (
                        cast("dict[str, object]", extra_dict.get("extra", {}))
                        if isinstance(extra_dict.get("extra"), dict)
                        else {}
                    )
                    if extra:
                        for key, value in extra.items():
                            if key in {
                                "entry_count",
                                "output_size_bytes",
                                "file_path",
                                "write_time_seconds",
                                "throughput_entries_per_sec",
                            }:
                                context_parts.append(f"{key}={value}")

                if context_parts:
                    context_line = f"  ├─ {context_color}Context:{reset} {', '.join(str(part) for part in context_parts)}"
                    lines.append(context_line)

            # Format execution info
            execution_parts: list[str] = []
            if isinstance(execution_raw, dict) and execution_raw:
                execution = cast("dict[str, object]", execution_raw)
                func_name = str(execution.get("function", ""))
                line_num = str(execution.get("line", ""))
                uptime = str(execution.get("uptime_seconds", ""))
                if func_name and line_num:
                    execution_parts.append(f"{func_name}:{line_num}")
                if uptime:
                    execution_parts.append(f"uptime={uptime}s")

            if execution_parts:
                execution_line = f"  └─ {execution_color}Execution:{reset} {', '.join(str(part) for part in execution_parts)}"
                lines.append(execution_line)

            return "\n".join(lines)

        def _format_full(
            self,
            event_dict: FlextTypes.Core.Dict,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            *,
            correlation_id: str,
        ) -> str:
            """Format full log entry with all available information.

            Args:
                event_dict: Event dictionary
                timestamp: Timestamp string
                level: Log level string
                service_name: Service name string
                message: Log message string
                correlation_id: Correlation ID string

            Returns:
                str: Formatted full log entry

            """
            lines: list[str] = []

            # Main line - fix the method call to use keyword argument
            main_line = self._format_compact(
                timestamp, level, service_name, message, correlation_id=correlation_id
            )
            lines.append(main_line)

            # Extract all sections
            context = cast("dict[str, object]", event_dict.get("context", {}))
            execution = cast("dict[str, object]", event_dict.get("execution", {}))
            service = cast("dict[str, object]", event_dict.get("service", {}))
            system = cast("dict[str, object]", event_dict.get("system", {}))

            # Color setup
            context_color = self._info_colors["context"]
            execution_color = self._info_colors["execution"]
            service_color = self._info_colors["service"]
            system_color = self._info_colors["system"]
            reset = self._info_colors["reset"]

            # Format context
            if context:
                extra = (
                    cast("dict[str, object]", context.get("extra", {}))
                    if isinstance(context.get("extra"), dict)
                    else {}
                )
                if extra:
                    # Cast to proper type to avoid unknown type issues
                    extra_dict = extra
                    context_parts: list[str] = [
                        f"{k}={v}" for k, v in extra_dict.items()
                    ]
                    if context_parts:
                        lines.append(
                            f"  ├─ {context_color}Context:{reset} {', '.join(str(part) for part in context_parts)}"
                        )

            # Format execution
            if isinstance(execution, dict) and execution:
                exec_parts: list[str] = []
                for key in ["function", "line", "uptime_seconds"]:
                    if execution.get(key):
                        if key == "uptime_seconds":
                            exec_parts.append(f"uptime={execution[key]}s")
                        elif key in {"function", "line"}:
                            if key == "function" and "line" in execution:
                                exec_parts.append(
                                    f"{execution['function']}:{execution['line']}"
                                )
                                break
                            if key == "line" and "function" not in execution:
                                exec_parts.append(f"line={execution[key]}")
                        else:
                            exec_parts.append(f"{key}={execution[key]}")
                if exec_parts:
                    lines.append(
                        f"  ├─ {execution_color}Execution:{reset} {', '.join(str(part) for part in exec_parts)}"
                    )

            # Format service info
            if service:
                service_parts: list[str] = [
                    f"{key}={service[key]}"
                    for key in ["name", "version", "instance_id", "environment"]
                    if service.get(key)
                ]
                if service_parts:
                    lines.append(
                        f"  ├─ {service_color}Service:{reset} {', '.join(str(part) for part in service_parts)}"
                    )

            # Format system info (last item gets └─)
            if system:
                system_parts: list[str] = []
                for key in ["hostname", "platform", "python_version", "process_id"]:
                    if system.get(key) and key == "process_id":
                        system_parts.append(f"pid={system[key]}")
                    elif system.get(key):
                        system_parts.append(f"{key}={system[key]}")
                if system_parts:
                    lines.append(
                        f"  └─ {system_color}System:{reset} {', '.join(str(part) for part in system_parts)}"
                    )

            return "\n".join(lines)

    class _ContextManager:
        """Centralized context management helper - optimized nested class."""

        def __init__(self, logger_instance: FlextLogger) -> None:
            self._logger = logger_instance

        def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
            """Set correlation ID with validation.

            Args:
                correlation_id: Correlation ID to set

            Returns:
                FlextResult[None]: Success or failure result

            """
            if not correlation_id:
                return FlextResult[None].fail(
                    "Correlation ID must be a non-empty string"
                )

            self._logger.set_correlation_id_internal(correlation_id)
            return FlextResult[None].ok(None)

        def set_request_context(
            self, model: FlextModels.LoggerRequestContextModel
        ) -> FlextResult[None]:
            """Set request context using validated model.

            Args:
                model: Request context model to set

            Returns:
                FlextResult[None]: Success or failure result

            """
            try:
                local = self._logger.get_local_storage()
                if not hasattr(local, "request_context"):
                    local.request_context = {}

                # Always clear existing context for new request context
                local.request_context.clear()

                # Add individual fields if they exist
                local.request_context["request_id"] = model.request_id
                if model.method:
                    local.request_context["method"] = model.method
                if model.path:
                    local.request_context["path"] = model.path
                if model.headers:
                    local.request_context["headers"] = model.headers
                if model.query_params:
                    local.request_context["query_params"] = model.query_params
                if model.user_id:
                    local.request_context["user_id"] = model.user_id
                if model.endpoint:
                    local.request_context["endpoint"] = model.endpoint
                if model.correlation_id:
                    self._logger.set_correlation_id_internal(model.correlation_id)

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set request context: {e}")

        def clear_request_context(self) -> FlextResult[None]:
            """Clear request context safely.

            Returns:
                FlextResult[None]: Success or failure result

            """
            try:
                local = self._logger.get_local_storage()
                if hasattr(local, "request_context"):
                    local.request_context.clear()
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to clear request context: {e}")

        def set_persistent_context(
            self, model: FlextModels.LoggerPermanentContextModel
        ) -> FlextResult[None]:
            """Set persistent context using validated model.

            Args:
                model: Persistent context model to set

            Returns:
                FlextResult[None]: Success or failure result

            """
            try:
                persistent_context = self._logger.get_persistent_context()

                if model.replace_existing or model.merge_strategy == "replace":
                    self._logger.set_persistent_context_dict(
                        dict(model.permanent_context)
                    )
                elif model.merge_strategy == "update":
                    persistent_context.update(model.permanent_context)
                elif model.merge_strategy == "merge_deep":
                    self._deep_merge_context(
                        persistent_context, model.permanent_context
                    )

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set persistent context: {e}")

        def bind_context(
            self, model: FlextModels.LoggerContextBindingModel
        ) -> FlextResult[FlextLogger]:
            """Create bound logger instance using validated model.

            Args:
                model: Context binding model to use

            Returns:
                FlextResult[FlextLogger]: Bound logger instance or error

            """
            try:
                # Get logger attributes using public method
                attrs = self._logger.get_logger_attributes()

                # Create new logger instance
                bound_logger = FlextLogger(
                    name=str(attrs["name"]),
                    _level=cast("FlextTypes.Config.LogLevel", attrs["level"]),
                    _service_name=cast("str | None", attrs["service_name"]),
                    _service_version=cast("str | None", attrs["service_version"]),
                    _correlation_id=cast("str | None", attrs["correlation_id"]),
                    _force_new=model.force_new_instance,
                )

                # Copy request context if requested
                local = self._logger.get_local_storage()
                if model.copy_request_context and hasattr(local, "request_context"):
                    context_model = FlextModels.LoggerRequestContextModel(
                        request_id=str(uuid.uuid4()),
                        **local.request_context,
                    )
                    bound_logger._context_manager.set_request_context(context_model)

                # Copy persistent context if requested
                if model.copy_permanent_context:
                    persistent_context = self._logger.get_persistent_context()
                    if persistent_context:
                        persistent_model = FlextModels.LoggerPermanentContextModel(
                            app_name="flext-core",
                            app_version=FlextConstants.Core.VERSION,
                            environment="development",
                            permanent_context=persistent_context,
                        )
                        bound_logger._context_manager.set_persistent_context(
                            persistent_model
                        )

                # Add new bound context
                if model.context_data:
                    # Extract known fields from context_data
                    context_kwargs: dict[str, object] = {
                        "request_id": str(uuid.uuid4())
                    }
                    if "method" in model.context_data:
                        context_kwargs["method"] = str(model.context_data["method"])
                    if "path" in model.context_data:
                        context_kwargs["path"] = str(model.context_data["path"])
                    if "headers" in model.context_data and isinstance(
                        model.context_data["headers"], dict
                    ):
                        context_kwargs["headers"] = {
                            str(k): str(v)
                            for k, v in cast(
                                "dict[object, object]", model.context_data["headers"]
                            ).items()
                        }
                    if "query_params" in model.context_data and isinstance(
                        model.context_data["query_params"], dict
                    ):
                        context_kwargs["query_params"] = {
                            str(k): str(v)
                            for k, v in cast(
                                "dict[object, object]",
                                model.context_data["query_params"],
                            ).items()
                        }
                    if "correlation_id" in model.context_data:
                        context_kwargs["correlation_id"] = str(
                            model.context_data["correlation_id"]
                        )

                    new_context_model = FlextModels.LoggerRequestContextModel(
                        request_id=str(context_kwargs.get("request_id", "")),
                        method=str(context_kwargs.get("method"))
                        if context_kwargs.get("method")
                        else None,
                        path=str(context_kwargs.get("path"))
                        if context_kwargs.get("path")
                        else None,
                        headers=cast(
                            "dict[str, str]", context_kwargs.get("headers", {})
                        )
                        if isinstance(context_kwargs.get("headers", {}), dict)
                        else {},
                        query_params=cast(
                            "dict[str, str]", context_kwargs.get("query_params", {})
                        )
                        if isinstance(context_kwargs.get("query_params", {}), dict)
                        else {},
                        correlation_id=str(context_kwargs.get("correlation_id"))
                        if context_kwargs.get("correlation_id")
                        else None,
                    )
                    bound_logger._context_manager.set_request_context(new_context_model)

                return FlextResult[FlextLogger].ok(bound_logger)
            except Exception as e:
                return FlextResult[FlextLogger].fail(f"Failed to bind context: {e}")

        def get_consolidated_context(self) -> dict[str, object]:
            """Get all context data consolidated for log entry building.

            Returns:
                dict[str, object]: Consolidated context data

            """
            consolidated: dict[str, object] = {}

            # Add request context
            local = self._logger.get_local_storage()
            if hasattr(local, "request_context"):
                consolidated.update(cast("dict[str, object]", local.request_context))

            # Add permanent context
            persistent_context = self._logger.get_persistent_context()
            consolidated.update(persistent_context)

            return consolidated

        def _deep_merge_context(
            self, target: dict[str, object], source: dict[str, object]
        ) -> None:
            """Deep merge context dictionaries."""
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target[key], dict)
                    and isinstance(value, dict)
                ):
                    self._deep_merge_context(
                        cast("dict[str, object]", target[key]),
                        cast("dict[str, object]", value),
                    )
                else:
                    target[key] = value

    @staticmethod
    def _validated_log_level(level: str) -> FlextTypes.Config.LogLevel:
        """Validate and coerce a raw string to Flext log level without casts.

        Args:
            level: Log level string to validate

        Returns:
            FlextTypes.Config.LogLevel: Validated log level

        """
        mapping: dict[str, str] = {
            "DEBUG": FlextConstants.Config.LogLevel.DEBUG,
            "INFO": FlextConstants.Config.LogLevel.INFO,
            "WARNING": FlextConstants.Config.LogLevel.WARNING,
            "ERROR": FlextConstants.Config.LogLevel.ERROR,
            "CRITICAL": FlextConstants.Config.LogLevel.CRITICAL,
        }
        upper = level.upper()
        # mapping values are already FlextTypes.Config.LogLevel
        return cast(
            "FlextTypes.Config.LogLevel",
            mapping.get(upper, FlextConstants.Config.LogLevel.INFO),
        )

    @classmethod
    def _is_configured(cls) -> bool:
        """Check if logger is configured.

        Returns:
            bool: True if logger is configured, False otherwise

        """
        return cls._configured

    def __str__(self) -> str:
        """Return logger name for string representation.

        Returns:
            str: Logger name string

        """
        return self._name

    def __repr__(self) -> str:
        """Return detailed representation.

        Returns:
            str: Detailed string representation

        """
        return f"FlextLogger(name='{self._name}', level='{self._level}')"


__all__: FlextTypes.Core.StringList = [
    "FlextLogger",
]
