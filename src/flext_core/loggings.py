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
import traceback
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import ClassVar, Self, TypedDict

import structlog
from structlog.typing import EventDict, Processor

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextLogger:
    """Structured logger that binds to ``FlextContext`` automatically.

    It fulfils the modernization requirement for context-first observability:
    correlation IDs, latency metrics, and sanitised payloads are attached to
    every log entry so downstream services obtain consistent telemetry.
    """

    # Comprehensive logging implementation with 735 lines of functionality
    # Features: Logger caching, thread-local storage, global correlation IDs
    # Capabilities: Performance tracking in logging, data sanitization, request context
    # Advanced logging for projects requiring structured logging and monitoring

    # Class-level configuration and shared state
    # OVER-COMPLEX: 6 different class variables for logging configuration
    _configured: ClassVar[bool] = False
    _configuration: ClassVar[dict[str, object]] = {}  # Singleton configuration storage
    _global_correlation_id: ClassVar[str | None] = None  # Global correlation ID - WHY?
    _service_info: ClassVar[FlextTypes.Core.Dict] = {}
    _request_context: ClassVar[FlextTypes.Core.Dict] = {}
    _performance_tracking: ClassVar[
        dict[str, float]
    ] = {}  # Performance tracking in LOGGING!

    # Logger instance cache for singleton pattern
    _instances: ClassVar[dict[str, Self]] = {}  # Instance caching - over-engineering

    # Thread-local storage for per-request context
    _local = (
        threading.local()
    )  # Thread-local for logging - most apps are single-threaded!

    # Instance attributes type declarations
    _name: str
    _level: FlextTypes.Config.LogLevel
    _environment: FlextTypes.Config.Environment

    def __new__(cls, name: str, **kwargs: object) -> Self:
        """Create or return cached logger instance."""
        # Check if this is a bind() call that needs a new instance
        force_new = kwargs.pop("_force_new", False)

        if not force_new and name in cls._instances:
            return cls._instances[name]

        instance = super().__new__(cls)
        if not force_new:  # Only cache if not forced new
            cls._instances[name] = instance
        return instance

    def __init__(
        self,
        name: str,
        level: FlextTypes.Config.LogLevel | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
        *,
        _force_new: bool = False,  # Accept but ignore this parameter
    ) -> None:
        """Initialize structured logger instance using Pydantic validation and FlextConfig singleton."""
        # Validate initialization parameters using Pydantic model
        try:
            from flext_core.models import FlextModels
            init_model = FlextModels.LoggerInitializationModel(
                name=name,
                level=level,
                service_name=service_name,
                service_version=service_version,
                correlation_id=correlation_id
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Logger initialization validation failed: {e}. Using provided values as-is.",
                         UserWarning, stacklevel=2)
            # Use parameters directly if validation fails
            init_model = None

        if not self._is_configured():
            # Always (re)configure structlog to ensure processors reflect stored config
            raw_kwargs = self._get_configuration() or {}
            allowed_keys = {
                "log_level",
                "json_output",
                "include_source",
                "structured_output",
            }
            config_kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed_keys}

            # Call configure with proper typed arguments
            log_level = str(config_kwargs.get("log_level", "INFO"))
            json_output = config_kwargs.get("json_output")
            include_source = bool(config_kwargs.get("include_source", True))
            structured_output = bool(config_kwargs.get("structured_output", True))

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
        validated_level = init_model.level if init_model else level
        validated_service_name = init_model.service_name if init_model else service_name
        validated_service_version = init_model.service_version if init_model else service_version
        validated_correlation_id = init_model.correlation_id if init_model else correlation_id

        self._name = validated_name
        # Load configuration early so .env and FLEXT_* vars are available
        config = FlextConfig.get_global_instance()

        # Resolve log level with strict precedence and deterministic behavior
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

        # 0) Detect test environment deterministically (pytest session or explicit config)
        is_pytest = (
            os.getenv("PYTEST_CURRENT_TEST") is not None or "pytest" in sys.modules
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
            resolved_level = "WARNING"

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

        # 4) Configuration/defaults
        if resolved_level is None:
            cfg_level = str(config.log_level).upper()
            resolved_level = cfg_level if cfg_level in valid_levels else "INFO"

        self._level = self._validated_log_level(resolved_level)

        # Use environment from configuration singleton for consistency
        self._environment = config.environment

        # Initialize service context using validated parameters
        self._service_name = validated_service_name or self._extract_service_name()
        self._service_version = validated_service_version or self._get_version()

        # Set up performance tracking
        self._start_time = time.time()

        # Instance-level correlation ID (can override global)
        context_id = FlextContext.Correlation.get_correlation_id()

        self._correlation_id = (
            validated_correlation_id
            or self._global_correlation_id
            or context_id  # Check global context
            or self._generate_correlation_id()
        )

        # Set up structured logger with enriched context (import locally)

        self._structlog_logger = structlog.get_logger(validated_name)

        # Initialize persistent context
        self._persistent_context: FlextTypes.Core.Dict = {
            "service": {
                "name": self._service_name,
                "version": self._service_version,
                "instance_id": self._get_instance_id(),
                "environment": self._get_environment(),
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
        """
        return hash(str(self))

    def _extract_service_name(self) -> str:
        """Extract service name from logger name or environment."""
        if service_name := os.environ.get("SERVICE_NAME"):
            return service_name

        # Extract from module name
        min_parts = 2
        parts = self._name.split(".")
        if len(parts) >= min_parts and parts[0].startswith("flext_"):
            return parts[0].replace("_", "-")

        return "flext-core"

    def _get_project_specific_env_var(self, suffix: str) -> str | None:
        """Get project-specific environment variable name for the current service."""
        service_name = self._extract_service_name()
        if service_name.startswith("flext-"):
            # Convert flext-core -> FLEXT_CORE, flext-ldap -> FLEXT_LDAP, etc.
            env_name = service_name.upper().replace("-", "_")
            return f"{env_name}_{suffix}"
        return None

    def _get_version(self) -> str:
        """Get service version."""
        return (
            os.environ.get("SERVICE_VERSION") or FlextConstants.Core.VERSION or "0.9.0"
        )

    def _get_environment(self) -> str:
        """Determine current environment."""
        return (
            os.environ.get("ENVIRONMENT") or os.environ.get("ENV") or "development"
        ).lower()

    def _get_instance_id(self) -> str:
        """Get unique instance identifier."""
        return os.environ.get("INSTANCE_ID") or f"{platform.node()}-{os.getpid()}"

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID."""
        return FlextUtilities.Generators.generate_correlation_id()

    def _get_current_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(UTC).isoformat()

    def _sanitize_context(self, context: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        """Sanitize context by redacting sensitive data."""
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

        sanitized: FlextTypes.Core.Dict = {}
        for key, value in context.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(value)
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
        """Build structured log entry."""
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
        if isinstance(service_ctx, dict):
            entry["service"] = dict(service_ctx)
        if isinstance(system_ctx, dict):
            entry["system"] = dict(system_ctx)

        # Add request context if available
        request_context = getattr(self._local, "request_context", None)
        # Always set request (may be empty)
        entry["request"] = (
            dict(request_context) if isinstance(request_context, dict) else {}
        )

        # Add permanent context if available
        permanent_context = getattr(self, "_permanent_context", None)
        if isinstance(permanent_context, dict) and permanent_context:
            entry["permanent"] = dict(permanent_context)

        # Add performance metrics
        if duration_ms is not None:
            entry["performance"] = {
                "duration_ms": round(duration_ms, 3),
                "timestamp": self._get_current_timestamp(),
            }

        # Add error details if present
        if error:
            if isinstance(error, Exception):
                # Handle Exception objects with full details
                entry["error"] = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "stack_trace": traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    ),
                    "module": getattr(error, "__module__", "unknown"),
                }
            else:
                # Handle string error messages
                entry["error"] = {
                    "type": "StringError",
                    "message": str(error),
                    "stack_trace": None,
                    "module": "unknown",
                }

        # Add sanitized user context
        if context:
            sanitized_context = self._sanitize_context(dict(context))
            entry["context"] = sanitized_context
        else:
            # Ensure context key exists for type stability
            entry["context"] = {}

        # Add execution context
        entry["execution"] = {
            "function": self._get_calling_function(),
            "line": self._get_calling_line(),
            "uptime_seconds": round(time.time() - self._start_time, 3),
        }

        return entry

    def _get_calling_function(self) -> str:
        """Get the name of the calling function."""
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
        """Get the line number of the calling code."""
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
            import warnings
            warnings.warn(f"Failed to set correlation ID: {result.error}", UserWarning, stacklevel=2)  # Instance-level correlation ID

    def set_request_context(self, **context: object) -> None:
        """Set request-specific context."""
        if not hasattr(self._local, "request_context"):
            self._local.request_context = {}
        self._local.request_context.update(context)

    def clear_request_context(self) -> None:
        """Clear request context."""
        if hasattr(self._local, "request_context"):
            self._local.request_context.clear()

    def bind(self, **context: object) -> FlextLogger:
        """Create logger instance with bound context."""
        # Create a new logger instance with same configuration
        # Use _force_new=True to bypass singleton pattern for bind()
        bound_logger = FlextLogger(
            name=self._name,
            level=self._level,
            service_name=getattr(self, "_service_name", None),
            service_version=getattr(self, "_service_version", None),
            correlation_id=getattr(self, "_correlation_id", None),
            _force_new=True,
        )

        # Copy existing request context
        if hasattr(self._local, "request_context"):
            bound_logger.set_request_context(**self._local.request_context)

        # Copy existing permanent context
        if hasattr(self, "_permanent_context"):
            bound_logger._permanent_context = self._permanent_context.copy()

        # Add new bound context
        bound_logger.set_request_context(**context)

        return bound_logger

    def set_context(
        self,
        context_dict: FlextTypes.Core.Dict | None = None,
        **context: object,
    ) -> None:
        """Set permanent context data."""
        if not hasattr(self, "_permanent_context"):
            self._permanent_context: FlextTypes.Core.Dict = {}

        if context_dict is not None:
            # Replace existing context with new dict
            self._permanent_context = dict(context_dict)
            # Add any additional kwargs
            self._permanent_context.update(context)
        else:
            # Just update existing context with kwargs
            self._permanent_context.update(context)

    def with_context(self, **context: object) -> FlextLogger:
        """Create logger instance with additional context."""
        return self.bind(**context)

    def start_operation(self, operation_name: str, **context: object) -> str:
        """Start tracking operation with metrics."""
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

    # Standard logging methods with enhanced context
    def trace(self, message: str, *args: object, **context: object) -> None:
        """Log trace message."""
        formatted_message = message % args if args else message
        entry = self._build_log_entry("TRACE", formatted_message, context)
        self._structlog_logger.debug(
            formatted_message,
            **entry,
        )  # Use debug since structlog doesn't have trace

    def debug(self, message: str, *args: object, **context: object) -> None:
        """Log debug message."""
        formatted_message = message % args if args else message
        # Use simple output if structured_output is disabled
        if not self._configuration.get("structured_output", True):
            self._structlog_logger.debug(formatted_message, **context)
        else:
            entry = self._build_log_entry("DEBUG", formatted_message, context)
            self._structlog_logger.debug(formatted_message, **entry)

    def info(self, message: str, *args: object, **context: object) -> None:
        """Log info message."""
        formatted_message = message % args if args else message
        # Use simple output if structured_output is disabled
        if not self._configuration.get("structured_output", True):
            self._structlog_logger.info(formatted_message, **context)
        else:
            entry = self._build_log_entry("INFO", formatted_message, context)
            self._structlog_logger.info(formatted_message, **entry)

    def warning(self, message: str, *args: object, **context: object) -> None:
        """Log warning message."""
        formatted_message = message % args if args else message
        # Use simple output if structured_output is disabled
        if not self._configuration.get("structured_output", True):
            self._structlog_logger.warning(formatted_message, **context)
        else:
            entry = self._build_log_entry("WARNING", formatted_message, context)
            self._structlog_logger.warning(formatted_message, **entry)

    def error(
        self,
        message: str,
        *args: object,
        error: Exception | str | None = None,
        **context: object,
    ) -> None:
        """Log error message with context and error details."""
        formatted_message = message % args if args else message
        # Use simple output if structured_output is disabled
        if not self._configuration.get("structured_output", True):
            if error:
                context["error"] = str(error)
            self._structlog_logger.error(formatted_message, **context)
        else:
            entry = self._build_log_entry("ERROR", formatted_message, context, error)
            self._structlog_logger.error(formatted_message, **entry)

    def critical(
        self,
        message: str,
        *args: object,
        error: Exception | str | None = None,
        **context: object,
    ) -> None:
        """Log critical message with context and error details."""
        formatted_message = message % args if args else message
        # Use simple output if structured_output is disabled
        if not self._configuration.get("structured_output", True):
            if error:
                context["error"] = str(error)
            self._structlog_logger.critical(formatted_message, **context)
        else:
            entry = self._build_log_entry("CRITICAL", formatted_message, context, error)
            self._structlog_logger.critical(formatted_message, **entry)

    def exception(self, message: str, *args: object, **context: object) -> None:
        """Log exception with stack trace and context."""
        formatted_message = message % args if args else message
        exc_info = sys.exc_info()
        error = exc_info[1] if isinstance(exc_info[1], Exception) else None
        entry = self._build_log_entry("ERROR", formatted_message, context, error)
        self._structlog_logger.error(formatted_message, **entry)

    @classmethod
    def configure(
        cls,
        *,
        log_level: str = "INFO",
        json_output: bool | None = None,
        include_source: bool = True,
        structured_output: bool = True,
        log_verbosity: str = "detailed",
    ) -> None:
        """Configure structured logging system using Pydantic validation.

        This configuration is stored as a singleton and reused by all FlextLogger instances.
        Parameters are consolidated and validated through LoggerConfigurationModel.

        Args:
            log_level: Logging level string
            json_output: Use JSON output format
            include_source: Include source code location info
            structured_output: Use structured logging format
            log_verbosity: Console output verbosity ('compact', 'detailed', 'full')

        """
        # Use Pydantic model for parameter consolidation and validation
        try:
            config_model = FlextModels.LoggerConfigurationModel(
                log_level=log_level,
                json_output=json_output,
                include_source=include_source,
                structured_output=structured_output,
                log_verbosity=log_verbosity,
            )
        except Exception as e:
            # If validation fails, fall back to original behavior with warnings
            import warnings
            warnings.warn(
                f"Logger configuration validation failed: {e}. Using provided values as-is.",
                UserWarning,
                stacklevel=2
            )
            config_model = type("Config", (), {
                "log_level": log_level,
                "json_output": json_output,
                "include_source": include_source,
                "structured_output": structured_output,
                "log_verbosity": log_verbosity,
            })()

        # Reset if already configured (allow reconfiguration)
        if cls._configured and structlog.is_configured():
            structlog.reset_defaults()

        # Auto-detect verbosity from environment if not specified
        # Note: We can't use project-specific detection here since this is a class method
        # Project-specific variables are handled in individual logger initialization
        env_verbosity = os.environ.get("FLEXT_LOG_VERBOSITY", config_model.log_verbosity).lower()
        if env_verbosity in {"compact", "detailed", "full"}:
            effective_verbosity = env_verbosity
        else:
            effective_verbosity = config_model.log_verbosity

        # Store configuration in singleton for reuse
        cls._configuration = {
            "log_level": config_model.log_level,
            "json_output": config_model.json_output,
            "include_source": config_model.include_source,
            "structured_output": config_model.structured_output,
            "log_verbosity": effective_verbosity,
        }

        # Auto-detect output format if not specified
        effective_json_output = config_model.json_output
        if effective_json_output is None:
            env = os.environ.get("ENVIRONMENT", "development").lower()
            effective_json_output = env in {"production", "staging", "prod"}
            cls._configuration["json_output"] = effective_json_output

        processors: list[Processor] = [
            # Essential processors
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
        ]

        # Add timestamp processor with ISO 8601 format
        processors.append(
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="@timestamp"),
        )

        # Add source information if requested
        if config_model.include_source:
            processors.append(
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ],
                ),
            )

        # Add structured processors
        if config_model.structured_output:
            processors.extend(
                [
                    cls._add_correlation_processor,
                    cls._add_performance_processor,
                    cls._sanitize_processor,
                ],
            )

        # Choose output format
        if effective_json_output:
            processors.append(
                structlog.processors.JSONRenderer(
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            )
        elif not config_model.structured_output:
            # For non-structured output, use KeyValueRenderer for simple readable output
            processors.append(
                structlog.processors.KeyValueRenderer(
                    key_order=["message"],  # Show message first
                    drop_missing=True,
                ),
            )
        else:
            processors.append(
                FlextLogger._create_enhanced_console_renderer(effective_verbosity)
            )

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure stdlib logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stderr,
            level=getattr(logging, config_model.log_level.upper(), logging.INFO),
        )

        cls._configured = True

    @staticmethod
    def _add_correlation_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add correlation ID to log entries."""
        if FlextLogger._global_correlation_id:
            event_dict["correlation_id"] = FlextLogger._global_correlation_id
        return event_dict

    @staticmethod
    def _add_performance_processor(
        _logger: logging.Logger,
        _method_name: str,
        event_dict: EventDict,
    ) -> EventDict:
        """Add metadata to log entries."""
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
        """Sanitize sensitive data from log entries."""
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
        verbosity: str = "detailed",
    ) -> "FlextLogger._ConsoleRenderer":
        """Create enhanced console renderer with configurable verbosity - using nested helper class."""
        return FlextLogger._ConsoleRenderer(verbosity=verbosity)

    @classmethod
    def set_global_correlation_id(cls, correlation_id: str | None) -> None:
        """Set global correlation ID."""
        cls._global_correlation_id = correlation_id

    @classmethod
    def get_global_correlation_id(cls) -> str | None:
        """Get global correlation ID."""
        return cls._global_correlation_id

    @classmethod
    def get_configuration(cls) -> dict[str, object]:
        """Get current logging configuration.

        Returns:
            Dictionary with current configuration settings

        """
        return cls._configuration.copy()

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

        def __init__(self, verbosity: str = "detailed") -> None:
            """Initialize renderer with specified verbosity level."""
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
            """Render log entry with hierarchical formatting."""
            return self._format_event(event_dict)

        def _format_event(self, event_dict: FlextTypes.Core.Dict) -> str:
            """Format event dictionary into hierarchical log output."""
            # Extract core information with proper type casting
            timestamp = str(event_dict.get("@timestamp", ""))
            level = str(event_dict.get("level", "INFO")).upper()
            logger_name = str(event_dict.get("logger_name", ""))
            message = str(event_dict.get("event", ""))
            correlation_id = str(event_dict.get("correlation_id", ""))

            # Extract service info with type safety
            service_info = event_dict.get("service", {})
            if isinstance(service_info, dict):
                service_name = str(service_info.get("name", logger_name))
            else:
                service_name = logger_name

            # Format main log line based on verbosity
            if self.verbosity == "compact":
                return self._format_compact(
                    timestamp, level, service_name, message, correlation_id
                )
            if self.verbosity == "detailed":
                return self._format_detailed(
                    event_dict, timestamp, level, service_name, message, correlation_id
                )
            # full
            return self._format_full(
                event_dict, timestamp, level, service_name, message, correlation_id
            )

        def _format_compact(
            self,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            correlation_id: str,
        ) -> str:
            """Format compact log entry."""
            # Clean timestamp (remove microseconds and timezone info for readability)
            clean_timestamp = timestamp.split(".", maxsplit=1)[0] + "Z" if timestamp else ""

            level_color = self._level_colors.get(level.lower(), "")
            reset = self._info_colors["reset"]
            correlation_color = self._info_colors["correlation"]

            # Format: timestamp [LEVEL] [service] message [correlation_id]
            correlation_part = (
                f" {correlation_color}[{correlation_id}]{reset}" if correlation_id else ""
            )

            return f"{clean_timestamp} {level_color}[{level}]{reset} [{service_name}] {message}{correlation_part}"

        def _format_detailed(
            self,
            event_dict: FlextTypes.Core.Dict,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            correlation_id: str,
        ) -> str:
            """Format detailed log entry with context tree."""
            lines = []

            # Main line
            main_line = self._format_compact(
                timestamp, level, service_name, message, correlation_id
            )
            lines.append(main_line)

            # Add context information if available
            context = event_dict.get("context", {})
            execution = event_dict.get("execution", {})

            context_color = self._info_colors["context"]
            execution_color = self._info_colors["execution"]
            reset = self._info_colors["reset"]

            if context or execution:
                # Format context info
                context_parts = []
                if isinstance(context, dict) and context:
                    # Extract meaningful context data
                    extra = (
                        context.get("extra", {})
                        if isinstance(context.get("extra"), dict)
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
                    context_line = (
                        f"  ├─ {context_color}Context:{reset} {', '.join(context_parts)}"
                    )
                    lines.append(context_line)

                # Format execution info
                execution_parts = []
                if isinstance(execution, dict) and execution:
                    func_name = execution.get("function", "")
                    line_num = execution.get("line", "")
                    uptime = execution.get("uptime_seconds", "")
                    if func_name and line_num:
                        execution_parts.append(f"{func_name}:{line_num}")
                    if uptime:
                        execution_parts.append(f"uptime={uptime}s")

                if execution_parts:
                    execution_line = f"  └─ {execution_color}Execution:{reset} {', '.join(execution_parts)}"
                    lines.append(execution_line)

            return "\n".join(lines)

        def _format_full(
            self,
            event_dict: FlextTypes.Core.Dict,
            timestamp: str,
            level: str,
            service_name: str,
            message: str,
            correlation_id: str,
        ) -> str:
            """Format full log entry with all available information."""
            lines = []

            # Main line
            main_line = self._format_compact(
                timestamp, level, service_name, message, correlation_id
            )
            lines.append(main_line)

            # Extract all sections
            context = event_dict.get("context", {})
            execution = event_dict.get("execution", {})
            service = event_dict.get("service", {})
            system = event_dict.get("system", {})

            # Color setup
            context_color = self._info_colors["context"]
            execution_color = self._info_colors["execution"]
            service_color = self._info_colors["service"]
            system_color = self._info_colors["system"]
            reset = self._info_colors["reset"]

            # Format context
            if isinstance(context, dict) and context:
                extra = (
                    context.get("extra", {})
                    if isinstance(context.get("extra"), dict)
                    else {}
                )
                if extra:
                    context_parts = [f"{k}={v}" for k, v in extra.items()]
                    if context_parts:
                        lines.append(
                            f"  ├─ {context_color}Context:{reset} {', '.join(context_parts)}"
                        )

            # Format execution
            if isinstance(execution, dict) and execution:
                exec_parts = []
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
                        f"  ├─ {execution_color}Execution:{reset} {', '.join(exec_parts)}"
                    )

            # Format service info
            if isinstance(service, dict) and service:
                service_parts = [
                    f"{key}={service[key]}"
                    for key in ["name", "version", "instance_id", "environment"]
                    if service.get(key)
                ]
                if service_parts:
                    lines.append(
                        f"  ├─ {service_color}Service:{reset} {', '.join(service_parts)}"
                    )

            # Format system info (last item gets └─)
            if isinstance(system, dict) and system:
                system_parts = []
                for key in ["hostname", "platform", "python_version", "process_id"]:
                    if system.get(key):
                        if key == "process_id":
                            system_parts.append(f"pid={system[key]}")
                        else:
                            system_parts.append(f"{key}={system[key]}")
                if system_parts:
                    lines.append(
                        f"  └─ {system_color}System:{reset} {', '.join(system_parts)}"
                    )

            return "\n".join(lines)

    class _ContextManager:
        """Centralized context management helper - optimized nested class."""

        def __init__(self, logger_instance: "FlextLogger") -> None:
            self._logger = logger_instance

        def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
            """Set correlation ID with validation."""
            if not correlation_id or not isinstance(correlation_id, str):
                return FlextResult[None].fail("Correlation ID must be a non-empty string")
            
            self._logger._correlation_id = correlation_id
            return FlextResult[None].ok(None)

        def set_request_context(self, model: FlextModels.LoggerRequestContextModel) -> FlextResult[None]:
            """Set request context using validated model."""
            try:
                if not hasattr(self._logger._local, "request_context"):
                    self._logger._local.request_context = {}
                
                if model.clear_existing:
                    self._logger._local.request_context.clear()
                
                self._logger._local.request_context.update(model.request_context)
                
                if model.correlation_id:
                    self._logger._correlation_id = model.correlation_id
                
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set request context: {e}")

        def clear_request_context(self) -> FlextResult[None]:
            """Clear request context safely."""
            try:
                if hasattr(self._logger._local, "request_context"):
                    self._logger._local.request_context.clear()
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to clear request context: {e}")

        def set_permanent_context(self, model: FlextModels.LoggerPermanentContextModel) -> FlextResult[None]:
            """Set permanent context using validated model."""
            try:
                if not hasattr(self._logger, "_permanent_context"):
                    self._logger._permanent_context: FlextTypes.Core.Dict = {}
                
                if model.replace_existing or model.merge_strategy == "replace":
                    self._logger._permanent_context = dict(model.permanent_context)
                elif model.merge_strategy == "update":
                    self._logger._permanent_context.update(model.permanent_context)
                elif model.merge_strategy == "merge_deep":
                    self._deep_merge_context(self._logger._permanent_context, model.permanent_context)
                
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set permanent context: {e}")

        def bind_context(self, model: FlextModels.LoggerContextBindingModel) -> FlextResult["FlextLogger"]:
            """Create bound logger instance using validated model."""
            try:
                # Create new logger instance
                bound_logger = FlextLogger(
                    name=self._logger._name,
                    level=self._logger._level,
                    service_name=getattr(self._logger, "_service_name", None),
                    service_version=getattr(self._logger, "_service_version", None),
                    correlation_id=getattr(self._logger, "_correlation_id", None),
                    _force_new=model.force_new_instance,
                )

                # Copy request context if requested
                if model.copy_request_context and hasattr(self._logger._local, "request_context"):
                    context_model = FlextModels.LoggerRequestContextModel(
                        request_context=self._logger._local.request_context
                    )
                    bound_logger._context_manager.set_request_context(context_model)

                # Copy permanent context if requested
                if model.copy_permanent_context and hasattr(self._logger, "_permanent_context"):
                    permanent_model = FlextModels.LoggerPermanentContextModel(
                        permanent_context=self._logger._permanent_context
                    )
                    bound_logger._context_manager.set_permanent_context(permanent_model)

                # Add new bound context
                if model.context_data:
                    new_context_model = FlextModels.LoggerRequestContextModel(
                        request_context=model.context_data
                    )
                    bound_logger._context_manager.set_request_context(new_context_model)

                return FlextResult["FlextLogger"].ok(bound_logger)
            except Exception as e:
                return FlextResult["FlextLogger"].fail(f"Failed to bind context: {e}")

        def get_consolidated_context(self) -> dict[str, object]:
            """Get all context data consolidated for log entry building."""
            consolidated = {}
            
            # Add request context
            if hasattr(self._logger._local, "request_context"):
                consolidated.update(self._logger._local.request_context)
            
            # Add permanent context
            if hasattr(self._logger, "_permanent_context"):
                consolidated.update(self._logger._permanent_context)
            
            return consolidated

        def _deep_merge_context(self, target: dict[str, object], source: dict[str, object]) -> None:
            """Deep merge context dictionaries."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge_context(target[key], value)  # type: ignore
                else:
                    target[key] = value

    @staticmethod
    def _validated_log_level(level: str) -> FlextTypes.Config.LogLevel:
        """Validate and coerce a raw string to Flext log level without casts."""
        mapping: dict[str, FlextTypes.Config.LogLevel] = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "CRITICAL",
        }
        upper = level.upper()
        return mapping.get(upper, "INFO")

    @classmethod
    def _is_configured(cls) -> bool:
        """Check if logger is configured."""
        return cls._configured

    @classmethod
    def _get_configuration(cls) -> dict[str, object]:
        """Get logger configuration."""
        return cls._configuration


__all__: FlextTypes.Core.StringList = [
    "FlextLogger",
]
