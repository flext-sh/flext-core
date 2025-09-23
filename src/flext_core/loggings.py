"""Structured logging utilities enabling the context-first pillar for 1.0.0.

The module mirrors the expectations in ``README.md`` and
``docs/architecture.md`` by binding log records to ``FlextContext`` metadata
and providing the default processors shared across FLEXT packages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
import traceback
import uuid
import warnings
from collections.abc import Mapping
from datetime import UTC, date, datetime, time as datetime_time
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self, cast, override

import colorlog
import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from structlog.typing import EventDict, Processor

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextLogger:
    """Structured logging solution for the FLEXT ecosystem.

    FlextLogger provides enhanced logging with structured output, context management,
    correlation tracking, and performance monitoring. Use FlextLogger throughout
    FLEXT applications for consistent, structured logging.

    **ECOSYSTEM USAGE**: Create logger instances directly:
        ```python
        from flext_core import FlextLogger

        logger = FlextLogger(__name__)
        logger.info("Operation completed", extra={"user_id": "123"})

        # Context management
        logger.set_correlation_id("req-123")
        logger.bind_context({"service": "user-service"})
        ```

    **UNIFIED ARCHITECTURE**: Single class design containing all logging
    functionality with internal Pydantic models for validation.

    Provides structured logging with configurable output formats, correlation tracking,
    request context binding, performance monitoring, and full Pydantic v2 model integration.
    """

    # =============================================================================
    # INTERNAL MODELS (moved from FlextModels to break circular dependency)
    # =============================================================================

    class LogEntry(BaseModel):
        """Structured log entry model with comprehensive validation."""

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )

        # Core required fields
        message: str = Field(description="Log message content")
        level: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().log_level,
            description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        )
        timestamp: str = Field(
            default_factory=lambda: datetime.now(UTC).isoformat(),
            description="ISO timestamp when log entry was created",
        )
        logger: str = Field(description="Logger name that created this entry")

        # Optional core fields
        correlation_id: str | None = Field(
            default=None, description="Correlation ID for tracing related log entries"
        )
        context: FlextTypes.Core.Dict = Field(
            default_factory=dict,
            description="Additional context data for the log entry",
        )

        # Extended structured fields used in actual logging
        logger_name: str | None = Field(
            default=None, description="Name of the logger instance"
        )
        module: str | None = Field(
            default=None, description="Module where the log entry originated"
        )
        function: str | None = Field(
            default=None, description="Function where the log entry originated"
        )
        line_number: int | None = Field(
            default=None, description="Line number where the log entry originated", ge=1
        )
        request: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Request-specific context data"
        )
        service: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Service-specific context data"
        )
        system: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="System-level context data"
        )
        execution: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Execution context data"
        )
        permanent: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Permanent context data"
        )
        performance: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Performance-related metrics"
        )
        error: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Error-specific information"
        )

        @field_validator("level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level is one of the allowed values."""
            valid_levels = FlextConstants.Logging.VALID_LEVELS
            v_upper = v.upper()
            if v_upper not in valid_levels:
                msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
                raise ValueError(msg)
            return v_upper

        @field_validator(
            "context",
            "request",
            "service",
            "system",
            "execution",
            "permanent",
            "performance",
            "error",
        )
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data is JSON serializable."""
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                msg = f"Context data must be JSON serializable: {e}"
                raise ValueError(msg) from e
            return v

        @field_validator("timestamp")
        @classmethod
        def validate_timestamp_format(cls, v: str) -> str:
            """Validate timestamp is in ISO format."""
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                msg = f"Invalid timestamp format: {v}. Must be ISO format: {e}"
                raise ValueError(msg) from e
            return v

        @model_validator(mode="after")
        def validate_entry_consistency(self) -> Self:
            """Validate log entry consistency and enrich if needed."""
            # Ensure logger_name matches logger if not explicitly set
            if not self.logger_name and self.logger:
                self.logger_name = self.logger

            # Validate context size limits
            max_context_size = FlextConstants.Performance.MAX_METADATA_SIZE
            for field_name in [
                "context",
                "request",
                "service",
                "system",
                "execution",
                "permanent",
                "performance",
                "error",
            ]:
                field_value = getattr(self, field_name, {})
                if len(str(field_value)) > max_context_size:
                    msg = f"Field '{field_name}' context data too large (max {max_context_size} characters)"
                    raise ValueError(msg)

            return self

        def to_dict(self, *, exclude_none: bool = True) -> dict[str, object]:
            """Convert log entry to dictionary format for compatibility."""
            return self.model_dump(exclude_none=exclude_none)

        def to_structured_dict(self) -> dict[str, object]:
            """Convert to structured dictionary with only non-empty fields."""
            return self.model_dump(
                exclude_none=True, exclude_defaults=True, exclude_unset=True
            )

    class LoggerInitializationModel(BaseModel):
        """Logger initialization with advanced validation."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: str
        log_level: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().log_level
        )
        structured_output: bool = True
        include_source: bool = True
        json_output: bool | None = None

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level is valid."""
            valid_levels = FlextConstants.Logging.VALID_LEVELS
            v_upper = v.upper()
            if v_upper not in valid_levels:
                msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
                raise ValueError(msg)
            return v_upper

    class LoggerRequestContextModel(BaseModel):
        """Logger request context model."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        method: str | None = None
        path: str | None = None
        headers: dict[str, str] = Field(default_factory=dict)
        query_params: dict[str, str] = Field(default_factory=dict)
        correlation_id: str | None = None
        user_id: str | None = None
        endpoint: str | None = None
        custom_data: dict[str, str] = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_request_context(self) -> Self:
            """Validate request context consistency."""
            if (
                self.method
                and self.method not in FlextConstants.Platform.VALID_HTTP_METHODS
            ):
                msg = f"Invalid HTTP method: {self.method}"
                raise ValueError(msg)
            return self

    class LoggerContextBindingModel(BaseModel):
        """Logger context binding model."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        logger_name: str
        context_data: FlextTypes.Core.Dict = Field(default_factory=dict)
        bind_type: Literal["temporary", "permanent"] = "temporary"
        clear_existing: bool = False
        force_new_instance: bool = False
        copy_request_context: bool = False
        copy_permanent_context: bool = False

        @field_validator("context_data")
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data."""
            max_context_keys = 100
            if len(v) > max_context_keys:
                msg = f"Context data too large (max {max_context_keys} keys)"
                raise ValueError(msg)
            return v

    class LoggerPermanentContextModel(BaseModel):
        """Logger permanent context model."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        app_name: str
        app_version: str
        environment: FlextTypes.Config.Environment
        host: str | None = None
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)
        permanent_context: FlextTypes.Core.Dict = Field(default_factory=dict)
        replace_existing: bool = False
        merge_strategy: Literal["replace", "update", "merge_deep"] = "update"

        @field_validator("environment", mode="before")
        @classmethod
        def normalize_environment(cls, value: object) -> FlextTypes.Config.Environment:
            """Normalize and validate environment against shared constants."""
            # Convert to string if needed (handles any input type)
            str_value: str
            if isinstance(value, str):
                str_value = value
            elif hasattr(value, "value"):
                # Handle enum-like objects
                str_value = str(getattr(value, "value"))
            else:
                # Fallback to string conversion
                str_value = str(value)

            normalized = str_value.lower()
            valid_envs = set(FlextConstants.Config.ENVIRONMENTS)
            if normalized not in valid_envs:
                msg = f"Environment must be one of {sorted(valid_envs)}"
                raise ValueError(msg)
            return cast("FlextTypes.Config.Environment", normalized)

        @model_validator(mode="after")
        def validate_permanent_context(self) -> Self:
            """Validate permanent context."""
            # Use only the enum values as the single source of truth for valid environments.
            valid_envs = {
                env.value.lower()
                for env in FlextConstants.Environment.ConfigEnvironment
            }
            if self.environment.lower() not in valid_envs:
                sorted_envs = sorted(valid_envs)
                msg = (
                    f"Invalid environment: {self.environment}. "
                    f"Must be one of {sorted_envs}"
                )
                raise ValueError(msg)
            return self

    # =============================================================================
    # EXISTING IMPLEMENTATION CONTINUES UNCHANGED...
    # =============================================================================

    # Static class attributes
    _configured: bool = False
    _global_correlation_id: FlextTypes.Core.Optional[FlextTypes.Identifiers.Id] = None
    _logger_cache: dict[str, FlextLogger] = {}
    _cache_lock = threading.Lock()

    # Nested console renderer class
    class _ConsoleRenderer:
        """Enhanced console renderer with configurable verbosity."""

        def __init__(self, verbosity: str = FlextConstants.Logging.VERBOSITY) -> None:
            self._verbosity = verbosity.lower()

        def __call__(
            self,
            _logger: logging.Logger,
            _name: str,
            event_dict: EventDict,
        ) -> str:
            # Simplified renderer focused on essential information
            timestamp = event_dict.get("timestamp", datetime.now(UTC).isoformat())
            logger_name = event_dict.get("logger", "unknown")
            level = event_dict.get("level", "INFO")
            message = event_dict.get("event", "")

            # Extract correlation ID
            correlation_id = event_dict.get("correlation_id")
            correlation_info = f" [corr:{correlation_id[:8]}]" if correlation_id else ""

            # Build base log line
            base_line = (
                f"{timestamp} | {level:8} | {logger_name}{correlation_info} | {message}"
            )

            if self._verbosity == "minimal":
                return base_line

            # Add context for detailed verbosity
            if self._verbosity == "detailed":
                context = event_dict.get("context", {})
                if context:
                    context_str = json.dumps(
                        context, indent=None, separators=(",", ":")
                    )
                    return f"{base_line} | {context_str}"

            return base_line

    class _ContextManager:
        """Centralized context manager for optimized operations."""

        def __init__(self, logger: FlextLogger) -> None:
            self._logger = logger

        def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
            """Set correlation ID - optimized implementation."""
            try:
                self._logger.set_correlation_id_internal(correlation_id)
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set correlation ID: {e}")

        def set_request_context(
            self, model: FlextLogger.LoggerRequestContextModel
        ) -> FlextResult[None]:
            """Set request context using model - optimized implementation."""
            try:
                # Convert model to dict for storage
                context_dict = model.model_dump()
                self._logger.get_local_storage().request_context = context_dict
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set request context: {e}")

        def clear_request_context(self) -> FlextResult[None]:
            """Clear request context - optimized implementation."""
            try:
                local_storage = self._logger.get_local_storage()
                if hasattr(local_storage, "request_context"):
                    del local_storage.request_context
                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to clear request context: {e}")

        def set_permanent_context(
            self, model: FlextLogger.LoggerPermanentContextModel
        ) -> FlextResult[None]:
            """Set permanent context using model - optimized implementation."""
            try:
                # Convert model to dict for storage
                context_dict = {
                    "app_name": model.app_name,
                    "app_version": model.app_version,
                    "environment": model.environment,
                    "host": model.host,
                    "metadata": model.metadata,
                    **model.permanent_context,
                }

                if model.replace_existing:
                    self._logger.set_persistent_context_dict(context_dict)
                else:
                    current_context = self._logger.get_persistent_context()
                    if model.merge_strategy == "replace":
                        current_context.update(context_dict)
                    elif model.merge_strategy == "update":
                        for key, value in context_dict.items():
                            current_context[key] = value
                    elif model.merge_strategy == "merge_deep":
                        # Simple deep merge implementation
                        def deep_merge(
                            target: dict[str, object], source: dict[str, object]
                        ) -> None:
                            for key, value in source.items():
                                if (
                                    key in target
                                    and isinstance(target[key], dict)
                                    and isinstance(value, dict)
                                ):
                                    deep_merge(
                                        cast("dict[str, object]", target[key]),
                                        cast("dict[str, object]", value),
                                    )
                                else:
                                    target[key] = value

                        deep_merge(current_context, context_dict)

                return FlextResult[None].ok(None)
            except Exception as e:
                return FlextResult[None].fail(f"Failed to set permanent context: {e}")

        def bind_context(
            self, model: FlextLogger.LoggerContextBindingModel
        ) -> FlextResult[FlextLogger]:
            """Bind context to create new logger instance - optimized implementation."""
            try:
                # Create new logger instance
                attrs = self._logger.get_logger_attributes()
                new_logger = FlextLogger(
                    name=cast("str", attrs["name"]),
                    _level=cast("str", attrs["level"]),
                    _service_name=cast("str | None", attrs.get("service_name")),
                    _service_version=cast("str | None", attrs.get("service_version")),
                    _correlation_id=cast("str | None", attrs.get("correlation_id")),
                    _force_new=model.force_new_instance,
                )

                # Copy existing contexts if requested
                if model.copy_request_context and hasattr(
                    self._logger.get_local_storage(), "request_context"
                ):
                    context_model = FlextLogger.LoggerRequestContextModel(
                        **self._logger.get_local_storage().request_context
                    )
                    new_logger._context_manager.set_request_context(context_model)

                if model.copy_permanent_context:
                    persistent_model = FlextLogger.LoggerPermanentContextModel(
                        app_name="copied",
                        app_version="1.0.0",
                        environment=cast(
                            "FlextTypes.Config.Environment", "development"
                        ),
                        permanent_context=self._logger.get_persistent_context(),
                        replace_existing=True,
                    )
                    new_logger._context_manager.set_permanent_context(persistent_model)

                # Add new bound context - fix the dict type issue
                if not model.clear_existing:
                    # Create new request context model from bound data
                    context_dict = {
                        str(k): str(v) for k, v in model.context_data.items()
                    }
                    new_context_model = FlextLogger.LoggerRequestContextModel(
                        custom_data=context_dict
                    )
                    new_logger._context_manager.set_request_context(new_context_model)
                else:
                    new_logger._context_manager.clear_request_context()

                return FlextResult[FlextLogger].ok(new_logger)
            except Exception as e:
                return FlextResult[FlextLogger].fail(f"Failed to bind context: {e}")

    @staticmethod
    def _configure_structlog(
        log_level: str = FlextConstants.Logging.DEFAULT_LEVEL,
        *,
        json_output: bool | None = None,
        include_source: bool = FlextConstants.Logging.INCLUDE_SOURCE,
        log_verbosity: str = FlextConstants.Logging.VERBOSITY,
        log_file: FlextTypes.Core.Optional[FlextTypes.Identifiers.Path] = None,
        log_file_max_size: int = 10 * 1024 * 1024,  # 10MB default
        log_file_backup_count: int = 5,
        console_enabled: bool = FlextConstants.Logging.CONSOLE_ENABLED,
        console_color_enabled: bool = FlextConstants.Logging.CONSOLE_COLOR_ENABLED,
        track_performance: bool = FlextConstants.Logging.TRACK_PERFORMANCE,
        include_correlation_id: bool = FlextConstants.Logging.INCLUDE_CORRELATION_ID,
        mask_sensitive_data: bool = FlextConstants.Logging.MASK_SENSITIVE_DATA,
    ) -> None:
        """Configure structlog with enhanced settings."""
        # Configure standard library logging first
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(message)s",
            handlers=[],
        )

        # Determine JSON output preference
        use_json = json_output if json_output is not None else (not console_enabled)

        # Create processor pipeline
        processors: list[Processor] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
        ]

        # Add conditional processors
        if include_correlation_id:
            processors.append(FlextLogger._add_correlation_processor)

        if track_performance:
            processors.append(FlextLogger._add_performance_processor)

        if mask_sensitive_data:
            processors.append(FlextLogger._sanitize_processor)

        if include_source:
            processors.append(structlog.processors.CallsiteParameterAdder())

        # Add timestamper
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

        # Add final renderer
        if use_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(
                FlextLogger._create_enhanced_console_renderer(log_verbosity)
            )

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # Configure handlers
        logger = logging.getLogger()
        logger.handlers.clear()

        # Console handler
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            if console_color_enabled and use_json:
                console_handler.setFormatter(
                    colorlog.ColoredFormatter(
                        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
                        log_colors={
                            "DEBUG": "cyan",
                            "INFO": "green",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "red,bg_white",
                        },
                    )
                )
            logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=log_file_max_size,
                backupCount=log_file_backup_count,
                encoding="utf-8",
            )
            logger.addHandler(file_handler)

        # Mark as configured
        FlextLogger._configured = True

    def _validated_log_level(self, level: str) -> str:
        """Validate and return normalized log level."""
        valid_levels = FlextConstants.Logging.VALID_LEVELS
        level_upper = level.upper()
        if level_upper not in valid_levels:
            warnings.warn(
                f"Invalid log level: {level}. Using {FlextConstants.Logging.DEFAULT_LEVEL}",
                UserWarning,
                stacklevel=3,
            )
            return FlextConstants.Logging.DEFAULT_LEVEL
        return level_upper

    def _is_configured(self) -> bool:
        """Check if structlog is configured."""
        return self._configured

    def __new__(cls, name: str, **kwargs) -> FlextLogger:
        """Create logger instance with caching support.
        
        Args:
            name: Logger name
            **kwargs: Additional initialization parameters
            
        Returns:
            FlextLogger: Cached or new logger instance
        """
        # Check cache first (unless forcing new instance)
        if not kwargs.get('_force_new', False):
            with cls._cache_lock:
                if name in cls._logger_cache:
                    return cls._logger_cache[name]
        
        # Create new instance
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        name: str,
        *,
        _level: str | None = None,
        _service_name: str | None = None,
        _service_version: str | None = None,
        _correlation_id: str | None = None,
        _force_new: bool = False,
    ) -> None:
        """Initialize logger with enhanced validation and configuration.

        Args:
            name: Logger name (typically __name__ or module path)
            _level: Optional log level override
            _service_name: Optional service name override
            _service_version: Optional service version override
            _correlation_id: Optional correlation ID override
            _force_new: Force creation of new instance (for testing)

        """
        super().__init__()
        self._local = internal.invalid()
        model_log_level = _level or FlextConstants.Logging.DEFAULT_LEVEL

        # Validate using LoggerInitializationModel with exception handling
        init_model: FlextLogger.LoggerInitializationModel | None = None
        try:
            init_model = FlextLogger.LoggerInitializationModel(
                name=name,
                log_level=model_log_level,
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

            # Use get_logging_config() to get all configuration values
            logging_config = global_config.get_logging_config()

            # Configure structlog with all logging configuration parameters
            FlextLogger._configure_structlog(
                log_level=str(
                    logging_config.get("level", FlextConstants.Logging.DEFAULT_LEVEL)
                ),
                json_output=cast("bool | None", logging_config.get("json_output")),
                include_source=bool(
                    logging_config.get(
                        "include_source", FlextConstants.Logging.INCLUDE_SOURCE
                    )
                ),
                log_verbosity=str(
                    logging_config.get(
                        "log_verbosity", FlextConstants.Logging.VERBOSITY
                    )
                ),
                log_file=cast(
                    "FlextTypes.Core.Optional[FlextTypes.Identifiers.Path]",
                    logging_config.get("log_file"),
                ),
                log_file_max_size=int(
                    cast(
                        "int", logging_config.get("log_file_max_size", 10 * 1024 * 1024)
                    )
                ),
                log_file_backup_count=int(
                    cast("int", logging_config.get("log_file_backup_count", 5))
                ),
                console_enabled=bool(
                    logging_config.get(
                        "console_enabled", FlextConstants.Logging.CONSOLE_ENABLED
                    )
                ),
                console_color_enabled=bool(
                    logging_config.get(
                        "console_color_enabled",
                        FlextConstants.Logging.CONSOLE_COLOR_ENABLED,
                    )
                ),
                track_performance=bool(
                    logging_config.get(
                        "track_performance", FlextConstants.Logging.TRACK_PERFORMANCE
                    )
                ),
                include_correlation_id=bool(
                    logging_config.get(
                        "include_correlation_id",
                        FlextConstants.Logging.INCLUDE_CORRELATION_ID,
                    )
                ),
                mask_sensitive_data=bool(
                    logging_config.get(
                        "mask_sensitive_data",
                        FlextConstants.Logging.MASK_SENSITIVE_DATA,
                    )
                ),
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

        resolved_level: FlextTypes.Config.LogLevel | None = None

        # 1) Explicit parameter takes precedence when valid
        if isinstance(validated_level, str) and validated_level:
            cand = validated_level.upper()
            if cand in valid_levels:
                resolved_level = cast("FlextTypes.Config.LogLevel", cand)

        # 2) Testing default: prefer WARNING in test sessions when no explicit level
        if resolved_level is None and is_test_env:
            resolved_level = cast(
                "FlextTypes.Config.LogLevel", FlextConstants.Logging.WARNING
            )

        # Removed debug print statement for production code

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
                    resolved_level = cast("FlextTypes.Config.LogLevel", env_level_upper)

            # Fallback to global FLEXT_LOG_LEVEL
            if resolved_level is None:
                env_level = os.getenv("FLEXT_LOG_LEVEL")
                env_level_upper = (
                    env_level.upper() if isinstance(env_level, str) else None
                )
                if env_level_upper in valid_levels:
                    resolved_level = cast("FlextTypes.Config.LogLevel", env_level_upper)

        # 4) Configuration/defaults from FlextConfig singleton
        if resolved_level is None:
            cfg_level = str(config.log_level).upper()
            resolved_level = cast(
                "FlextTypes.Config.LogLevel",
                (
                    cfg_level
                    if cfg_level in valid_levels
                    else FlextConstants.Logging.DEFAULT_LEVEL
                ),
            )

        self._level = self._validated_log_level(resolved_level)

        # Removed debug print statement for production code

        # Use environment from configuration singleton for consistency
        # Environment is already typed correctly in FlextConfig
        self._environment = cast("FlextTypes.Config.Environment", config.environment)

        # Initialize service context using validated parameters
        self._service_name = validated_service_name or self._extract_service_name()
        self._service_version = validated_service_version or FlextConstants.Core.VERSION

        # Set up performance tracking
        self._start_time = time.time()

        # Instance-level correlation ID (can override global)
        context_id = None
        context_id = FlextContext.Correlation.get_correlation_id()

        self._correlation_id = (
            validated_correlation_id
            or self._global_correlation_id
            or context_id  # Check global context
            or f"corr_{str(uuid.uuid4())[:8]}"
        )

        # Set up structured logger with enriched context

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

        # Add to cache (unless forcing new instance)
        if not _force_new:
            with self._cache_lock:
                self._logger_cache[name] = self

    @override
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

    @override
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

    def _get_project_specific_env_var(
        self, suffix: str
    ) -> FlextTypes.Core.Optional[FlextTypes.Identifiers.Name]:
        """Get project-specific environment variable name for the current service.

        Args:
            suffix: Suffix to append to the environment variable name

        Returns:
            FlextTypes.Core.Optional[FlextTypes.Identifiers.Name]: Environment variable name or None if not applicable

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

    def _sanitize_context(self, context: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
        """Sanitize context by redacting sensitive data and ensuring JSON serialization.

        Args:
            context: Context dictionary to sanitize

        Returns:
            FlextTypes.Core.Dict: Sanitized context dictionary with JSON-serializable values

        """
        config = FlextConfig.get_global_instance()
        should_mask = getattr(
            config, "mask_sensitive_data", FlextConstants.Logging.MASK_SENSITIVE_DATA
        )

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

        def _serialize_value(value: object) -> object:
            """Convert value to JSON-serializable format."""
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, date):
                return value.isoformat()
            if isinstance(value, datetime_time):
                return value.isoformat()
            if isinstance(value, (Decimal, uuid.UUID, Path)):
                return str(value)
            if isinstance(value, bytes):
                return value.decode(
                    FlextConstants.Mixins.DEFAULT_ENCODING, errors="replace"
                )
            if isinstance(value, Exception):
                return {"type": value.__class__.__name__, "message": str(value)}
            if hasattr(value, "__dict__") and not isinstance(
                value, (str, int, float, bool, list, dict)
            ):
                # Convert complex objects to dict representation
                try:
                    return {
                        str(k): _serialize_value(v) for k, v in value.__dict__.items()
                    }
                except Exception:
                    return str(value)
            elif isinstance(value, (list, tuple)):
                # Type narrowing: value is confirmed to be list or tuple
                # Items in collections are still of type object
                return [
                    _serialize_value(item)
                    for item in cast("list[object] | tuple[object, ...]", value)
                ]
            elif isinstance(value, dict):
                # Type narrowing: value is confirmed to be dict
                # Keys and values in dict are still of type object
                dict_result: dict[str, object] = {}
                for k, v in cast("dict[object, object]", value).items():
                    dict_result[str(k)] = _serialize_value(v)
                return dict_result
            return value

        sanitized: FlextTypes.Core.Dict = {}
        for key, value in context.items():
            key_lower = str(key).lower()
            if should_mask and any(
                sensitive in key_lower for sensitive in sensitive_keys
            ):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(
                    cast("dict[str, object]", value)
                )
            else:
                sanitized[key] = _serialize_value(value)

        return sanitized

    def _build_log_entry(
        self,
        level: str,
        message: str,
        context: Mapping[str, object] | None = None,
        error: Exception | FlextTypes.Core.Optional[str] = None,
        duration_ms: float | None = None,
    ) -> FlextLogger.LogEntry:
        """Build structured log entry using internal LogEntry model.

        Args:
            level: Log level
            message: Log message
            context: Optional context dictionary
            error: Optional error information
            duration_ms: Optional duration in milliseconds

        Returns:
            FlextLogger.LogEntry: Structured log entry with validation

        """
        # Get current timestamp
        timestamp = self._get_current_timestamp()

        # Build service context
        service_ctx = self._persistent_context.get("service", {})
        service_data = (
            dict(cast("dict[str, object]", service_ctx))
            if isinstance(service_ctx, dict)
            else {}
        )

        # Build system context
        system_ctx = self._persistent_context.get("system", {})
        system_data = (
            dict(cast("dict[str, object]", system_ctx))
            if isinstance(system_ctx, dict)
            else {}
        )

        # Build request context - FIX: Access through local storage properly
        local_storage = self.get_local_storage()
        request_context = getattr(local_storage, "request_context", None)
        request_data = (
            dict(request_context) if isinstance(request_context, dict) else {}
        )

        # Build permanent context - FIX: Access permanent context properly
        permanent_data = dict(self._persistent_context)

        # Build execution context
        execution_info: FlextTypes.Core.Dict = {}
        config = FlextConfig.get_global_instance()
        if getattr(config, "track_timing", FlextConstants.Logging.TRACK_TIMING):
            function_name = self._get_calling_function()
            if function_name and function_name != "unknown":
                execution_info["function"] = function_name
            line_number_value = self._get_calling_line()
            if line_number_value:
                execution_info["line"] = line_number_value
            uptime_seconds = round(time.time() - self._start_time, 3)
            if uptime_seconds >= 0:
                execution_info["uptime_seconds"] = uptime_seconds

        # Build performance metrics
        performance_data: FlextTypes.Core.Dict | None = None
        if duration_ms is not None and getattr(
            config, "track_performance", FlextConstants.Logging.TRACK_PERFORMANCE
        ):
            performance_data = {
                "duration_ms": round(duration_ms, 3),
                "timestamp": timestamp,
            }

        # Build error details
        error_data: FlextTypes.Core.Dict = {}
        if error is not None:
            if isinstance(error, Exception):
                error_data = {
                    "type": error.__class__.__name__,
                    "message": str(error),
                    "details": getattr(error, "args", ()),
                }
                if error.__traceback__ is not None:
                    error_data["stack_trace"] = traceback.format_tb(error.__traceback__)
            else:
                error_data = {
                    "type": "StringError",
                    "message": str(error),
                    "stack_trace": None,
                }

        # Build context data
        context_data: FlextTypes.Core.Dict = {}
        if (
            getattr(config, "include_context", FlextConstants.Logging.INCLUDE_CONTEXT)
            and context is not None
        ):
            # Since context is Mapping[str, object] | None, we know it's a Mapping when not None
            context_dict = dict(context)

            if context_dict:
                context_data = (
                    self._sanitize_context(context_dict)
                    if getattr(
                        config,
                        "mask_sensitive_data",
                        FlextConstants.Logging.MASK_SENSITIVE_DATA,
                    )
                    else dict(context_dict)
                )

        # Determine correlation ID
        correlation_id = None
        if (
            getattr(
                config,
                "include_correlation_id",
                FlextConstants.Logging.INCLUDE_CORRELATION_ID,
            )
            and self._correlation_id
        ):
            correlation_id = self._correlation_id

        # Determine line number for the model - proper type handling
        line_number: int | None = None
        if execution_info.get("line"):
            line_number = cast("int", execution_info["line"])

        # Create internal LogEntry with individual parameters
        try:
            return FlextLogger.LogEntry(
                message=str(message),
                level=level.upper(),
                timestamp=timestamp,
                logger=self._name,
                correlation_id=correlation_id,
                context=context_data,
                function=cast("str | None", execution_info.get("function")),
                line_number=line_number,
                request=request_data,
                service=service_data,
                system=system_data,
                execution=execution_info,
                permanent=permanent_data,
                performance=performance_data,
                error=error_data,
            )
        except Exception as e:
            # Fallback to minimal entry if validation fails
            warnings.warn(
                f"LogEntry validation failed: {e}. Using minimal entry.",
                UserWarning,
                stacklevel=2,
            )
            return FlextLogger.LogEntry(
                message=str(message),
                level=level.upper(),
                timestamp=timestamp,
                logger=self._name,
            )

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

    def set_correlation_id(self, correlation_id: str) -> FlextResult[None]:
        """Set correlation ID for request tracing - enhanced with FlextResult return.

        Args:
            correlation_id: Correlation ID to set for tracing

        Returns:
            FlextResult[None]: Success or failure result with detailed error information

        """
        result = self._context_manager.set_correlation_id(correlation_id)
        if result.is_failure:
            warnings.warn(
                f"Failed to set correlation ID: {result.error}",
                UserWarning,
                stacklevel=2,
            )
        return result

    def set_request_context(self, **context: object) -> None:
        """Set request-specific context - optimized through context manager."""
        # Extract known fields from context
        model_kwargs: FlextTypes.Core.Dict = {}
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
        endpoint = (
            str(model_kwargs["endpoint"]) if model_kwargs.get("endpoint") else None
        )

        model = FlextLogger.LoggerRequestContextModel(
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

    def clear_request_context(self) -> FlextResult[None]:
        """Clear request context - enhanced with FlextResult return.

        Returns:
            FlextResult[None]: Success or failure result with detailed error information

        """
        result = self._context_manager.clear_request_context()
        if result.is_failure:
            warnings.warn(
                f"Failed to clear request context: {result.error}",
                UserWarning,
                stacklevel=2,
            )
        return result

    def bind(self, **context: object) -> FlextLogger:
        """Create logger instance with bound context - optimized through context manager.

        Args:
            **context: Context data to bind

        Returns:
            FlextLogger: New logger instance with bound context

        """
        model = FlextLogger.LoggerContextBindingModel(
            logger_name=self._name,
            context_data=context,
            bind_type="temporary",
            clear_existing=False,
            force_new_instance=True,  # Force creation of new instance for bind
            copy_permanent_context=True,  # Copy permanent context to bound logger
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

        # Handle the result properly with type checking
        if result.is_success:
            # Since result.unwrap() returns FlextLogger when successful, no isinstance check needed
            return result.unwrap()

        # Additional fallback if unwrap failed or wrong type
        return FlextLogger(
            name=self._name,
            _level=self._level,
            _service_name=getattr(self, "_service_name", None),
            _service_version=getattr(self, "_service_version", None),
            _correlation_id=getattr(self, "_correlation_id", None),
            _force_new=True,
        )

    def set_context(
        self,
        context_dict: FlextTypes.Core.Dict | None = None,
        **context: object,
    ) -> None:
        """Set persistent context data - optimized through context manager."""
        # Merge context_dict and kwargs
        final_context: FlextTypes.Core.Dict = {}
        if context_dict is not None:
            final_context.update(context_dict)
        final_context.update(context)

        # Since LoggerPermanentContextModel doesn't exist, skip the model-based approach
        # and directly set the context using a simpler method
        try:
            # Directly update persistent context without model validation
            if not hasattr(self, "_persistent_context"):
                self._persistent_context = {}

            if context_dict is not None:
                # Replace mode
                self._persistent_context = final_context.copy()
            else:
                # Update mode
                self._persistent_context.update(final_context)

            result = FlextResult[None].ok(None)
        except Exception as e:
            result = FlextResult[None].fail(f"Failed to set persistent context: {e}")
        if result.is_failure:
            warnings.warn(
                f"Failed to set persistent context: {result.error}",
                UserWarning,
                stacklevel=2,
            )

    def with_context(self, **context: object) -> FlextLogger:
        """Create logger instance with additional context - enhanced fluent interface.

        This method supports fluent chaining for building complex logging contexts:
        logger.with_context(user_id="123").with_context(request_id="abc").info("Processing request")

        Args:
            **context: Context data to add

        Returns:
            FlextLogger: New logger instance with additional context for method chaining

        """
        return self.bind(**context)

    def with_operation_context(
        self, operation_name: FlextTypes.Identifiers.Name, **context: object
    ) -> FlextResult[FlextLogger]:
        """Enhanced method combining operation tracking with context binding.

        This method demonstrates advanced FLEXT patterns by combining:
        - FlextTypes for enhanced type safety
        - FlextResult for explicit error handling
        - Context management through validated models
        - Fluent interface support

        Args:
            operation_name: Name of the operation to track
            **context: Additional context data for the operation

        Returns:
            FlextResult[FlextLogger]: Success with bound logger or failure with error details

        Example:
            result = logger.with_operation_context("user_registration", user_id="123")
            if result.is_success:
                op_logger = result.unwrap()
                op_logger.info("Operation starting")

        """
        try:
            # Create operation context with enhanced tracking
            operation_context = {
                "operation_name": operation_name,
                "operation_id": f"op_{uuid.uuid4().hex[:8]}",
                "start_time": time.time(),
                **context,
            }

            # Use the existing bind method with operation context
            bound_logger = self.bind(**operation_context)

            # Log operation start with structured context
            bound_logger.info(
                f"Operation started: {operation_name}", **operation_context
            )

            return FlextResult[FlextLogger].ok(bound_logger)

        except Exception as e:
            return FlextResult[FlextLogger].fail(
                f"Failed to create operation context: {e}",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
            )

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
        operations_dict = cast(
            "dict[str, dict[str, object]]", getattr(self._local, "operations", {})
        )
        operations_dict[operation_id] = {
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
        entry_dict = entry.to_dict()
        self._structlog_logger.debug(
            formatted_message,
            **entry_dict,
        )  # Use debug since structlog doesn't have trace  # Use debug since structlog doesn't have trace

    def start_trace(
        self, operation_name: FlextTypes.Identifiers.Name, **context: object
    ) -> FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]:
        """Start a distributed trace if tracing is enabled.

        Args:
            operation_name: Name of the operation being traced
            **context: Additional context for the trace

        Returns:
            FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]: Trace ID if tracing enabled, None otherwise

        """
        config = FlextConfig.get_global_instance()
        if not getattr(config, "enable_tracing", FlextConstants.Logging.ENABLE_TRACING):
            return None

        # Simple trace implementation - generate trace ID and log start

        trace_id = str(uuid.uuid4())[:8]

        self.debug(
            f"TRACE_START: {operation_name}",
            trace_id=trace_id,
            operation=operation_name,
            **context,
        )
        return trace_id

    def end_trace(
        self,
        trace_id: FlextTypes.Core.Optional[FlextTypes.Identifiers.Id],
        operation_name: FlextTypes.Identifiers.Name,
        **context: object,
    ) -> None:
        """End a distributed trace if tracing is enabled.

        Args:
            trace_id: Trace ID from start_trace
            operation_name: Name of the operation being traced
            **context: Additional context for the trace

        """
        config = FlextConfig.get_global_instance()
        if (
            not getattr(config, "enable_tracing", FlextConstants.Logging.ENABLE_TRACING)
            or trace_id is None
        ):
            return

        self.debug(
            f"TRACE_END: {operation_name}",
            trace_id=trace_id,
            operation=operation_name,
            **context,
        )

    def debug(self, message: str, *args: object, **context: object) -> None:
        """Log debug message - LoggerProtocol implementation."""
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            # Simple mode: let structlog handle message formatting
            self._structlog_logger.debug(message, *args, **context)
        else:
            # Structured mode: build complete log entry
            formatted_message = message % args if args else message
            entry = self._build_log_entry(
                FlextConstants.Logging.DEBUG, formatted_message, context
            )
            # Convert to dict for structured logging, excluding message to avoid conflicts
            entry_dict = entry.to_dict()
            # Remove message field to avoid conflict with structlog's event parameter
            entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
            # Use message as event parameter for structlog
            self._structlog_logger.debug(formatted_message, **entry_context)

    def info(self, message: str, *args: object, **context: object) -> None:
        """Log info message - LoggerProtocol implementation."""
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            # Simple mode: let structlog handle message formatting
            self._structlog_logger.info(message, *args, **context)
        else:
            # Structured mode: build complete log entry
            formatted_message = message % args if args else message
            entry = self._build_log_entry(
                FlextConstants.Logging.INFO, formatted_message, context
            )
            # Convert to dict for structured logging, excluding message to avoid conflicts
            entry_dict = entry.to_dict()
            # Remove message field to avoid conflict with structlog's event parameter
            entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
            # Use message as event parameter for structlog
            self._structlog_logger.info(formatted_message, **entry_context)

    def warning(self, message: str, *args: object, **context: object) -> None:
        """Log warning message - LoggerProtocol implementation."""
        # Get structured_output setting from FlextConfig singleton
        global_config = FlextConfig.get_global_instance()
        structured_output = getattr(
            global_config, "structured_output", FlextConstants.Logging.STRUCTURED_OUTPUT
        )

        if not structured_output:
            # Simple mode: let structlog handle message formatting
            self._structlog_logger.warning(message, *args, **context)
        else:
            # Structured mode: build complete log entry
            formatted_message = message % args if args else message
            entry = self._build_log_entry(
                FlextConstants.Logging.WARNING, formatted_message, context
            )
            # Convert to dict for structured logging, excluding message to avoid conflicts
            entry_dict = entry.to_dict()
            # Remove message field to avoid conflict with structlog's event parameter
            entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
            # Use message as event parameter for structlog
            self._structlog_logger.warning(formatted_message, **entry_context)

    def warn(self, message: str, *args: object, **context: object) -> None:
        """Log warning message - alias for warning method for compatibility.

        This method provides compatibility with standard Python logging interface
        which supports both warn() and warning() methods.
        """
        self.warning(message, *args, **context)

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
            # Simple mode: pass context in **kwargs, error as part of context
            if error:
                context["error"] = str(error)
            # structlog expects message as event parameter (first positional)
            self._structlog_logger.error(formatted_message, **context)
        else:
            # Structured mode: build complete log entry
            entry = self._build_log_entry(
                FlextConstants.Config.LogLevel.ERROR,
                formatted_message,
                context,
                error,
            )
            # Convert to dict for structured logging, excluding message to avoid conflicts
            entry_dict = entry.to_dict()
            # Remove message field to avoid conflict with structlog's event parameter
            entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
            # Use message as event parameter for structlog
            self._structlog_logger.error(formatted_message, **entry_context)

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
            # Simple mode: pass context in **kwargs, error as part of context
            if error:
                context["error"] = str(error)
            # structlog expects message as event parameter (first positional)
            self._structlog_logger.critical(formatted_message, **context)
        else:
            # Structured mode: build complete log entry
            entry = self._build_log_entry(
                FlextConstants.Config.LogLevel.CRITICAL,
                formatted_message,
                context,
                error,
            )
            # Convert to dict for structured logging, excluding message to avoid conflicts
            entry_dict = entry.to_dict()
            # Remove message field to avoid conflict with structlog's event parameter
            entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
            # Use message as event parameter for structlog
            self._structlog_logger.critical(formatted_message, **entry_context)

    def exception(
        self,
        message: str,
        *args: object,
        exc_info: bool = True,
        **context: object,
    ) -> None:
        """Log exception with stack trace and context - LoggerProtocol implementation."""
        formatted_message = message % args if args else message
        error: Exception | None = None
        if exc_info:
            _, exc_value, _ = sys.exc_info()
            if isinstance(exc_value, Exception):
                error = exc_value
        entry = self._build_log_entry(
            FlextConstants.Config.LogLevel.ERROR, formatted_message, context, error
        )
        # Convert to dict for structured logging, excluding message to avoid conflicts
        entry_dict = entry.to_dict()
        # Remove message field to avoid conflict with structlog's event parameter
        entry_context = {k: v for k, v in entry_dict.items() if k != "message"}
        self._structlog_logger.error(formatted_message, **entry_context)

    # _load_config_flags method removed - configuration now accessed directly from FlextConfig

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
        config = FlextConfig.get_global_instance()
        if (
            getattr(
                config,
                "include_correlation_id",
                FlextConstants.Logging.INCLUDE_CORRELATION_ID,
            )
            and FlextLogger._global_correlation_id
        ):
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
        config = FlextConfig.get_global_instance()
        if not getattr(
            config, "mask_sensitive_data", FlextConstants.Logging.MASK_SENSITIVE_DATA
        ):
            return event_dict

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
    def set_global_correlation_id(
        cls, correlation_id: FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]
    ) -> None:
        """Set global correlation ID."""
        cls._global_correlation_id = correlation_id

    @classmethod
    def get_global_correlation_id(
        cls,
    ) -> FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]:
        """Get global correlation ID.

        Returns:
            FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]: Global correlation ID or None if not set

        """
        return cls._global_correlation_id

    def get_correlation_id(self) -> FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]:
        """Get instance correlation ID.

        Returns:
            FlextTypes.Core.Optional[FlextTypes.Identifiers.Id]: Instance correlation ID or None if not set

        """
        return self._correlation_id

    def set_correlation_id_internal(self, correlation_id: str) -> None:
        """Set instance correlation ID (internal use).

        Args:
            correlation_id: Correlation ID to set

        """
        self._correlation_id = correlation_id

    def get_local_storage(self) -> internal.invalid:
        """Get thread-local storage.

        Returns:
            internal.invalid: Thread-local storage instance

        """
        return self._local

    def get_persistent_context(self) -> FlextTypes.Core.Dict:
        """Get persistent context dictionary.

        Returns:
            FlextTypes.Core.Dict: Persistent context dictionary

        """
        if not hasattr(self, "_persistent_context"):
            self._persistent_context = {}
        return self._persistent_context

    def set_persistent_context_dict(self, context: FlextTypes.Core.Dict) -> None:
        """Set persistent context dictionary (internal use).

        Args:
            context: Context dictionary to set

        """
        self._persistent_context = context

    def get_logger_attributes(self) -> FlextTypes.Core.Dict:
        """Get logger attributes for binding.

        Returns:
            FlextTypes.Core.Dict: Logger attributes dictionary

        """
        return {
            "name": self._name,
            "level": self._level,
            "service_name": getattr(self, "_service_name", None),
            "service_version": getattr(self, "_service_version", None),
            "correlation_id": getattr(self, "_correlation_id", None),
        }

    @classmethod
    def get_configuration(cls) -> FlextTypes.Core.Dict:
        """Get current logging configuration from FlextConfig singleton.

        Returns:
            Dictionary with current configuration settings from FlextConfig

        """
        global_config = FlextConfig.get_global_instance()
        return global_config.get_logging_config()

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured.

        Returns:
            bool: True if logging is configured, False otherwise.

        """
        return cls._configured

    # Public accessor methods for testing
    @property
    def name(self) -> str:
        """Get logger name."""
        return self._name

    @property
    def level(self) -> str:
        """Get logger level."""
        return self._level

    @property
    def service_name(self) -> str:
        """Get service name."""
        return getattr(self, "_service_name", "")

    @property
    def service_version(self) -> str | None:
        """Get service version."""
        return getattr(self, "_service_version", None)

    @property
    def environment(self) -> str:
        """Get environment."""
        return getattr(self, "_environment", "")

    def get_instance_id(self) -> str:
        """Get logger instance ID."""
        return f"{self._name}_{id(self)}"

    def get_current_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now(UTC).isoformat()

    def build_log_entry(
        self,
        level: str,
        message: str,
        context: dict[str, object] | None = None,
        error: Exception | str | None = None,
        duration_ms: float | None = None,
        **_kwargs: object,
    ) -> FlextLogger.LogEntry:
        """Build log entry with provided parameters."""
        # Use the internal _build_log_entry method to ensure proper context handling
        return self._build_log_entry(level, message, context, error, duration_ms)

    def sanitize_context(self, context: dict[str, object]) -> dict[str, object]:
        """Sanitize context data by removing sensitive information."""
        # Enhanced implementation to match test expectations
        sanitized: dict[str, object] = {}

        # Extended list of sensitive keywords to match test expectations
        sensitive_keywords = [
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
        ]

        for key, value in context.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keywords):
                sanitized[key] = "[REDACTED]"  # Use [REDACTED] format to match tests
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_context(cast("dict[str, object]", value))
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def configure(cls, **kwargs: object) -> FlextResult[bool]:
        """Configure logging with the provided settings.

        Args:
            **kwargs: Configuration parameters (reserved for future use)

        Returns:
            FlextResult[bool]: Success result with configuration status

        """
        try:
            # Note: kwargs is reserved for future configuration options
            # Currently, we just set the configured flag
            _ = kwargs  # Explicitly acknowledge kwargs to avoid unused argument warning

            # Set configured flag
            cls._configured = True
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(f"Failed to configure logging: {e}")

    # LogEntry moved to internal models for circular dependency resolution


__all__: FlextTypes.Core.StringList = [
    "FlextLogger",
]
