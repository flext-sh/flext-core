"""Structlog configuration and processor chain building.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging

import structlog

from flext_core import FlextSettings, c, m, p, t
from flext_core._runtime import FlextRuntimeBase
from flext_core._utilities._logging_config_parts import (
    logging_config_part_01,
    logging_config_part_02,
)


class FlextUtilitiesLoggingConfig(logging_config_part_02.FlextUtilitiesLoggingConfig):
    @staticmethod
    def logging_settings() -> p.LoggingSettings:
        """Return the elected application provider, or the actual Core settings."""
        selected = logging_config_part_01.FlextUtilitiesLoggingConfig._runtime_settings
        return selected if selected is not None else FlextSettings.fetch_global()

    @staticmethod
    def logging_options() -> p.StructlogOptions:
        """Return the active composition for scoped lifecycle restoration."""
        selected = logging_config_part_01.FlextUtilitiesLoggingConfig._options
        return selected if selected is not None else m.StructlogOptions()

    @classmethod
    def resolve_logging_settings(
        cls,
        *,
        runtime_settings: p.LoggingSettings | None = None,
        log_level: str | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
        logging_profile: str | None = None,
    ) -> p.LoggingSettings:
        """Validate optional overrides without I/O or process-wide mutation."""
        current = (
            runtime_settings if runtime_settings is not None else cls.logging_settings()
        )
        overrides: t.MutableMappingKV[str, t.SettingsOverride | None] = {
            key: value
            for key, value in (
                ("log_level", log_level),
                ("debug", debug),
                ("trace", trace),
                ("logging_profile", logging_profile),
            )
            if value is not None
        }
        selected = current.clone(**overrides)
        level = FlextRuntimeBase.resolve_effective_log_level(
            log_level=c.LogLevel(
                str(logging.getLevelName(cls.level_number(selected.log_level)))
            ),
            debug=selected.debug,
            trace=selected.trace,
            log_level_explicit=(
                log_level is not None
                or (debug is None and "log_level" in current.model_fields_set)
            ),
        )
        return selected.clone(log_level=level.value)

    @classmethod
    def create_logging_handler(
        cls,
        *,
        stream: t.TextStream | None = None,
        console_renderer: bool = False,
        log_level: str | None = None,
    ) -> p.LoggingHandler:
        """Compose native per-destination presentation with fail-loud delivery."""
        handler = cls._StreamHandler(stream)
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=(
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False)
                    if console_renderer
                    else structlog.processors.JSONRenderer(),
                )
            )
        )
        if log_level is not None:
            handler.setLevel(cls.level_number(log_level))
        return handler

    @classmethod
    def configure_structlog(
        cls,
        *,
        settings: p.StructlogOptions | None = None,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> None:
        """Initialize defaults once or apply explicitly supplied typed options.

        Cached public loggers lazily consume the same elected composition.
        Caller-supplied handlers retain ownership of their delivery lifecycle.
        """
        owner = logging_config_part_01.FlextUtilitiesLoggingConfig
        if owner._options is not None and settings is None and runtime_settings is None:
            return
        options = settings if settings is not None else cls.logging_options()
        selected = cls.resolve_logging_settings(runtime_settings=runtime_settings)
        if runtime_settings is not None and options.log_level is not None:
            if options.log_level != cls.level_number(selected.log_level):
                options = m.StructlogOptions.model_validate(
                    options,
                    from_attributes=True,
                ).model_copy(update={"log_level": None})
        elif options.log_level is not None:
            selected = cls.resolve_logging_settings(
                runtime_settings=selected,
                log_level=str(logging.getLevelName(options.log_level)),
            )
        if options.handlers and options.logger_factory is not None:
            msg = "Select native handlers or a logger factory, not both"
            raise ValueError(msg)
        native_handlers: t.LoggingNativeHandlers = []
        for handler in options.handlers:
            if not isinstance(handler, logging.Handler):
                msg = "Selected handlers must implement the native logging.Handler boundary"
                raise TypeError(msg)
            if not isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter):
                msg = "Selected handlers require a native ProcessorFormatter"
                raise ValueError(msg)
            if type(handler).handleError is logging.Handler.handleError:
                msg = "Selected handlers must propagate sink failures; use create_logging_handler"
                raise ValueError(msg)
            native_handlers.append(handler)
        threshold = cls.level_number(selected.log_level)
        processors = cls._build_structlog_processors(
            console_renderer=(
                options.console_renderer
                if options.console_renderer is not None
                else selected.logging_profile == "console"
            ),
            options=options,
            trace=selected.trace,
        )
        # The threshold is enforced by ``drop_below_threshold`` at emit time,
        # so the bound logger itself must not filter statically.
        wrapper_arg = (
            options.wrapper_class_factory()
            if options.wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(logging.NOTSET)
        )
        logger_factory = options.logger_factory
        if options.handlers:

            def native_factory(*names: str) -> p.OutputLogger:
                output = logging.getLogger(
                    names[0] if names else c.LOGGER_NAME_FLEXT_CORE,
                    level=logging.NOTSET,
                )
                output.handlers = native_handlers
                output.propagate = False
                return output

            logger_factory = native_factory
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory,
            async_logging=(
                options.async_logging
                if options.async_logging is not None
                else selected.async_logging
            ),
        )
        structlog.configure(
            processors=processors,
            wrapper_class=wrapper_arg,
            logger_factory=factory_to_use,
            cache_logger_on_first_use=options.cache_logger_on_first_use,
        )
        owner._runtime_settings = (
            runtime_settings if runtime_settings is not None else selected
        )
        owner._options = options
        owner._log_threshold = threshold

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if logging_config_part_01.FlextUtilitiesLoggingConfig._options is None:
            cls.configure_structlog()

    @staticmethod
    def level_number(level: int | str) -> int:
        """Validate a standard severity and return its native number."""
        value = (
            level
            if isinstance(level, int)
            else logging.getLevelNamesMapping().get(level.strip().upper())
        )
        if value is None or logging.getLevelName(value) not in c.LogLevel:
            msg = f"Invalid log level: {level}"
            raise ValueError(msg)
        return value

    @classmethod
    def apply_log_level(
        cls,
        *,
        log_level: str | None = None,
        debug: bool | None = None,
        trace: bool | None = None,
    ) -> None:
        """Apply the effective level of settings values to every logger.

        Resolution is owned by ``resolve_effective_log_level``; the result
        reaches loggers that were already created and cached.
        """
        selected = cls.resolve_logging_settings(
            log_level=log_level,
            debug=debug,
            trace=trace,
        )
        cls.configure_structlog(runtime_settings=selected)


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
