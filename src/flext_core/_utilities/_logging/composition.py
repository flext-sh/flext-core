"""Structlog composition and elected output-factory resolution.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

from flext_core import c, m, p
from flext_core._utilities._logging.async_writer import FlextUtilitiesLoggingAsyncWriter
from flext_core._utilities._logging.handlers import FlextUtilitiesLoggingHandlers
from flext_core._utilities._logging.state import FlextUtilitiesLoggingState
from flext_core._utilities._logging.stream import FlextUtilitiesLoggingStream

if TYPE_CHECKING:
    import types


class FlextUtilitiesLoggingComposition(FlextUtilitiesLoggingHandlers):
    """Own the process and scoped structlog composition boundaries."""

    @staticmethod
    def structlog() -> types.ModuleType:
        """Return the imported structlog module for owner-internal access.

        Returns:
            The imported structlog module for owner-internal access.

        """
        return structlog

    @staticmethod
    def _resolve_logger_factory(
        *,
        logger_factory: p.OutputLoggerFactory | None,
        async_logging: bool,
    ) -> p.OutputLoggerFactory | None:
        """Keep the selected native factory or the acknowledged stderr writer.

        Returns:
            The selected output factory.

        """
        state = FlextUtilitiesLoggingState.current()
        if (
            not async_logging or logger_factory is not None
        ) and state.async_writer is not None:
            # Existing runtime loggers still own bindings to this transport.
            # A new composition drains it within the shutdown deadline, so a
            # blocked stream fails the transition instead of hanging it;
            # process shutdown closes it.
            state.async_writer.drain()
        if logger_factory is not None:
            return logger_factory
        if not async_logging:
            return structlog.PrintLoggerFactory(
                file=FlextUtilitiesLoggingStream.TextStream(sys.stderr)
            )
        if state.async_writer is None:
            state.async_writer = FlextUtilitiesLoggingAsyncWriter(sys.stderr)
        return structlog.PrintLoggerFactory(file=state.async_writer)

    @classmethod
    def compose_native_logger(
        cls,
        name: str,
        *,
        runtime_settings: p.LoggingSettings,
        options: p.StructlogOptions,
    ) -> p.NativeLoggingWrapper:
        """Compose a scoped logger without replacing the process configuration.

        Returns:
            A binding whose processors, threshold and sink remain runtime-owned.

        Raises:
            ValueError: If no output factory can be elected.

        """
        console = (
            options.console_renderer
            if options.console_renderer is not None
            else runtime_settings.logging_profile == c.LoggingProfile.CONSOLE
        )
        wrapper = (
            options.wrapper_class_factory()
            if options.wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(logging.NOTSET)
        )
        factory = options.logger_factory
        if options.handlers:
            factory = cls._handler_factory(options)
        if factory is None:
            factory = cls._resolve_logger_factory(
                logger_factory=None,
                async_logging=runtime_settings.async_logging
                if options.async_logging is None
                else options.async_logging,
            )
        if factory is None:
            msg = "Logging composition did not elect an output factory"
            raise ValueError(msg)
        return structlog.wrap_logger(
            factory(name),
            processors=cls._build_structlog_processors(
                console_renderer=console,
                options=options,
                trace=runtime_settings.trace,
                runtime_settings=runtime_settings,
                debug=runtime_settings.debug,
            ),
            wrapper_class=wrapper,
            cache_logger_on_first_use=options.cache_logger_on_first_use,
        ).bind()

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
        owner = FlextUtilitiesLoggingState
        if (
            owner.current().options is not None
            and settings is None
            and runtime_settings is None
        ):
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
                log_level=logging.getLevelName(options.log_level),
            )
        threshold = cls.level_number(selected.log_level)
        processors = cls._build_structlog_processors(
            console_renderer=(
                options.console_renderer
                if options.console_renderer is not None
                else selected.logging_profile == c.LoggingProfile.CONSOLE
            ),
            options=options,
            trace=selected.trace,
            debug=selected.debug,
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
            logger_factory = cls._handler_factory(options)
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
        owner.configure_logging_state(
            runtime_settings if runtime_settings is not None else selected,
            options,
            threshold,
        )

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if FlextUtilitiesLoggingState.current().options is None:
            cls.configure_structlog()

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


__all__: list[str] = ["FlextUtilitiesLoggingComposition"]
