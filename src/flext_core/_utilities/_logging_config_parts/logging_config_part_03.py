"""Structlog configuration and processor chain building.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from .logging_config_part_02 import (
    FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfigPart02,
)

if TYPE_CHECKING:
    from structlog.types import Processor

    from flext_core import FlextProtocols as p, FlextTypes as t


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingConfigPart02):
    @classmethod
    def configure_structlog(
        cls,
        *,
        settings: p.BaseModel | None = None,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: t.SequenceOf[Processor] | None = None,
        wrapper_class_factory: t.LoggerWrapperFactory | None = None,
        logger_factory: t.LoggerFactory = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults."""
        if cls._structlog_configured:
            return
        (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        ) = cls._resolve_structlog_params(
            settings,
            log_level=log_level,
            console_renderer=console_renderer,
            additional_processors=additional_processors,
            wrapper_class_factory=wrapper_class_factory,
            logger_factory=logger_factory,
            cache_logger_on_first_use=cache_logger_on_first_use,
        )
        processors = cls._build_structlog_processors(
            console_renderer=console_renderer,
            additional_processors=additional_processors,
        )
        wrapper_arg = (
            wrapper_class_factory()
            if wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(level)
        )
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory, async_logging=async_logging
        )
        configure_fn = getattr(structlog, "configure", None)
        if configure_fn is not None and callable(configure_fn):
            _ = configure_fn(
                processors=processors,
                wrapper_class=wrapper_arg,
                logger_factory=factory_to_use if callable(factory_to_use) else None,
                cache_logger_on_first_use=cache_logger_on_first_use,
            )
        cls._structlog_configured = True

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if not cls._structlog_configured:
            cls.configure_structlog()
            cls._structlog_configured = True


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
