"""Structlog configuration and processor chain building.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging

import structlog

from flext_core import c, p
from flext_core._runtime._base import FlextRuntimeBase
from flext_core._utilities._logging_config_parts.logging_config_part_02 import (
    FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfigPart02,
)


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingConfigPart02):
    @classmethod
    def configure_structlog(
        cls,
        *,
        settings: p.StructlogOptions | None = None,
    ) -> None:
        """Initialize defaults once or apply explicitly supplied typed options.

        Explicit configuration applies to subsequently constructed loggers.
        Already cached bound loggers retain their processor chain.
        """
        if cls._structlog_configured and settings is None:
            return
        (
            level,
            console_renderer,
            processing_stages,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        ) = cls._resolve_structlog_params(settings)
        threshold = cls.level_number(level)
        processors = cls._build_structlog_processors(
            console_renderer=console_renderer,
            processing_stages=processing_stages,
        )
        # The threshold is enforced by ``drop_below_threshold`` at emit time,
        # so the bound logger itself must not filter statically.
        wrapper_arg = (
            wrapper_class_factory()
            if wrapper_class_factory is not None
            else structlog.make_filtering_bound_logger(logging.NOTSET)
        )
        factory_to_use = cls._resolve_logger_factory(
            logger_factory=logger_factory,
            async_logging=async_logging,
        )
        structlog.configure(
            processors=processors,
            wrapper_class=wrapper_arg,
            logger_factory=factory_to_use,
            cache_logger_on_first_use=cache_logger_on_first_use,
        )
        cls._publish_logging_state(configured=True, threshold=threshold)

    @classmethod
    def ensure_structlog_configured(cls) -> None:
        """Ensure structlog is configured (called automatically on first use)."""
        if not cls._structlog_configured:
            cls.configure_structlog()

    @staticmethod
    def level_number(level: int | str) -> int:
        """Return the stdlib number of a level given by number or name.

        Returns:
            The stdlib number of a level given by number or name.

        """
        if isinstance(level, int):
            return level
        return logging.getLevelNamesMapping()[level.upper()]

    @classmethod
    def apply_log_level(cls, *, log_level: str, debug: bool, trace: bool) -> None:
        """Apply the effective level of settings values to every logger.

        Resolution is owned by ``resolve_effective_log_level``; the result
        reaches loggers that were already created and cached.
        """
        cls.ensure_structlog_configured()
        effective = FlextRuntimeBase.resolve_effective_log_level(
            trace=trace,
            debug=debug,
            log_level=c.LogLevel(log_level.upper()),
        )
        cls._publish_logging_state(
            configured=True,
            threshold=cls.level_number(effective),
        )


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
