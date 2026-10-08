"""Structlog processor chain assembly and level-based context filtering.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1). Handles processor construction,
structlog parameter resolution, and async writer lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level

from flext_core import c, p, t
from flext_core._models.config import FlextModelsConfig
from flext_core._utilities._logging_config_parts.logging_config_part_01 import (
    FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfigPart01,
)


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingConfigPart01):
    @staticmethod
    def level_based_context_filter(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: t.LoggingEvent,
    ) -> t.LoggingEvent:
        """Filter context variables based on log level.

        Returns:
            The resulting typed event mapping.

        """
        level_hierarchy = {
            "debug": 10,
            "info": 20,
            "warning": 30,
            c.WarningLevel.ERROR: 40,
            "critical": 50,
        }
        logger_level_attr = (
            getattr(logger, "level", None) if logger is not None else None
        )
        current_level = (
            logger_level_attr
            if isinstance(logger_level_attr, int)
            else level_hierarchy.get(method_name.lower(), 20)
        )
        filtered_dict: t.LoggingEvent = {}
        for key, value in event_dict.items():
            if not key.startswith("_level_"):
                filtered_dict[key] = value
                continue
            parts = key.split("_", c.DEFAULT_MAX_WORKERS)
            if len(parts) < c.DEFAULT_MAX_WORKERS:
                filtered_dict[key] = value
                continue
            required_level = level_hierarchy.get(parts[2].lower(), 10)
            if current_level < required_level:
                continue
            actual_key = parts[3]
            normalized_key = "settings" if actual_key == "config" else actual_key
            filtered_dict[normalized_key] = value
        return filtered_dict

    @staticmethod
    def drop_below_threshold(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: t.LoggingEvent,
    ) -> t.LoggingEvent:
        """Drop events under the active threshold, read at emit time.

        Reading the threshold per event keeps it re-applicable after loggers
        were cached by ``cache_logger_on_first_use``.

        Returns:
            The resulting typed event mapping.

        Raises:
            DropEvent: If ``level < FlextUtilitiesLoggingConfigPart01.log_threshold()``.

        """
        _ = logger, method_name
        level = logging.getLevelNamesMapping()[str(event_dict["level"]).upper()]
        if level < FlextUtilitiesLoggingConfigPart01.log_threshold():
            raise structlog.DropEvent
        return event_dict

    @staticmethod
    def _resolve_structlog_params(
        settings: p.StructlogOptions | None,
    ) -> tuple[
        int,
        bool,
        t.SequenceOf[p.LoggingStage],
        t.LoggerWrapperFactory | None,
        t.LoggerFactory,
        bool,
        bool,
    ]:
        """Extract structlog params from the settings model over FLEXT defaults.

        Returns:
            The declared options, with the typed stages in registration order.

        """
        options = (
            settings if settings is not None else FlextModelsConfig.StructlogOptions()
        )
        level = options.log_level if options.log_level is not None else logging.INFO
        return (
            level,
            options.console_renderer,
            options.processing_stages,
            options.wrapper_class_factory,
            options.logger_factory,
            options.cache_logger_on_first_use,
            options.async_logging,
        )

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        processing_stages: t.SequenceOf[p.LoggingStage],
    ) -> t.SequenceOf[t.LoggingProcessor]:
        """Assemble the structlog processor chain.

        Returns:
            The event stages followed by one terminal renderer.

        """
        processors: t.MutableSequenceOf[t.LoggingProcessor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.drop_below_threshold,
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
        ]
        processors.extend(processing_stages)
        if console_renderer:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(JSONRenderer())
        return processors

    @classmethod
    def _resolve_logger_factory(
        cls,
        *,
        logger_factory: t.LoggerFactory,
        async_logging: bool,
    ) -> t.LoggerFactory | None:
        """Resolve the logger factory, enabling async output when requested.

        Returns:
            The resulting ``t.LoggerFactory | None``.

        """
        if logger_factory is not None:
            return logger_factory
        if not async_logging:
            return None
        if cls._async_writer is None:
            cls._async_writer = cls._AsyncLogWriter(sys.stdout)
        return structlog.PrintLoggerFactory(file=cls._async_writer)


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
