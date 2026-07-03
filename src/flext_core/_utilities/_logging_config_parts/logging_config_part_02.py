"""Structlog configuration and processor chain building.

Extracted from FlextLogger as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys
import types
import typing
from contextlib import suppress

import structlog
from structlog.processors import JSONRenderer, StackInfoRenderer, TimeStamper
from structlog.stdlib import add_log_level
from structlog.types import Processor

from flext_core import (
    FlextConstants as c,
    FlextModelsPydantic as mp,
    FlextProtocols as p,
    FlextTypes as t,
)

from .logging_config_part_01 import (
    FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfigPart01,
)


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingConfigPart01):
    @staticmethod
    def structlog() -> types.ModuleType:
        """Return the imported structlog module."""
        return structlog

    @staticmethod
    def level_based_context_filter(
        logger: p.Logger | None,
        method_name: str,
        event_dict: t.ScalarMapping,
    ) -> t.ScalarMapping:
        """Filter context variables based on log level."""
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
        filtered_dict: t.MutableConfigurationMapping = {}
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
    def _resolve_structlog_params(
        settings: mp.BaseModel | None,
        *,
        log_level: int | None,
        console_renderer: bool,
        additional_processors: t.SequenceOf[Processor] | None,
        wrapper_class_factory: t.LoggerWrapperFactory | None,
        logger_factory: t.LoggerFactory,
        cache_logger_on_first_use: bool,
    ) -> tuple[
        int,
        bool,
        t.SequenceOf[Processor] | None,
        t.LoggerWrapperFactory | None,
        t.LoggerFactory,
        bool,
        bool,
    ]:
        """Extract structlog params from settings model or pass-through args."""
        async_logging = True
        if settings is not None:
            log_level = getattr(settings, "log_level", log_level)
            console_renderer = getattr(settings, "console_renderer", console_renderer)
            cfg_processors = getattr(settings, "additional_processors", None)
            if cfg_processors:
                additional_processors = cfg_processors
            wrapper_class_factory = getattr(
                settings,
                "wrapper_class_factory",
                wrapper_class_factory,
            )
            logger_factory = getattr(settings, "logger_factory", logger_factory)
            cache_logger_on_first_use = getattr(
                settings,
                "cache_logger_on_first_use",
                cache_logger_on_first_use,
            )
            async_logging = getattr(settings, "async_logging", True)
        level = log_level if log_level is not None else logging.INFO
        return (
            level,
            console_renderer,
            additional_processors,
            wrapper_class_factory,
            logger_factory,
            cache_logger_on_first_use,
            async_logging,
        )

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        additional_processors: t.SequenceOf[Processor] | None,
    ) -> t.SequenceOf[Processor]:
        """Assemble the structlog processor chain."""
        processors: t.MutableSequenceOf[Processor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
        ]
        if additional_processors:
            processors.extend(proc for proc in additional_processors if callable(proc))
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
        """Resolve the logger factory, enabling async output when requested."""
        if logger_factory is not None:
            return logger_factory
        if async_logging:
            with suppress(AttributeError):
                print_logger_factory = structlog.PrintLoggerFactory
                if callable(print_logger_factory):
                    return cls._build_async_logger_factory(print_logger_factory)
            with suppress(AttributeError):
                write_logger_factory = structlog.WriteLoggerFactory
                if callable(write_logger_factory):
                    return cls._build_async_logger_factory(write_logger_factory)
        return None

    @classmethod
    def _build_async_logger_factory(
        cls,
        factory_builder: typing.Callable[..., t.LoggerFactory],
    ) -> t.LoggerFactory:
        """Build a structlog logger factory bound to the shared async writer."""
        if cls._async_writer is None:
            cls._async_writer = cls._AsyncLogWriter(sys.stdout)
        return factory_builder(file=cls._async_writer)


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
