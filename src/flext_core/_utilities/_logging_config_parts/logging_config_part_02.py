"""Structlog processor chain assembly and level-based context filtering.

Extracted from FlextUtilitiesLogging as an MRO mixin under the configured
source-size budget. Handles processor construction,
structlog parameter resolution, and async writer lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import sys
from functools import partial

import structlog
from structlog.processors import (
    CallsiteParameterAdder,
    ExceptionRenderer,
    JSONRenderer,
    StackInfoRenderer,
    TimeStamper,
)
from structlog.stdlib import PositionalArgumentsFormatter, add_log_level
from structlog.tracebacks import ExceptionDictTransformer

from flext_core import c, m, p, t
from flext_core._utilities._logging_config_parts import (
    FlextUtilitiesLoggingStream,
    logging_config_part_01,
)
from flext_core._utilities.metadata import FlextUtilitiesMetadata


class FlextUtilitiesLoggingConfig(logging_config_part_01.FlextUtilitiesLoggingConfig):
    @staticmethod
    def level_based_context_filter(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: t.LoggingEvent,
    ) -> t.LoggingEvent:
        """Filter level-tagged context using the native severity mapping."""
        level_hierarchy = logging.getLevelNamesMapping()
        current_level = level_hierarchy[method_name.upper()]
        filtered_dict: t.LoggingEvent = {}
        for key, value in event_dict.items():
            if not key.startswith("_level_"):
                filtered_dict[key] = value
                continue
            parts = key.split("_", 3)
            if len(parts) < 4:
                filtered_dict[key] = value
                continue
            required_level = level_hierarchy[parts[2].upper()]
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
        """Apply the elected threshold before native enrichment and hooks."""
        _ = logger, method_name
        level = logging.getLevelNamesMapping()[str(event_dict["level"]).upper()]
        if level < logging_config_part_01.FlextUtilitiesLoggingConfig.log_threshold():
            raise structlog.DropEvent
        return event_dict

    @staticmethod
    def _exception_metadata(
        logger: p.OutputLogger,
        method_name: str,
        event: t.LoggingEvent,
    ) -> t.LoggingEvent:
        """Keep causal metadata before the native processor consumes exc_info."""
        info = FlextUtilitiesLoggingConfig.resolve_logging_exception_info(
            event.get("exc_info")
        )
        if info is not None:
            exception = info[1]
            event["_original_exception"] = exception
            event["exception_type"] = type(exception).__name__
            event["exception_message"] = str(exception)
        return event

    @staticmethod
    def _observe_event(
        logger: p.OutputLogger,
        method_name: str,
        event: t.LoggingEvent,
        *,
        options: p.StructlogOptions,
    ) -> t.LoggingEvent:
        """Sanitize the SDK boundary, then notify typed subscribers once."""
        original_exception = event.pop("_original_exception", None)

        # ConsoleRenderer accepts text in `exception`; native structured stacks
        # remain ordinary structured context, shared with JSON and observers.
        if isinstance(event.get("exception"), list):
            event["stack_trace"] = event.pop("exception")
        record = m.LoggingRecord.model_validate({
            "message": str(event["event"]),
            "level": str(event["level"]),
            "context": dict(event),
        })
        sanitized: t.LoggingEvent = dict(
            FlextUtilitiesMetadata.redact_metadata(
                record.context,
                keys=tuple(
                    c.SENSITIVE_ERROR_DATA_KEYS | frozenset(options.redacted_keys)
                ),
                values=options.redacted_values,
                replacement="[REDACTED]",
            )
        )
        record = m.LoggingRecord.model_validate({
            "message": sanitized["event"],
            "level": sanitized["level"],
            "context": sanitized,
            "exception": original_exception,
        })
        for observer in options.observers:
            observer(record)
        return sanitized

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        options: p.StructlogOptions,
        trace: bool,
    ) -> t.SequenceOf[t.LoggingProcessor]:
        """Compose one shared event pipeline and a native terminal renderer."""
        from flext_core._bound_logger import FlextBoundLogger
        from flext_core._utilities.logging import FlextUtilitiesLogging

        processors: t.MutableSequenceOf[t.LoggingProcessor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.drop_below_threshold,
            PositionalArgumentsFormatter(),
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            CallsiteParameterAdder(
                additional_ignores=[
                    FlextBoundLogger.__module__,
                    FlextUtilitiesLogging.__module__,
                ]
            ),
            StackInfoRenderer(),
            cls._exception_metadata,
        ]
        processors.extend(options.processing_stages)
        processors.extend((
            ExceptionRenderer(
                ExceptionDictTransformer(show_locals=False, max_frames=sys.maxsize)
                if trace
                else lambda info: f"{info[0].__name__}: {info[1]}"
            ),
            partial(cls._observe_event, options=options),
        ))
        if options.handlers:
            processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)
        elif console_renderer:
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
        """Keep the selected native factory or the acknowledged stderr writer."""
        owner = logging_config_part_01.FlextUtilitiesLoggingConfig
        if (
            not async_logging or logger_factory is not None
        ) and owner._async_writer is not None:
            owner._async_writer.shutdown()
            owner._async_writer = None
        if logger_factory is not None:
            return logger_factory
        if not async_logging:
            return structlog.PrintLoggerFactory(
                file=FlextUtilitiesLoggingStream.TextStream(sys.stderr)
            )
        if owner._async_writer is None:
            owner._async_writer = cls._AsyncLogWriter(sys.stderr)
        return structlog.PrintLoggerFactory(file=owner._async_writer)


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
