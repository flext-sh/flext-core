"""Structlog processor chain assembly and level-based context filtering.

Extracted from FlextUtilitiesLogging as an MRO mixin under the configured
source-size budget. Handles processor construction,
structlog parameter resolution, and async writer lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import logging
import sys
from collections.abc import MutableSequence, Sequence
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
from structlog.typing import Processor

from flext_core import c, m, p, t
from flext_core._utilities import FlextUtilitiesMetadata
from flext_core._utilities._logging_config import FlextUtilitiesLoggingConfigPart01


class FlextUtilitiesLoggingConfigPart02(FlextUtilitiesLoggingConfigPart01):
    @staticmethod
    def level_based_context_filter(
        _logger: p.OutputLogger | None,
        method_name: str,
        event_dict: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Filter level-tagged context using the native severity mapping.

        Returns:
            The event without context above its severity.

        """
        level_hierarchy = logging.getLevelNamesMapping()
        current_level = level_hierarchy[method_name.upper()]
        filtered_dict: p.LoggingEvent = {}
        for key, value in event_dict.items():
            if not key.startswith("_level_"):
                filtered_dict[key] = value
                continue
            required, separator, actual_key = key.removeprefix("_level_").partition("_")
            if not separator:
                filtered_dict[key] = value
                continue
            required_level = level_hierarchy[required.upper()]
            if current_level < required_level:
                continue
            normalized_key = "settings" if actual_key == "config" else actual_key
            filtered_dict[normalized_key] = value
        return filtered_dict

    @staticmethod
    def drop_below_threshold(
        logger: p.OutputLogger | None,
        method_name: str,
        event_dict: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Apply the elected threshold before native enrichment and hooks.

        Returns:
            The admitted event.

        Raises:
            DropEvent: If the event falls below the elected threshold.

        """
        _ = logger, method_name
        level = logging.getLevelNamesMapping()[str(event_dict["level"]).upper()]
        if level < FlextUtilitiesLoggingConfigPart01.log_threshold():
            raise structlog.DropEvent
        return event_dict

    @staticmethod
    def _exception_metadata(
        logger: p.OutputLogger,
        method_name: str,
        event: p.LoggingEvent,
    ) -> p.LoggingEvent:
        """Keep causal metadata before the native processor consumes exc_info.

        Returns:
            The event with its original exception metadata.

        """
        _ = logger, method_name
        info = FlextUtilitiesLoggingConfigPart02.resolve_logging_exception_info(
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
        event: p.LoggingEvent,
        *,
        options: p.StructlogOptions,
    ) -> p.LoggingEvent:
        """Sanitize the SDK boundary, then notify typed subscribers once.

        Returns:
            The sanitized event delivered to subscribers.

        """
        _ = logger, method_name
        original_exception = event.pop("_original_exception", None)

        # ConsoleRenderer accepts text in `exception`; native structured stacks
        # remain ordinary structured context, shared with JSON and observers.
        if isinstance(event.get("exception"), list):
            event["stack_trace"] = event.pop("exception")
        record = m.LoggingRecord.model_validate({
            "message": str(event[c.HandlerType.EVENT]),
            "level": str(event["level"]),
            "context": dict(event),
        })
        sanitized: p.LoggingEvent = dict(
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
            "message": sanitized[c.HandlerType.EVENT],
            "level": sanitized["level"],
            "context": sanitized,
            "exception": original_exception,
        })
        for observer in options.observers:
            observer(record)
        return sanitized

    @staticmethod
    def _format_exception(exc_info: t.LoggingExceptionInfo) -> str:
        """Render native exception information without expanding its traceback.

        Returns:
            The exception type and message.

        """
        return f"{exc_info[0].__name__}: {exc_info[1]}"

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        options: p.StructlogOptions,
        trace: bool,
    ) -> Sequence[Processor]:
        """Compose one shared event pipeline and a native terminal renderer.

        Returns:
            The ordered native processors.

        """
        processors: MutableSequence[Processor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.drop_below_threshold,
            PositionalArgumentsFormatter(),
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            CallsiteParameterAdder(
                additional_ignores=[
                    "flext_core.bound_logger",
                    "flext_core._utilities.logging",
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
                else cls._format_exception
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


__all__: list[str] = ["FlextUtilitiesLoggingConfigPart02"]
