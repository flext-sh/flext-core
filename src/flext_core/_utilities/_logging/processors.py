"""Native structlog processor pipeline, filters and observer stages.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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

from flext_core import c, m, p
from flext_core._utilities._logging.exception_info import (
    FlextUtilitiesLoggingExceptionInfo,
)
from flext_core._utilities._logging.handler_stream import (
    FlextUtilitiesLoggingHandlerStream,
)
from flext_core._utilities._logging.levels import FlextUtilitiesLoggingLevels
from flext_core._utilities.metadata import FlextUtilitiesMetadata


class FlextUtilitiesLoggingProcessors(
    FlextUtilitiesLoggingLevels,
    FlextUtilitiesLoggingExceptionInfo,
    FlextUtilitiesLoggingHandlerStream,
):
    """Own the shared event pipeline and its presentation-only stages."""

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

    @classmethod
    def _console_event(
        cls,
        logger: p.OutputLogger | None,
        method_name: str,
        event: p.LoggingEvent,
        *,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> p.LoggingEvent:
        """Select console detail without changing an observer or sibling sink.

        Returns:
            A presentation-only copy of the sanitized event.

        """
        _ = logger, method_name
        selected = runtime_settings or cls.logging_settings()
        hidden = (
            frozenset()
            if selected.debug or selected.trace
            else frozenset(
                item.value for item in structlog.processors.CallsiteParameter
            )
        )
        return {key: value for key, value in event.items() if key not in hidden}

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

    @classmethod
    def _build_structlog_processors(
        cls,
        *,
        console_renderer: bool,
        options: p.StructlogOptions,
        trace: bool,
        runtime_settings: p.LoggingSettings | None = None,
        debug: bool = False,
    ) -> Sequence[Processor]:
        """Compose one shared event pipeline and a native terminal renderer.

        Returns:
            The ordered native processors.

        """
        processors: MutableSequence[Processor] = [
            structlog.contextvars.merge_contextvars,
            add_log_level,
            cls.drop_below_threshold
            if runtime_settings is None
            else partial(
                cls.drop_below_threshold,
                threshold=cls.level_number(runtime_settings.log_level),
            ),
            PositionalArgumentsFormatter(),
            cls.level_based_context_filter,
            TimeStamper(fmt="iso"),
            StackInfoRenderer(),
            cls._exception_metadata,
        ]
        if any((
            debug,
            trace,
            not console_renderer,
            any(
                not isinstance(handler, cls._StreamHandler) or handler.include_callsite
                for handler in options.handlers
            ),
            options.observers,
            options.processing_stages,
        )):
            processors.insert(
                -2,
                CallsiteParameterAdder(
                    additional_ignores=[
                        "flext_core._utilities._logging.bound_logger",
                        "flext_core._utilities.logging",
                    ]
                ),
            )
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
            processors.append(
                partial(cls._console_event, runtime_settings=runtime_settings)
            )
            processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))
        else:
            processors.append(JSONRenderer())
        return processors


__all__: list[str] = ["FlextUtilitiesLoggingProcessors"]
