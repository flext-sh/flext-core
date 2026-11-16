"""Native logging handler composition and validation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import MutableSequence
from functools import partial

import structlog
from structlog.typing import Processor

from flext_core import c, p, t
from flext_core._utilities._logging.processors import FlextUtilitiesLoggingProcessors
from flext_core._utilities._logging.stream import FlextUtilitiesLoggingStream


class FlextUtilitiesLoggingHandlers(FlextUtilitiesLoggingProcessors):
    """Own the native handler graph and its fail-loud delivery boundary."""

    @classmethod
    def _handler_factory(cls, options: p.StructlogOptions) -> p.OutputLoggerFactory:
        """Own one native handler graph per composition.

        Returns:
            The factory bound to that graph's native manager.

        Raises:
            TypeError: If a handler or formatter has an incompatible type.
            ValueError: If factories conflict or sink failures are not propagated.

        """
        if options.handlers and options.logger_factory is not None:
            msg = "Select native handlers or a logger factory, not both"
            raise ValueError(msg)
        admitted: t.LoggingNativeHandlers = []
        for handler in options.handlers:
            if not isinstance(handler, logging.Handler):
                msg = (
                    "Selected handlers must implement the native "
                    "logging.Handler boundary"
                )
                raise TypeError(msg)
            if not isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter):
                msg = "Selected handlers require a native ProcessorFormatter"
                raise TypeError(msg)
            if type(handler).handleError is logging.Handler.handleError:
                msg = (
                    "Selected handlers must propagate sink failures; "
                    "use create_logging_handler"
                )
                raise ValueError(msg)
            admitted.append(handler)
        root = logging.RootLogger(logging.NOTSET)
        root.handlers = admitted
        manager = logging.Manager(root)

        def factory(*names: str) -> p.OutputLogger:
            return manager.getLogger(names[0] if names else c.LOGGER_NAME_FLEXT_CORE)

        return factory

    @classmethod
    def create_logging_handler(
        cls,
        *,
        stream: t.TextStream | None = None,
        console_renderer: bool = False,
        log_level: str | None = None,
        runtime_settings: p.LoggingSettings | None = None,
    ) -> p.LoggingHandler:
        """Compose native per-destination presentation with fail-loud delivery.

        Returns:
            The configured native handler.

        """
        handler = FlextUtilitiesLoggingStream.StreamHandler(stream)
        handler.include_callsite = not console_renderer
        processors: MutableSequence[Processor] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        ]
        if console_renderer:
            processors.append(
                partial(cls._console_event, runtime_settings=runtime_settings)
            )
        processors.append(
            structlog.dev.ConsoleRenderer(colors=False)
            if console_renderer
            else structlog.processors.JSONRenderer()
        )
        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=processors,
            )
        )
        if log_level is not None:
            handler.setLevel(cls.level_number(log_level))
        return handler


__all__: list[str] = ["FlextUtilitiesLoggingHandlers"]
