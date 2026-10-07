"""Structured logging with context propagation and dependency injection.

Family module for ``FlextUtilitiesLogging``: the composed class lives inside
its private family package so canonical facade files import it from a
private-family origin (ENFORCE-046); the root ``loggings.py`` namespace
module re-exports the single canonical binding.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar, Self

from flext_core._bound_logger import FlextBoundLogger
from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext as ulc
from flext_core.constants import c
from flext_core.models import m
from flext_core.protocols import p
from flext_core.typings import t

if TYPE_CHECKING:
    import types


# NOTE (multi-agent): mro-i6nq.12 — consolidated _loggings_parts/part_01..05
# into this single facade module.
class FlextUtilitiesLogging(ulc):
    """Context-aware utility logger tuned for dispatcher-centric CQRS flows.

    Composed via MRO from:
    - FlextUtilitiesLoggingConfig — structlog configuration, async writer, processors
    - ulc — context binding, value normalization, source paths
    """

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    _level_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    class PerformanceTracker:
        """Context manager for performance tracking with automatic logging."""

        def __init__(self, logger: p.Logger, operation_name: str) -> None:
            """Initialize with logger and operation name."""
            super().__init__()
            self.logger = logger
            self._operation_name = operation_name
            self._start_time: float = 0.0

        def __enter__(self) -> Self:
            """Start tracking.

            Returns:
                The resulting ``Self``.

            """
            self._start_time = time.time()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            _exc_tb: types.TracebackType | None,
        ) -> None:
            """Log timing; the context-manager traceback is intentionally unused."""
            elapsed = time.time() - self._start_time
            success = exc_type is None
            status = "success" if success else "failed"
            context: m.ConfigMap = m.ConfigMap(
                root={
                    c.MetadataKey.DURATION_SECONDS: elapsed,
                    c.HandlerType.OPERATION: self._operation_name,
                    c.FIELD_STATUS: status,
                },
            )
            if not success:
                context["exception_type"] = exc_type.__name__ if exc_type else ""
                context["exception_message"] = str(exc_val) if exc_val else ""
            # The FLEXT logger never %-interpolates positional args (it records
            # them as ``arg_<n>`` context), so the event text is composed here.
            emit = self.logger.info if success else self.logger.error
            event = f"{self._operation_name} {status}"
            _ = emit(event, **FlextUtilitiesLogging.to_container_context(context.root))

    @classmethod
    def fetch_logger(cls, name: str) -> p.Logger:
        """Fetch the canonical public logger wrapper.

        Returns:
            The resulting ``p.Logger``.

        """
        return cls.create_module_logger(name)

    @classmethod
    def create_module_logger(
        cls,
        name: str,
        *,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> p.Logger:
        """Create a logger instance for a module.

        Returns:
            The resulting ``p.Logger``.

        """
        cls.ensure_structlog_configured()
        merged_context: t.MutableJsonMapping = {}
        if context is not None:
            merged_context.update(
                cls.to_container_context({
                    key: value for key, value in context.items() if value is not None
                }),
            )
        # Construct the concrete logging owner, never ``cls``: this classmethod
        # is re-exposed through foreign test/utility facades whose construction
        # would return a facade instance instead of a logger.
        logger: p.Logger = FlextBoundLogger(name, context=merged_context)
        return logger
