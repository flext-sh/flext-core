"""Structured logging with context propagation and dependency injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import structlog

from flext_core import FlextConstants as c, FlextProtocols as p, FlextTypes as t
from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext as ulc

if TYPE_CHECKING:
    from structlog.typing import Context


class FlextUtilitiesLogging(ulc):
    """Context-aware utility logger tuned for dispatcher-centric CQRS flows.

    Composed via MRO from:
    - FlextUtilitiesLoggingConfig — structlog configuration, async writer, processors
    - ulc — context binding, value normalization, source paths
    """

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    _level_contexts: ClassVar[t.ScopedContainerRegistry] = {}

    _structlog_instance: p.Logger | None = None

    _structlog_configured: ClassVar[bool] = False

    def __init__(
        self,
        name: str,
        *,
        settings: p.Settings | None = None,
        _bound_logger: p.Logger | None = None,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> None:
        """Initialize FlextUtilitiesLogging with name and optional context."""
        super().__init__()
        if not name:
            msg = "logger name is required"
            raise ValueError(msg)
        resolved_name = name
        self.name = resolved_name
        if _bound_logger is not None:
            self._structlog_instance = _bound_logger
            return
        resolved_context: t.MutableJsonMapping = {}
        if context is not None:
            resolved_context = dict(
                FlextUtilitiesLogging.to_container_context({
                    key: value for key, value in context.items() if value is not None
                }),
            )
        if settings is not None:
            service_name = getattr(settings, c.ContextKey.SERVICE_NAME, None)
            service_version = getattr(settings, c.ContextKey.SERVICE_VERSION, None)
            correlation_id = getattr(settings, c.ContextKey.CORRELATION_ID, None)
            if isinstance(service_name, str) and service_name:
                resolved_context[c.ContextKey.SERVICE_NAME] = service_name
            if isinstance(service_version, str) and service_version:
                resolved_context[c.ContextKey.SERVICE_VERSION] = service_version
            if isinstance(correlation_id, str) and correlation_id:
                resolved_context[c.ContextKey.CORRELATION_ID] = correlation_id
        base_logger = type(self).resolve_bound_logger(resolved_name)
        self._structlog_instance = (
            base_logger.bind(
                **FlextUtilitiesLogging._to_scalar_context(resolved_context)
            )
            if resolved_context
            else base_logger
        )

    def __call__(self) -> Self:
        """Return self to support factory-style DI registration."""
        return self

    @property
    def _context(self) -> Context:
        """Context mapping for BindableLogger protocol compliance."""
        return {}

    @property
    def logger(self) -> p.Logger:
        """Wrapped structlog logger instance."""
        instance = self._structlog_instance
        if instance is None:
            instance = type(self).resolve_bound_logger(
                getattr(self, "name", __name__),
            )
            self._structlog_instance = instance
        return instance

    @classmethod
    def resolve_bound_logger(cls, name: str) -> p.Logger:
        """Fetch the underlying bound structlog logger for internal use."""
        cls.ensure_structlog_configured()
        if not name:
            msg = "logger name is required"
            raise ValueError(msg)
        logger: p.Logger = structlog.get_logger(name)
        return logger

    def bind(self, **context: t.JsonPayload) -> Self:
        """Bind additional context, returning new logger (original unchanged)."""
        bound_logger = self.logger.bind(**self.to_container_context(context))
        return self.__class__(self.name, _bound_logger=bound_logger)

    def new(self, **context: t.JsonPayload) -> Self:
        """Create new logger with context — implements BindableLogger protocol."""
        return self.bind(**context)


__all__: list[str] = ["FlextUtilitiesLogging"]
