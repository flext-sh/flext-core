"""Layer 0.5 runtime connectors exposing external libraries without circular imports.

This module isolates direct imports of third-party libraries that are required by
lower foundation layers (constants, config, loggings, models, exceptions,
typings, utilities). By centralizing the integration points here we eliminate
circular import risks while still providing direct access to the original APIs,
in line with FLEXT's *direct API* rule.

Dependency Layer: 0.5 (external runtime connectors only)
Dependencies: structlog, dependency_injector, Python stdlib
Used by: config, container, loggings, dispatcher and any lower-layer module
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import ModuleType

import structlog
from dependency_injector import containers, providers


class FlextRuntime:
    """Expose structlog and dependency-injector primitives to foundation layers.

    The class intentionally keeps ZERO imports from other ``flext_core`` modules
    so that any layer (including ``FlextConstants`` and ``FlextTypes``) can
    access runtime primitives without creating circular dependencies. All
    methods return the concrete library modules or perform direct configuration
    using the official APIs—no wrappers, aliases or indirection beyond this
    centralized access point.
    """

    _structlog_configured: bool = False

    @staticmethod
    def structlog() -> ModuleType:
        """Return the imported structlog module."""
        return structlog

    @staticmethod
    def dependency_providers() -> ModuleType:
        """Return the dependency-injector providers module."""
        return providers

    @staticmethod
    def dependency_containers() -> ModuleType:
        """Return the dependency-injector containers module."""
        return containers

    @classmethod
    def configure_structlog(
        cls,
        *,
        log_level: int | None = None,
        console_renderer: bool = True,
        additional_processors: Sequence[object] | None = None,
        wrapper_class_factory: object | None = None,
        logger_factory: object | None = None,
        cache_logger_on_first_use: bool = True,
    ) -> None:
        """Configure structlog once using FLEXT defaults.

        Args:
            log_level: Numeric log level. Defaults to ``logging.INFO``.
            console_renderer: When ``True`` use the console renderer, otherwise
                JSON renderer.
            additional_processors: Optional extra processors appended after the
                standard FLEXT processors.
            wrapper_class_factory: Custom wrapper factory passed to structlog.
                Falls back to :func:`structlog.make_filtering_bound_logger`.
            logger_factory: Custom logger factory. Defaults to
                :class:`structlog.PrintLoggerFactory`.
            cache_logger_on_first_use: Forwarded to structlog configuration.

        """
        if cls._structlog_configured:
            return

        module = structlog
        if module.is_configured():
            cls._structlog_configured = True
            return

        level_to_use = log_level if log_level is not None else logging.INFO

        processors: list[object] = [
            module.contextvars.merge_contextvars,
            module.processors.add_log_level,
            module.processors.TimeStamper(fmt="iso"),
            module.processors.StackInfoRenderer(),
        ]
        if additional_processors:
            processors.extend(additional_processors)

        if console_renderer:
            processors.append(module.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(module.processors.JSONRenderer())

        module.configure(
            processors=processors,
            wrapper_class=(
                wrapper_class_factory
                if wrapper_class_factory is not None
                else module.make_filtering_bound_logger(level_to_use)
            ),
            logger_factory=(
                logger_factory
                if logger_factory is not None
                else module.PrintLoggerFactory()
            ),
            cache_logger_on_first_use=cache_logger_on_first_use,
        )

        cls._structlog_configured = True


__all__ = ["FlextRuntime"]
