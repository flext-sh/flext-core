"""FLEXT Interfaces module.

Backward compatibility module that re-exports protocols as interfaces.
This module exists to maintain compatibility with existing tests.
"""

from __future__ import annotations

from flext_core.protocols import (
    FlextAuthProtocol,
    FlextConfigurable,
    FlextConnectionProtocol,
    FlextEventPublisher,
    FlextEventSubscriber,
    FlextHandler,
    FlextMiddleware,
    FlextObservabilityProtocol,
    FlextPlugin,
    FlextPluginContext,
    FlextRepository,
    FlextService,
    FlextUnitOfWork,
    FlextValidationRule,
    FlextValidator,
)

__all__: list[str] = [
    "FlextAuthProtocol",
    "FlextConfigurable",
    "FlextConnectionProtocol",
    "FlextEventPublisher",
    "FlextEventSubscriber",
    "FlextHandler",
    "FlextMiddleware",
    "FlextObservabilityProtocol",
    "FlextPlugin",
    "FlextPluginContext",
    "FlextRepository",
    "FlextService",
    "FlextUnitOfWork",
    "FlextValidationRule",
    "FlextValidator",
]
