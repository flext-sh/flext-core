"""FLEXT Interfaces module.

Backward compatibility module that re-exports protocols as interfaces.
This module exists to maintain compatibility with existing tests.
"""

# Re-export everything from protocols for backward compatibility
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

__all__ = [
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
