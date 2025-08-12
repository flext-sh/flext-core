"""FLEXT Interfaces module.

Backward compatibility module that re-exports protocols as interfaces.
This module exists to maintain compatibility with existing tests and
legacy code that expects interface imports.

All interfaces are actually protocols from the flext_core.protocols
module. New code should import directly from protocols module for
better clarity and future compatibility.

Example:
    Legacy interface imports (still supported):

    >>> from flext_core.interfaces import FlextService, FlextRepository

    >>> class MyService(FlextService):
    ...     pass

    Preferred modern imports:

    >>> from flext_core.protocols import FlextService, FlextRepository

    >>> class MyService(FlextService):
    ...     pass

Note:
    This module will be maintained for backward compatibility
    but new development should use the protocols module directly.

Author: FLEXT Development Team
Version: 2.0.0
License: MIT

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
