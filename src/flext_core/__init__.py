"""FLEXT Core - Foundation framework for domain-driven applications.

This module provides the public API for flext-core, a comprehensive framework
for building robust applications following domain-driven design and clean
architecture principles.

The framework provides core abstractions for:
- Railway-oriented error handling (FlextCore.Result)
- Dependency injection (FlextContainer)
- CQRS patterns (FlextBus, FlextDispatcher)
- Domain modeling (FlextModels)
- Structured logging (FlextLogger)
- Configuration management (FlextConfig)
- Context management (FlextContext)
- Protocol definitions (FlextProtocols)

For detailed documentation, see the README.md file in this directory.

Example:
    >>> from flext_core import FlextCore.Result, FlextContainer
    >>>
    >>> # Railway-oriented error handling
    >>> result = FlextCore.Result[str].ok("operation completed")
    >>> if result.is_success:
    ...     data = result.unwrap()
    >>> # Dependency injection
    >>> container = FlextContainer()
    >>> container.register("logger", FlextLogger(__name__))

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core.__version__ import __version__, __version_info__
from flext_core.api import FlextCore
from flext_core.base import FlextBase
from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities

__all__ = [
    "FlextBase",
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessors",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "__version__",
    "__version_info__",
]
