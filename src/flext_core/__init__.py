"""FLEXT Core - Foundation framework for domain-driven applications.

This module provides the public API for flext-core, a comprehensive framework
for building robust applications following domain-driven design and clean
architecture principles.

The framework provides core abstractions for:
- Railway-oriented error handling (FlextResult)
- Dependency injection (FlextContainer)
- CQRS patterns (FlextBus, FlextDispatcher)
- Domain modeling (FlextModels)
- Structured logging (FlextLogger)
- Configuration management (FlextConfig)
- Context management (FlextContext)
- Protocol definitions (FlextProtocols)

For detailed documentation, see the README.md file in this directory.

Example:
    >>> from flext_core import (
    ...     FlextBus,
    ...     FlextConfig,
    ...     FlextConstants,
    ...     FlextContainer,
    ...     FlextContext,
    ...     FlextDecorators,
    ...     FlextDispatcher,
    ...     FlextExceptions,
    ...     FlextHandlers,
    ...     FlextLogger,
    ...     FlextMixins,
    ...     FlextModels,
    ...     FlextProcessors,
    ...     FlextProtocols,
    ...     FlextRegistry,
    ...     FlextResult,
    ...     FlextRuntime,
    ...     FlextService,
    ...     FlextTypes,
    ...     FlextUtilities,
    ... )
    >>>
    >>> # Railway-oriented error handling
    >>> result = FlextResult[str].ok("operation completed")
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
from flext_core.typings import (
    Command,
    E,
    Event,
    F,
    FlextTypes,
    K,
    Message,
    P,
    Query,
    R,
    ResultT,
    T,
    T1_co,
    T2_co,
    T3_co,
    T_co,
    T_contra,
    TAggregate_co,
    TCacheKey_contra,
    TCacheValue_co,
    TCommand_contra,
    TConfigKey_contra,
    TDomainEvent_co,
    TEntity_co,
    TEvent_contra,
    TInput_contra,
    TItem_contra,
    TQuery_contra,
    TResult_co,
    TResult_contra,
    TState_co,
    TUtil_contra,
    TValue_co,
    TValueObject_co,
    U,
    V,
    W,
)
from flext_core.utilities import FlextUtilities

__all__ = [
    "Command",
    "E",
    "Event",
    "F",
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
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
    "FlextRuntime",
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "K",
    "Message",
    "P",
    "Query",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAggregate_co",
    "TCacheKey_contra",
    "TCacheValue_co",
    "TCommand_contra",
    "TConfigKey_contra",
    "TDomainEvent_co",
    "TEntity_co",
    "TEvent_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
    "__version__",
    "__version_info__",
]
