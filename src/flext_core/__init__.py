"""Public export surface for FLEXT-Core 1.0.0.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.cqrs import FlextCqrs
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels, WorkspaceStatus
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.service import FlextService
from flext_core.typings import (
    T1,
    T2,
    T3,
    # Core TypeVars
    E,
    F,
    # Main type system
    FlextTypes,
    K,
    MessageT,
    MessageT_contra,
    P,
    R,
    T,
    # Covariant TypeVars
    T_co,
    # Contravariant TypeVars
    T_contra,
    TAccumulate,
    TAggregate,
    TAggregate_co,
    # ALGAR TypeVars - removed non-existent imports
    TCacheKey,
    TCacheKey_contra,
    TCacheValue,
    TCacheValue_co,
    # CLI TypeVars - removed non-existent exports
    # Domain TypeVars
    TCommand,
    TCommand_contra,
    TConcurrent,
    TConfigKey,
    TConfigKey_contra,
    TConfigValue,
    TConfigValue_co,
    TDomainEvent,
    TDomainEvent_co,
    TEntity,
    TEntity_co,
    TEvent,
    TEvent_contra,
    TInput_contra,
    TItem,
    TKey,
    TKey_contra,
    TMessage,
    TParallel,
    TQuery,
    TQuery_contra,
    TResource,
    TResult,
    TResult_co,
    # Service TypeVars
    TService,
    TState,
    TState_co,
    TTimeout,
    TUtil,
    TValue,
    TValue_co,
    TValueObject_co,
    U,
    UParallel,
    UResource,
    V,
    W,
)
from flext_core.utilities import FlextUtilities
from flext_core.version import FlextVersionManager, __version__

__all__ = [
    "T1",
    "T2",
    "T3",
    "E",
    "F",
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCqrs",
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
    "FlextService",
    "FlextTypes",
    "FlextUtilities",
    "FlextVersionManager",
    "K",
    "MessageT",
    "MessageT_contra",
    "P",
    "R",
    "T",
    "TAccumulate",
    "TAggregate",
    "TAggregate_co",
    "TCacheKey",
    "TCacheKey_contra",
    "TCacheValue",
    "TCacheValue_co",
    "TCommand",
    "TCommand_contra",
    "TConcurrent",
    "TConfigKey",
    "TConfigKey_contra",
    "TConfigValue",
    "TConfigValue_co",
    "TDomainEvent",
    "TDomainEvent_co",
    "TEntity",
    "TEntity_co",
    "TEvent",
    "TEvent_contra",
    "TInput_contra",
    "TItem",
    "TKey",
    "TKey_contra",
    "TMessage",
    "TParallel",
    "TQuery",
    "TQuery_contra",
    "TResource",
    "TResult",
    "TResult_co",
    "TService",
    "TState",
    "TState_co",
    "TTimeout",
    "TUtil",
    "TValue",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "U",
    "UParallel",
    "UResource",
    "V",
    "W",
    "WorkspaceStatus",
    "__version__",
]
