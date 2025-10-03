"""Public export surface for FLEXT-Core 1.0.0.

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
from flext_core.service import FlextService
from flext_core.typings import (
    T1,
    T2,
    T3,
    E,
    F,
    FlextTypes,
    K,
    MessageT,
    MessageT_contra,
    P,
    R,
    T,
    T_co,
    T_contra,
    TAccumulate,
    TAggregate,
    TAggregate_co,
    TCommand,
    TCommand_contra,
    TEvent,
    TEvent_contra,
    TInput_contra,
    TItem,
    TQuery,
    TQuery_contra,
    TResult,
    TState,
    TState_co,
    TUtil,
    U,
    V,
    W,
)
from flext_core.utilities import FlextUtilities

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
    "K",
    "MessageT",
    "MessageT_contra",
    "P",
    "R",
    "T",
    "TAccumulate",
    "TAggregate",
    "TAggregate_co",
    "TCommand",
    "TCommand_contra",
    "TEvent",
    "TEvent_contra",
    "TInput_contra",
    "TItem",
    "TQuery",
    "TQuery_contra",
    "TResult",
    "TState",
    "TState_co",
    "TUtil",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
    "__version__",
    "__version_info__",
]
