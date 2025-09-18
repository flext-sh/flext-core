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
from flext_core.dispatcher_registry import FlextDispatcherRegistry
from flext_core.domain_services import FlextDomainService
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processing import FlextProcessing
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import (
    E,
    F,
    FlextTypes,
    P,
    R,
    T,
    T_co,
    U,
    V,
)
from flext_core.utilities import FlextUtilities
from flext_core.version import FlextVersionManager, __version__

__all__ = [
    "E",
    "F",
    "FlextBus",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCqrs",
    "FlextDispatcher",
    "FlextDispatcherRegistry",
    "FlextDomainService",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProcessing",
    "FlextProtocols",
    "FlextResult",
    "FlextTypes",
    "FlextUtilities",
    "FlextVersionManager",
    "P",
    "R",
    "T",
    "T_co",
    "U",
    "V",
    "__version__",
]
