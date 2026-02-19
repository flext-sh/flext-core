"""Public API for flext-core.

Provides access to core FLEXT components including dispatcher, models,
utilities, and configuration for the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.__version__ import __version__, __version_info__
from flext_core._beartype_conf import BEARTYPE_CONF
from flext_core.constants import FlextConstants, c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators, d
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions, e
from flext_core.handlers import FlextHandlers, h
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins, x
from flext_core.models import FlextModels, m
from flext_core.protocols import FlextProtocols, p
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult, r
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService, s
from flext_core.settings import FlextSettings
from flext_core.typings import (
    E,
    FlextTypes,
    MessageT_contra,
    P,
    R,
    ResultT,
    T,
    T_co,
    T_contra,
    T_Model,
    T_Namespace,
    T_Settings,
    U,
    t,
)
from flext_core.utilities import FlextUtilities, u

__all__ = [
    "BEARTYPE_CONF",
    "E",
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
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "__version__",
    "__version_info__",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
]
