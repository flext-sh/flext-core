"""Internal module for FlextProtocols nested classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.config import FlextProtocolsConfig
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.di import FlextProtocolsDI
from flext_core._protocols.handler import FlextProtocolsHandler
from flext_core._protocols.introspection import (
    ProtocolIntrospection as _ProtocolIntrospection,
)
from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.metaclass import (
    ProtocolModel,
    ProtocolModelMeta,
    ProtocolSettings,
)
from flext_core._protocols.metrics import FlextProtocolsMetrics
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols.service import FlextProtocolsService

__all__ = [
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContext",
    "FlextProtocolsDI",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsMetrics",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "ProtocolModel",
    "ProtocolModelMeta",
    "ProtocolSettings",
    "_ProtocolIntrospection",
]
