# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Internal module for FlextProtocols nested classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols import (
        base as base,
        config as config,
        container as container,
        context as context,
        handler as handler,
        logging as logging,
        registry as registry,
        result as result,
        service as service,
    )
    from flext_core._protocols.base import FlextProtocolsBase as FlextProtocolsBase
    from flext_core._protocols.config import (
        FlextProtocolsConfig as FlextProtocolsConfig,
    )
    from flext_core._protocols.container import (
        FlextProtocolsContainer as FlextProtocolsContainer,
    )
    from flext_core._protocols.context import (
        FlextProtocolsContext as FlextProtocolsContext,
    )
    from flext_core._protocols.handler import (
        FlextProtocolsHandler as FlextProtocolsHandler,
    )
    from flext_core._protocols.logging import (
        FlextProtocolsLogging as FlextProtocolsLogging,
    )
    from flext_core._protocols.registry import (
        FlextProtocolsRegistry as FlextProtocolsRegistry,
    )
    from flext_core._protocols.result import (
        FlextProtocolsResult as FlextProtocolsResult,
    )
    from flext_core._protocols.service import (
        FlextProtocolsService as FlextProtocolsService,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FlextProtocolsBase": ["flext_core._protocols.base", "FlextProtocolsBase"],
    "FlextProtocolsConfig": ["flext_core._protocols.config", "FlextProtocolsConfig"],
    "FlextProtocolsContainer": [
        "flext_core._protocols.container",
        "FlextProtocolsContainer",
    ],
    "FlextProtocolsContext": ["flext_core._protocols.context", "FlextProtocolsContext"],
    "FlextProtocolsHandler": ["flext_core._protocols.handler", "FlextProtocolsHandler"],
    "FlextProtocolsLogging": ["flext_core._protocols.logging", "FlextProtocolsLogging"],
    "FlextProtocolsRegistry": [
        "flext_core._protocols.registry",
        "FlextProtocolsRegistry",
    ],
    "FlextProtocolsResult": ["flext_core._protocols.result", "FlextProtocolsResult"],
    "FlextProtocolsService": ["flext_core._protocols.service", "FlextProtocolsService"],
    "base": ["flext_core._protocols.base", ""],
    "config": ["flext_core._protocols.config", ""],
    "container": ["flext_core._protocols.container", ""],
    "context": ["flext_core._protocols.context", ""],
    "handler": ["flext_core._protocols.handler", ""],
    "logging": ["flext_core._protocols.logging", ""],
    "registry": ["flext_core._protocols.registry", ""],
    "result": ["flext_core._protocols.result", ""],
    "service": ["flext_core._protocols.service", ""],
}

_EXPORTS: Sequence[str] = [
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "base",
    "config",
    "container",
    "context",
    "handler",
    "logging",
    "registry",
    "result",
    "service",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
