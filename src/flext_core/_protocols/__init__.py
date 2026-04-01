# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Internal module for FlextProtocols nested classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._protocols.base import *
    from flext_core._protocols.config import *
    from flext_core._protocols.container import *
    from flext_core._protocols.context import *
    from flext_core._protocols.handler import *
    from flext_core._protocols.logging import *
    from flext_core._protocols.registry import *
    from flext_core._protocols.result import *
    from flext_core._protocols.service import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "FlextProtocolsBase": "flext_core._protocols.base",
    "FlextProtocolsConfig": "flext_core._protocols.config",
    "FlextProtocolsContainer": "flext_core._protocols.container",
    "FlextProtocolsContext": "flext_core._protocols.context",
    "FlextProtocolsHandler": "flext_core._protocols.handler",
    "FlextProtocolsLogging": "flext_core._protocols.logging",
    "FlextProtocolsRegistry": "flext_core._protocols.registry",
    "FlextProtocolsResult": "flext_core._protocols.result",
    "FlextProtocolsService": "flext_core._protocols.service",
    "base": "flext_core._protocols.base",
    "config": "flext_core._protocols.config",
    "container": "flext_core._protocols.container",
    "context": "flext_core._protocols.context",
    "handler": "flext_core._protocols.handler",
    "logging": "flext_core._protocols.logging",
    "registry": "flext_core._protocols.registry",
    "result": "flext_core._protocols.result",
    "service": "flext_core._protocols.service",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
