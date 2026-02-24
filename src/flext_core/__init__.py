"""Public API for flext-core.

Runtime aliases: simple assignments only (c = FlextConstants, m = FlextModels, etc.).
Never use FlextRuntime.Aliases or any alias registry for c, m, r, t, u, p, d, e, h, s, x.

Access via project runtime alias only; no subdivision. Subprojects: nested classes
for organization, then class-level aliases at facade root so call sites use m.Foo,
m.Bar only (never m.ProjectName.Foo). MRO protocol only; direct methods.

Use at call sites: from flext_core import c, m, r, t, u, p, d, e, h, s, x
Classes (FlextContainer, FlextModels, etc.) are for inheritance and type annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.__version__ import __version__, __version_info__
from flext_core._beartype_conf import BEARTYPE_CONF
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
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
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
)
from flext_core.utilities import FlextUtilities

# Simple runtime aliases only (no FlextRuntime.Aliases.*)
c = FlextConstants
d = FlextDecorators
e = FlextExceptions
h = FlextHandlers
m = FlextModels
p = FlextProtocols
r = FlextResult
s = FlextService
t = FlextTypes
u = FlextUtilities
x = FlextMixins

# Public API (runtime aliases first; MRO protocol only)
__all__ = [
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
]
