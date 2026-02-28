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

from typing import TYPE_CHECKING, Any

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.__version__ import __version__, __version_info__
    from flext_core.constants import FlextConstants, FlextConstants as c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, FlextDecorators as d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, FlextExceptions as e
    from flext_core.handlers import FlextHandlers, FlextHandlers as h
    from flext_core.loggings import FlextLogger
    from flext_core.mixins import FlextMixins, FlextMixins as x
    from flext_core.models import FlextModels, FlextModels as m
    from flext_core.protocols import FlextProtocols, FlextProtocols as p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, FlextResult as r
    from flext_core.runtime import FlextRuntime
    from flext_core.service import FlextService, FlextService as s
    from flext_core.settings import FlextSettings
    from flext_core.typings import (
        E,
        FlextTypes,
        FlextTypes as t,
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
    from flext_core.utilities import FlextUtilities, FlextUtilities as u

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "E": ("flext_core.typings", "E"),
    "FlextConstants": ("flext_core.constants", "FlextConstants"),
    "FlextContainer": ("flext_core.container", "FlextContainer"),
    "FlextContext": ("flext_core.context", "FlextContext"),
    "FlextDecorators": ("flext_core.decorators", "FlextDecorators"),
    "FlextDispatcher": ("flext_core.dispatcher", "FlextDispatcher"),
    "FlextExceptions": ("flext_core.exceptions", "FlextExceptions"),
    "FlextHandlers": ("flext_core.handlers", "FlextHandlers"),
    "FlextLogger": ("flext_core.loggings", "FlextLogger"),
    "FlextMixins": ("flext_core.mixins", "FlextMixins"),
    "FlextModels": ("flext_core.models", "FlextModels"),
    "FlextProtocols": ("flext_core.protocols", "FlextProtocols"),
    "FlextRegistry": ("flext_core.registry", "FlextRegistry"),
    "FlextResult": ("flext_core.result", "FlextResult"),
    "FlextRuntime": ("flext_core.runtime", "FlextRuntime"),
    "FlextService": ("flext_core.service", "FlextService"),
    "FlextSettings": ("flext_core.settings", "FlextSettings"),
    "FlextTypes": ("flext_core.typings", "FlextTypes"),
    "FlextUtilities": ("flext_core.utilities", "FlextUtilities"),
    "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
    "P": ("flext_core.typings", "P"),
    "R": ("flext_core.typings", "R"),
    "ResultT": ("flext_core.typings", "ResultT"),
    "T": ("flext_core.typings", "T"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "U": ("flext_core.typings", "U"),
    "__version__": ("flext_core.__version__", "__version__"),
    "__version_info__": ("flext_core.__version__", "__version_info__"),
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
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


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
