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

import importlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type hints only - not loaded at runtime
    from flext_core import (
        E,
        FlextConstants,
        FlextContainer,
        FlextContext,
        FlextDecorators,
        FlextDispatcher,
        FlextExceptions,
        FlextHandlers,
        FlextLogger,
        FlextMixins,
        FlextModels,
        FlextProtocols,
        FlextRegistry,
        FlextResult,
        FlextRuntime,
        FlextService,
        FlextSettings,
        FlextTypes,
        FlextUtilities,
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
        c,
        d,
        e,
        h,
        m,
        p,
        r,
        s,
        t,
        u,
        x,
    )
    from flext_core.__version__ import __version__, __version_info__

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Version info
    "__version__": ("flext_core.__version__", "__version__"),
    "__version_info__": ("flext_core.__version__", "__version_info__"),
    # Facade classes and aliases
    "FlextConstants": ("flext_core.constants", "FlextConstants"),
    "c": ("flext_core.constants", "FlextConstants"),
    "FlextContainer": ("flext_core.container", "FlextContainer"),
    "FlextContext": ("flext_core.context", "FlextContext"),
    "FlextDecorators": ("flext_core.decorators", "FlextDecorators"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "FlextDispatcher": ("flext_core.dispatcher", "FlextDispatcher"),
    "FlextExceptions": ("flext_core.exceptions", "FlextExceptions"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "FlextHandlers": ("flext_core.handlers", "FlextHandlers"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "FlextLogger": ("flext_core.loggings", "FlextLogger"),
    "FlextMixins": ("flext_core.mixins", "FlextMixins"),
    "x": ("flext_core.mixins", "FlextMixins"),
    "FlextModels": ("flext_core.models", "FlextModels"),
    "m": ("flext_core.models", "FlextModels"),
    "FlextProtocols": ("flext_core.protocols", "FlextProtocols"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "FlextRegistry": ("flext_core.registry", "FlextRegistry"),
    "FlextResult": ("flext_core.result", "FlextResult"),
    "r": ("flext_core.result", "FlextResult"),
    "FlextRuntime": ("flext_core.runtime", "FlextRuntime"),
    "FlextService": ("flext_core.service", "FlextService"),
    "s": ("flext_core.service", "FlextService"),
    "FlextSettings": ("flext_core.settings", "FlextSettings"),
    "FlextTypes": ("flext_core.typings", "FlextTypes"),
    "t": ("flext_core.typings", "FlextTypes"),
    "FlextUtilities": ("flext_core.utilities", "FlextUtilities"),
    "u": ("flext_core.utilities", "FlextUtilities"),
    # TypeVars and special types
    "E": ("flext_core.typings", "E"),
    "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
    "P": ("flext_core.typings", "P"),
    "R": ("flext_core.typings", "R"),
    "ResultT": ("flext_core.typings", "ResultT"),
    "T": ("flext_core.typings", "T"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "U": ("flext_core.typings", "U"),
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
    """Lazy-load module attributes on first access (PEP 562).

    This defers all imports until actually needed, reducing startup time
    from ~1.2s to <50ms for bare `import flext_core`.

    Handles submodule namespace pollution: when a submodule like
    flext_core.__version__ is imported, Python adds it to the parent
    module's namespace. We need to check _LAZY_IMPORTS first to ensure
    we return the attribute, not the submodule.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in globals() to avoid repeated lookups
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


# Clean up submodule namespace pollution
# When submodules like flext_core.__version__ are imported, Python adds them
# to the parent module's namespace. We remove them to force __getattr__ usage.
def _cleanup_submodule_namespace() -> None:
    """Remove submodules from namespace to force __getattr__ usage."""
    # Get the current module
    current_module = sys.modules[__name__]

    # List of submodule names that might pollute the namespace
    submodule_names = [
        "__version__",
        "constants",
        "container",
        "context",
        "decorators",
        "dispatcher",
        "exceptions",
        "handlers",
        "loggings",
        "mixins",
        "models",
        "protocols",
        "registry",
        "result",
        "runtime",
        "service",
        "settings",
        "typings",
        "utilities",
    ]

    # Remove submodules from the module's namespace
    for submodule_name in submodule_names:
        if hasattr(current_module, submodule_name):
            attr = getattr(current_module, submodule_name)
            # Only remove if it's a module (not our lazy-loaded values)
            if isinstance(attr, type(sys)):
                delattr(current_module, submodule_name)


_cleanup_submodule_namespace()
