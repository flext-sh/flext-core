# AUTO-GENERATED FILE — Regenerate with: make gen
"""Root Typing Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._root_typing_parts._exports import (
    FLEXT_CORE__ROOT_TYPING_PARTS_LAZY_IMPORTS,
)
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._root_typing_parts.facades import (
        FlextConstants as FlextConstants,
        FlextContainer as FlextContainer,
        FlextContext as FlextContext,
        FlextDecorators as FlextDecorators,
        FlextDispatcher as FlextDispatcher,
        FlextExceptions as FlextExceptions,
        FlextHandlers as FlextHandlers,
        FlextLazy as FlextLazy,
        FlextLogger as FlextLogger,
        FlextMixins as FlextMixins,
        FlextModels as FlextModels,
        FlextProtocols as FlextProtocols,
        FlextRegistry as FlextRegistry,
        FlextResult as FlextResult,
        FlextRuntime as FlextRuntime,
        FlextService as FlextService,
        FlextSettings as FlextSettings,
        FlextTypes as FlextTypes,
        FlextUtilities as FlextUtilities,
        build_lazy_import_map as build_lazy_import_map,
        c as c,
        d as d,
        e as e,
        h as h,
        lazy as lazy,
        m as m,
        normalize_lazy_imports as normalize_lazy_imports,
        p as p,
        r as r,
        s as s,
        t as t,
        u as u,
        x as x,
    )

_LAZY_IMPORTS = FLEXT_CORE__ROOT_TYPING_PARTS_LAZY_IMPORTS


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
