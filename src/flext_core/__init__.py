# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.__version__ import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
    __version_info__,
)
from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from ._config import FlextConfig, config
    from ._settings import FlextSettings, settings
    from .constants import FlextConstants, FlextConstants as c
    from .container import FlextContainer
    from .context import FlextContext
    from .decorators import FlextDecorators, d
    from .dispatcher import FlextDispatcher
    from .exceptions import FlextExceptions, e
    from .handlers import FlextHandlers, h
    from .lazy import FlextLazy
    from .loggings import FlextUtilitiesLogging
    from .mixins import FlextMixins, x
    from .models import FlextModels, FlextModels as m
    from .protocols import FlextProtocols, FlextProtocols as p
    from .registry import FlextRegistry
    from .result import FlextResult, r
    from .runtime import FlextRuntime
    from .service import FlextService, s
    from .typings import FlextTypes, FlextTypes as t
    from .utilities import FlextUtilities, FlextUtilities as u

    _ = (
        c,
        FlextConstants,
        t,
        FlextTypes,
        p,
        FlextProtocols,
        m,
        FlextModels,
        u,
        FlextUtilities,
        d,
        FlextDecorators,
        e,
        FlextExceptions,
        h,
        FlextHandlers,
        r,
        FlextResult,
        s,
        FlextService,
        x,
        FlextMixins,
        FlextConfig,
        config,
        FlextSettings,
        settings,
        FlextContainer,
        FlextContext,
        FlextDispatcher,
        FlextLazy,
        FlextUtilitiesLogging,
        FlextRegistry,
        FlextRuntime,
    )


_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    "._config": ("FlextConfig", "config"),
    "._settings": ("FlextSettings", "settings"),
    ".constants": ("FlextConstants", "c"),
    ".container": ("FlextContainer",),
    ".context": ("FlextContext",),
    ".decorators": ("FlextDecorators", "d"),
    ".dispatcher": ("FlextDispatcher",),
    ".exceptions": ("FlextExceptions", "e"),
    ".handlers": ("FlextHandlers", "h"),
    ".lazy": ("FlextLazy",),
    ".loggings": ("FlextUtilitiesLogging",),
    ".mixins": ("FlextMixins", "x"),
    ".models": ("FlextModels", "m"),
    ".protocols": ("FlextProtocols", "p"),
    ".registry": ("FlextRegistry",),
    ".result": ("FlextResult", "r"),
    ".runtime": ("FlextRuntime",),
    ".service": ("FlextService", "s"),
    ".typings": ("FlextTypes", "t"),
    ".utilities": ("FlextUtilities", "u"),
}


_LAZY_ALIAS_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {}


_LAZY_IMPORTS = build_lazy_import_map(
    _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
)

_DIRECT_IMPORTS: tuple[str, ...] = (
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
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
    "FlextUtilitiesLogging",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "build_lazy_import_map",
    "c",
    "config",
    "d",
    "e",
    "h",
    "install_lazy_exports",
    "m",
    "p",
    "r",
    "s",
    "settings",
    "t",
    "u",
    "x",
)

__all__: tuple[str, ...] = (
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
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
    "FlextUtilitiesLogging",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "c",
    "config",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "settings",
    "t",
    "u",
    "x",
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
