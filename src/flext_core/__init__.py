# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

from .__version__ import __author__ as __author__
from .__version__ import __author_email__ as __author_email__
from .__version__ import __description__ as __description__
from .__version__ import __license__ as __license__
from .__version__ import __title__ as __title__
from .__version__ import __url__ as __url__
from .__version__ import __version__ as __version__
from .__version__ import __version_info__ as __version_info__

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

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType(
        build_lazy_import_map(
            MappingProxyType({
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
            }),
            alias_groups=MappingProxyType({}),
            sort_keys=False,
        )
    ),
    public_exports=__all__,
)
