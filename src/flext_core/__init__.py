# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

from flext_core.__version__ import (
    __author__ as __author__,
    __author_email__ as __author_email__,
    __description__ as __description__,
    __license__ as __license__,
    __title__ as __title__,
    __url__ as __url__,
    __version__ as __version__,
    __version_info__ as __version_info__,
)

if TYPE_CHECKING:
    from flext_core import services
    from flext_core._config import FlextConfig, config
    from flext_core._settings import FlextSettings, settings
    from flext_core.api import FlextApi, core
    from flext_core.base import FlextBase
    from flext_core.cli import FlextCli
    from flext_core.constants import (
        FlextConstants,
        FlextConstants as c,
        FlextConstantsEnforcement,
    )
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, FlextExceptions as e
    from flext_core.handlers import FlextHandlers, h
    from flext_core.lazy import FlextLazy, FlextLazyAttribute, lazy_attribute
    from flext_core.loggings import FlextUtilitiesLogging
    from flext_core.mixins import FlextMixins, FlextMixins as x
    from flext_core.models import FlextModels, FlextModels as m
    from flext_core.protocols import FlextProtocols, FlextProtocols as p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, FlextResult as r
    from flext_core.runtime import FlextRuntime
    from flext_core.service import FlextService, FlextService as s
    from flext_core.typings import FlextTypes, FlextTypes as t
    from flext_core.utilities import (
        FlextUtilities,
        FlextUtilities as u,
        FlextUtilitiesRuntimeViolationRegistry,
    )


__all__: tuple[str, ...] = (
    "FlextApi",
    "FlextBase",
    "FlextCli",
    "FlextConfig",
    "FlextConstants",
    "FlextConstantsEnforcement",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextLazyAttribute",
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
    "FlextUtilitiesRuntimeViolationRegistry",
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
    "core",
    "d",
    "e",
    "h",
    "lazy_attribute",
    "m",
    "p",
    "r",
    "s",
    "services",
    "settings",
    "t",
    "u",
    "x",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._config": ("FlextConfig", "config"),
            "._settings": ("FlextSettings", "settings"),
            ".api": ("FlextApi", "core"),
            ".base": ("FlextBase",),
            ".cli": ("FlextCli",),
            ".constants": ("FlextConstants", "FlextConstantsEnforcement", "c"),
            ".container": ("FlextContainer",),
            ".context": ("FlextContext",),
            ".decorators": ("FlextDecorators", "d"),
            ".dispatcher": ("FlextDispatcher",),
            ".exceptions": ("FlextExceptions", "e"),
            ".handlers": ("FlextHandlers", "h"),
            ".lazy": ("FlextLazy", "FlextLazyAttribute", "lazy_attribute"),
            ".loggings": ("FlextUtilitiesLogging",),
            ".mixins": ("FlextMixins", "x"),
            ".models": ("FlextModels", "m"),
            ".protocols": ("FlextProtocols", "p"),
            ".registry": ("FlextRegistry",),
            ".result": ("FlextResult", "r"),
            ".runtime": ("FlextRuntime",),
            ".service": ("FlextService", "s"),
            ".services": ("services",),
            ".typings": ("FlextTypes", "t"),
            ".utilities": (
                "FlextUtilities",
                "FlextUtilitiesRuntimeViolationRegistry",
                "u",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    ),
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
