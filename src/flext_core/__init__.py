# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
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
    from flext_core import services
    from flext_core._config import FlextConfig, StrictYamlConfigSource, config
    from flext_core._settings import FlextSettings, settings
    from flext_core.api import FlextApi, core
    from flext_core.base import FlextBase
    from flext_core.cli import FlextCli
    from flext_core.constants import FlextConstants, FlextConstantsEnforcement, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, e
    from flext_core.handlers import FlextHandlers, h
    from flext_core.lazy import (
        FlextLazy,
        FlextLazyMember,
        lazy_member,
        resolve_lazy_members,
    )
    from flext_core.loggings import FlextUtilitiesLogging
    from flext_core.mixins import FlextMixins, x
    from flext_core.models import FlextModels, m
    from flext_core.protocols import FlextProtocols, p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, r
    from flext_core.runtime import FlextRuntime
    from flext_core.service import FlextService, s
    from flext_core.typings import FlextTypes, t
    from flext_core.utilities import (
        FlextUtilities,
        FlextUtilitiesRuntimeViolationRegistry,
        u,
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
    "FlextLazyMember",
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
    "StrictYamlConfigSource",
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
    "core",
    "d",
    "e",
    "h",
    "install_lazy_exports",
    "lazy_member",
    "m",
    "p",
    "r",
    "resolve_lazy_members",
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
            "._config": ("FlextConfig", "StrictYamlConfigSource", "config"),
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
            ".lazy": (
                "FlextLazy",
                "FlextLazyMember",
                "lazy_member",
                "resolve_lazy_members",
            ),
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
