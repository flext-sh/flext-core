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
    from flext_core._config import FlextConfig, config
    from flext_core._settings import FlextSettings, settings
    from flext_core.api import FlextApi, core
    from flext_core.base import FlextBase
    from flext_core.cli import FlextCli
    from flext_core.config_sources import StrictYamlConfigSource
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
    from flext_core.mixins import FlextMixins
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
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextApi": ".api",
        "FlextBase": ".base",
        "FlextCli": ".cli",
        "FlextConfig": "._config",
        "FlextConstants": ".constants",
        "FlextConstantsEnforcement": ".constants",
        "FlextContainer": ".container",
        "FlextContext": ".context",
        "FlextDecorators": ".decorators",
        "FlextDispatcher": ".dispatcher",
        "FlextExceptions": ".exceptions",
        "FlextHandlers": ".handlers",
        "FlextLazy": ".lazy",
        "FlextLazyMember": ".lazy",
        "FlextMixins": ".mixins",
        "FlextModels": ".models",
        "FlextProtocols": ".protocols",
        "FlextRegistry": ".registry",
        "FlextResult": ".result",
        "FlextRuntime": ".runtime",
        "FlextService": ".service",
        "FlextSettings": "._settings",
        "FlextTypes": ".typings",
        "FlextUtilities": ".utilities",
        "FlextUtilitiesLogging": ".loggings",
        "FlextUtilitiesRuntimeViolationRegistry": ".utilities",
        "StrictYamlConfigSource": ".config_sources",
        "c": ".constants",
        "config": "._config",
        "core": ".api",
        "d": ".decorators",
        "e": ".exceptions",
        "h": ".handlers",
        "lazy_member": ".lazy",
        "m": ".models",
        "p": ".protocols",
        "r": ".result",
        "resolve_lazy_members": ".lazy",
        "s": ".service",
        "services": ".services",
        "settings": "._settings",
        "t": ".typings",
        "u": ".utilities",
    }),
    public_exports=__all__,
)
