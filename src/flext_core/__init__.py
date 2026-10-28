# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core.__version__ import (
        FlextVersion,
        __author__,
        __author_email__,
        __description__,
        __license__,
        __title__,
        __url__,
        __version__,
        __version_info__,
    )
    from flext_core._config import FlextConfig, config
    from flext_core._settings import FlextSettings, PydanticBaseSettingsSource, settings
    from flext_core._utilities.logging import FlextUtilitiesLogging
    from flext_core.api import FlextCore, core
    from flext_core.bound_logger import FlextBoundLogger
    from flext_core.cli import FlextCli
    from flext_core.config_sources import FlextStrictYamlConfigSource
    from flext_core.constants import FlextConstants, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecoratorsRuntime, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, FlextMroViolation, e
    from flext_core.flextlazy_attribute import FlextLazyAttribute
    from flext_core.flextlazy_member import FlextLazyMember
    from flext_core.flextlazy_native_export import FlextTypingNativeExport
    from flext_core.flextlazy_part_01 import FlextLazyPart01
    from flext_core.flextlazy_part_02 import FlextLazyPart02
    from flext_core.handlers import FlextHandlers, h
    from flext_core.lazy import (
        FlextLazy,
        cleanup_submodule_namespace,
        lazy,
        lazy_getattr,
        lazy_member,
        merge_lazy_imports,
        normalize_lazy_imports,
        resolve_lazy_members,
    )
    from flext_core.mixins import FlextMixins, x
    from flext_core.models import FlextModels, m
    from flext_core.protocols import FlextProtocols, p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, r
    from flext_core.service import FlextService, s
    from flext_core.typings import FlextTypes, t
    from flext_core.utilities import FlextUtilities, u

__all__: tuple[str, ...] = (
    "FlextBoundLogger",
    "FlextCli",
    "FlextConfig",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextDecoratorsRuntime",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextLazyAttribute",
    "FlextLazyMember",
    "FlextLazyPart01",
    "FlextLazyPart02",
    "FlextMixins",
    "FlextModels",
    "FlextMroViolation",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextService",
    "FlextSettings",
    "FlextStrictYamlConfigSource",
    "FlextTypes",
    "FlextTypingNativeExport",
    "FlextUtilities",
    "FlextUtilitiesLogging",
    "FlextVersion",
    "PydanticBaseSettingsSource",
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
    "cleanup_submodule_namespace",
    "config",
    "core",
    "d",
    "e",
    "h",
    "install_lazy_exports",
    "lazy",
    "lazy_getattr",
    "lazy_member",
    "m",
    "merge_lazy_imports",
    "normalize_lazy_imports",
    "p",
    "r",
    "resolve_lazy_members",
    "s",
    "settings",
    "t",
    "u",
    "x",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBoundLogger": ".bound_logger",
        "FlextCli": ".cli",
        "FlextConfig": "._config",
        "FlextConstants": ".constants",
        "FlextContainer": ".container",
        "FlextContext": ".context",
        "FlextCore": ".api",
        "FlextDecoratorsRuntime": ".decorators",
        "FlextDispatcher": ".dispatcher",
        "FlextExceptions": ".exceptions",
        "FlextHandlers": ".handlers",
        "FlextLazy": ".lazy",
        "FlextLazyAttribute": ".flextlazy_attribute",
        "FlextLazyMember": ".flextlazy_member",
        "FlextLazyPart01": ".flextlazy_part_01",
        "FlextLazyPart02": ".flextlazy_part_02",
        "FlextMixins": ".mixins",
        "FlextModels": ".models",
        "FlextMroViolation": ".exceptions",
        "FlextProtocols": ".protocols",
        "FlextRegistry": ".registry",
        "FlextResult": ".result",
        "FlextService": ".service",
        "FlextSettings": "._settings",
        "FlextStrictYamlConfigSource": ".config_sources",
        "FlextTypes": ".typings",
        "FlextTypingNativeExport": ".flextlazy_native_export",
        "FlextUtilities": ".utilities",
        "FlextVersion": ".__version__",
        "PydanticBaseSettingsSource": "._settings",
        "__author__": ".__version__",
        "__author_email__": ".__version__",
        "__description__": ".__version__",
        "__license__": ".__version__",
        "__title__": ".__version__",
        "__url__": ".__version__",
        "__version_info__": ".__version__",
        "c": ".constants",
        "cleanup_submodule_namespace": ".lazy",
        "config": "._config",
        "core": ".api",
        "d": ".decorators",
        "e": ".exceptions",
        "h": ".handlers",
        "lazy": ".lazy",
        "lazy_getattr": ".lazy",
        "lazy_member": ".lazy",
        "m": ".models",
        "merge_lazy_imports": ".lazy",
        "normalize_lazy_imports": ".lazy",
        "p": ".protocols",
        "r": ".result",
        "resolve_lazy_members": ".lazy",
        "s": ".service",
        "settings": "._settings",
        "t": ".typings",
        "u": ".utilities",
        "x": ".mixins",
        "FlextUtilitiesLogging": "._utilities.logging",
    }),
    public_exports=__all__,
)
