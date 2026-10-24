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
    from flext_core import services as services
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
    from flext_core._beartype_bootstrap import FlextCoreBeartypeBootstrap
    from flext_core._bound_logger import FlextBoundLogger
    from flext_core._config import FlextConfig, config
    from flext_core._settings import FlextSettings, PydanticBaseSettingsSource, settings
    from flext_core.api import FlextCore, core
    from flext_core.base import FlextBase
    from flext_core.cli import FlextCli
    from flext_core.config_sources import FlextStrictYamlConfigSource
    from flext_core.constants import FlextConstants, FlextConstantsEnforcement, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, FlextMroViolation, e
    from flext_core.flextlazy_attribute import FlextLazyAttribute
    from flext_core.flextlazy_member import FlextLazyMember
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
        FlextUtilitiesBeartypeConf,
        FlextUtilitiesBeartypeEngine,
        FlextUtilitiesEnforcement,
        FlextUtilitiesFamilySurface,
        FlextUtilitiesMapper,
        FlextUtilitiesRuntimeViolationRegistry,
        u,
    )


__all__: tuple[str, ...] = (
    "FlextBase",
    "FlextBoundLogger",
    "FlextCli",
    "FlextConfig",
    "FlextConstants",
    "FlextConstantsEnforcement",
    "FlextContainer",
    "FlextContext",
    "FlextCore",
    "FlextCoreBeartypeBootstrap",
    "FlextDecorators",
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
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextStrictYamlConfigSource",
    "FlextTypes",
    "FlextUtilities",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesEnforcement",
    "FlextUtilitiesFamilySurface",
    "FlextUtilitiesLogging",
    "FlextUtilitiesMapper",
    "FlextUtilitiesRuntimeViolationRegistry",
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
    "services",
    "settings",
    "t",
    "u",
    "x",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextBase": ".base",
        "FlextBoundLogger": "._bound_logger",
        "FlextCli": ".cli",
        "FlextConfig": "._config",
        "FlextConstants": ".constants",
        "FlextConstantsEnforcement": ".constants",
        "FlextContainer": ".container",
        "FlextContext": ".context",
        "FlextCore": ".api",
        "FlextCoreBeartypeBootstrap": "._beartype_bootstrap",
        "FlextDecorators": ".decorators",
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
        "FlextRuntime": ".runtime",
        "FlextService": ".service",
        "FlextSettings": "._settings",
        "FlextStrictYamlConfigSource": ".config_sources",
        "FlextTypes": ".typings",
        "FlextUtilities": ".utilities",
        "FlextUtilitiesBeartypeConf": ".utilities",
        "FlextUtilitiesBeartypeEngine": ".utilities",
        "FlextUtilitiesEnforcement": ".utilities",
        "FlextUtilitiesFamilySurface": ".utilities",
        "FlextUtilitiesLogging": ".loggings",
        "FlextUtilitiesMapper": ".utilities",
        "FlextUtilitiesRuntimeViolationRegistry": ".utilities",
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
        "services": ".services",
        "settings": "._settings",
        "t": ".typings",
        "u": ".utilities",
        "x": ".mixins",
    }),
    public_exports=__all__,
)

if not TYPE_CHECKING:
    from flext_core.__version__ import __version__
