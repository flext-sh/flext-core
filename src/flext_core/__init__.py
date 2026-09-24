# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

from .__version__ import (
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
    from flext_cli import cli, main
    from flext_infra import docs_main, infra
    from flext_tests import (
        active_rules,
        api,
        discover_repository_root,
        install_local_packages,
        load_infra_report,
        split_csv,
        td,
        tf,
        tk,
        tm,
        tv,
    )

    from . import services
    from ._config import FlextConfig, config
    from ._settings import FlextSettings, settings
    from .api import FlextApi, core
    from .base import FlextBase
    from .cli import FlextCli
    from .constants import (
        FlextConstants,
        FlextConstants as c,
        FlextConstantsEnforcement,
    )
    from .container import FlextContainer
    from .context import FlextContext
    from .decorators import FlextDecorators, d
    from .dispatcher import FlextDispatcher
    from .exceptions import FlextExceptions, FlextExceptions as e
    from .handlers import FlextHandlers, h
    from .lazy import FlextLazy, FlextLazyAttribute, lazy_attribute
    from .loggings import FlextUtilitiesLogging
    from .mixins import FlextMixins, FlextMixins as x
    from .models import FlextModels, FlextModels as m
    from .protocols import FlextProtocols, FlextProtocols as p
    from .registry import FlextRegistry
    from .result import FlextResult, FlextResult as r
    from .runtime import FlextRuntime
    from .service import FlextService, FlextService as s
    from .typings import FlextTypes, FlextTypes as t
    from .utilities import (
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
    "active_rules",
    "api",
    "c",
    "cli",
    "config",
    "core",
    "d",
    "discover_repository_root",
    "docs_main",
    "e",
    "h",
    "infra",
    "install_local_packages",
    "lazy_attribute",
    "load_infra_report",
    "m",
    "main",
    "p",
    "r",
    "s",
    "services",
    "settings",
    "split_csv",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "tv",
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
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
