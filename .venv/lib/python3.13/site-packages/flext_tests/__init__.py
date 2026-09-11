# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests package."""

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
    from flext_infra import c as infra_c, d, e, h, r, x

    from ._config import FlextTestsConfig, config
    from ._settings import FlextTestsSettings, settings
    from .base import FlextTestsCase, FlextTestsServiceBase, FlextTestsServiceBase as s
    from .constants import FlextTestsConstants, FlextTestsConstants as c
    from .docker import FlextTestsDocker, tk
    from .domains import FlextTestsDomains, td
    from .enforcement import (
        EnforcementCollector,
        EnforcementItem,
        EnforcementViolationError,
        active_rules,
        discover_workspace_root,
        load_infra_report,
        split_csv,
    )
    from .enforcement_plugin import SLOW_TIMEOUT_INI_OPTION
    from .files import FlextTestsFiles, tf
    from .kube import FlextTestsKube, tkube
    from .models import FlextTestsModels, FlextTestsModels as m
    from .protocols import FlextTestsProtocols, FlextTestsProtocols as p
    from .tmatchers import FlextTestsMatchersUtilities, tm
    from .typings import FlextTestsTypes, FlextTestsTypes as t
    from .utilities import (
        FlextTestsFixturesDSLMixin,
        FlextTestsUtilities,
        FlextTestsUtilities as u,
    )
    from .validator import FlextTestsValidator, tv
__all__: tuple[str, ...] = (
    "SLOW_TIMEOUT_INI_OPTION",
    "EnforcementCollector",
    "EnforcementItem",
    "EnforcementViolationError",
    "FlextTestsCase",
    "FlextTestsConfig",
    "FlextTestsConstants",
    "FlextTestsDocker",
    "FlextTestsDomains",
    "FlextTestsFiles",
    "FlextTestsFixturesDSLMixin",
    "FlextTestsKube",
    "FlextTestsMatchersUtilities",
    "FlextTestsModels",
    "FlextTestsProtocols",
    "FlextTestsServiceBase",
    "FlextTestsSettings",
    "FlextTestsTypes",
    "FlextTestsUtilities",
    "FlextTestsValidator",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "active_rules",
    "c",
    "config",
    "d",
    "discover_workspace_root",
    "e",
    "h",
    "infra_c",
    "load_infra_report",
    "m",
    "p",
    "r",
    "s",
    "settings",
    "split_csv",
    "t",
    "td",
    "tf",
    "tk",
    "tkube",
    "tm",
    "tv",
    "u",
    "x",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._config": ("FlextTestsConfig", "config"),
            "._settings": ("FlextTestsSettings", "settings"),
            ".base": ("FlextTestsCase", "FlextTestsServiceBase", "s"),
            ".constants": ("FlextTestsConstants", "c"),
            ".docker": ("FlextTestsDocker", "tk"),
            ".domains": ("FlextTestsDomains", "td"),
            ".enforcement": (
                "EnforcementCollector",
                "EnforcementItem",
                "EnforcementViolationError",
                "active_rules",
                "discover_workspace_root",
                "load_infra_report",
                "split_csv",
            ),
            ".enforcement_plugin": ("SLOW_TIMEOUT_INI_OPTION",),
            ".files": ("FlextTestsFiles", "tf"),
            ".kube": ("FlextTestsKube", "tkube"),
            ".models": ("FlextTestsModels", "m"),
            ".protocols": ("FlextTestsProtocols", "p"),
            ".tmatchers": ("FlextTestsMatchersUtilities", "tm"),
            ".typings": ("FlextTestsTypes", "t"),
            ".utilities": ("FlextTestsFixturesDSLMixin", "FlextTestsUtilities", "u"),
            ".validator": ("FlextTestsValidator", "tv"),
            "flext_infra": ("d", "e", "h", "r", "x"),
        }),
        alias_groups=MappingProxyType({"flext_infra": (("infra_c", "c"),)}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
