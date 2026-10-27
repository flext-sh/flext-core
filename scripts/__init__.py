# AUTO-GENERATED FILE — Regenerate with: make gen
"""Scripts package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core import (
        build_lazy_import_map,
        cleanup_submodule_namespace,
        config,
        core,
        d,
        e,
        h,
        install_lazy_exports,
        lazy,
        lazy_getattr,
        lazy_member,
        merge_lazy_imports,
        normalize_lazy_imports,
        r,
        resolve_lazy_members,
        s,
        settings,
        x,
    )
    from scripts.constants import ScriptsFlextConstants, c
    from scripts.models import ScriptsFlextModels, m
    from scripts.protocols import ScriptsFlextProtocols, p
    from scripts.typings import ScriptsFlextTypes, t
    from scripts.utilities import ScriptsFlextUtilities, u


__all__: tuple[str, ...] = (
    "ScriptsFlextConstants",
    "ScriptsFlextModels",
    "ScriptsFlextProtocols",
    "ScriptsFlextTypes",
    "ScriptsFlextUtilities",
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
        "ScriptsFlextConstants": ".constants",
        "ScriptsFlextModels": ".models",
        "ScriptsFlextProtocols": ".protocols",
        "ScriptsFlextTypes": ".typings",
        "ScriptsFlextUtilities": ".utilities",
        "build_lazy_import_map": "flext_core",
        "c": ".constants",
        "cleanup_submodule_namespace": "flext_core",
        "config": "flext_core",
        "core": "flext_core",
        "d": "flext_core",
        "e": "flext_core",
        "h": "flext_core",
        "install_lazy_exports": "flext_core",
        "lazy": "flext_core",
        "lazy_getattr": "flext_core",
        "lazy_member": "flext_core",
        "m": ".models",
        "merge_lazy_imports": "flext_core",
        "normalize_lazy_imports": "flext_core",
        "p": ".protocols",
        "r": "flext_core",
        "resolve_lazy_members": "flext_core",
        "s": "flext_core",
        "settings": "flext_core",
        "t": ".typings",
        "u": ".utilities",
        "x": "flext_core",
    }),
    public_exports=__all__,
)
