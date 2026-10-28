# AUTO-GENERATED FILE — Regenerate with: make gen
"""Examples package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from examples.constants import c
    from examples.ex_05_flext_mixins import ExamplesFlextMixins
    from examples.ex_06_flext_context import ExamplesFlextContext
    from examples.models import ExamplesFlextModels, m
    from examples.protocols import p
    from examples.shared import ExamplesFlextShared
    from examples.typings import t
    from examples.utilities import u
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


__all__: tuple[str, ...] = (
    "ExamplesFlextContext",
    "ExamplesFlextMixins",
    "ExamplesFlextModels",
    "ExamplesFlextShared",
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
        "ExamplesFlextContext": ".ex_06_flext_context",
        "ExamplesFlextMixins": ".ex_05_flext_mixins",
        "ExamplesFlextModels": ".models",
        "ExamplesFlextShared": ".shared",
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
