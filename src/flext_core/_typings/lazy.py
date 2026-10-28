"""Lazy export typing contracts for ``flext_core.lazy`` internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from types import ModuleType

from flext_core._typings import FlextTypingBase


class FlextTypesLazy:
    """Typing namespace for package-level lazy export internals."""

    type ModuleGlobalValue = (
        FlextTypingBase.JsonValue
        | FlextTypingBase.LazyImportMap
        | FlextTypingBase.Sequence[str]
        | ModuleType
        | type
        | Callable[
            ...,
            FlextTypingBase.JsonValue
            | FlextTypingBase.Sequence[str]
            | ModuleType
            | type
            | None,
        ]
        | None
    )
    type LazyExportHooks = tuple[
        Callable[[str], ModuleGlobalValue], Callable[[], list[str]]
    ]
    type ModuleGlobals = MutableMapping[
        str,
        ModuleGlobalValue | Callable[..., ModuleGlobalValue],
    ]


__all__: list[str] = ["FlextTypesLazy"]
