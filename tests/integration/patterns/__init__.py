# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Pattern implementation unit tests.

Tests for FLEXT Core design patterns:
- Command pattern and CQRS
- Handler patterns
- Validation patterns
- Logging patterns
- Field metadata patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import (
        FlextConstants,
        FlextContainer,
        FlextContext,
        FlextDecorators,
        FlextDispatcher,
        FlextExceptions,
        FlextLogger,
        FlextModels,
        FlextRegistry,
        FlextResult,
        FlextRuntime,
        FlextService,
        FlextSettings,
        h,
        p,
        t,
        u,
        x,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextConstants": ("flext_core", "FlextConstants"),
    "FlextContainer": ("flext_core", "FlextContainer"),
    "FlextContext": ("flext_core", "FlextContext"),
    "FlextDecorators": ("flext_core", "FlextDecorators"),
    "FlextDispatcher": ("flext_core", "FlextDispatcher"),
    "FlextExceptions": ("flext_core", "FlextExceptions"),
    "FlextLogger": ("flext_core", "FlextLogger"),
    "FlextModels": ("flext_core", "FlextModels"),
    "FlextRegistry": ("flext_core", "FlextRegistry"),
    "FlextResult": ("flext_core", "FlextResult"),
    "FlextRuntime": ("flext_core", "FlextRuntime"),
    "FlextService": ("flext_core", "FlextService"),
    "FlextSettings": ("flext_core", "FlextSettings"),
    "h": ("flext_core", "h"),
    "p": ("flext_core", "p"),
    "t": ("flext_core", "t"),
    "u": ("flext_core", "u"),
    "x": ("flext_core", "x"),
}

__all__ = [
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextLogger",
    "FlextModels",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "h",
    "p",
    "t",
    "u",
    "x",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
