# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""flext-core comprehensive test suite.

This package contains all tests for flext-core components.
Uses flext_tests directly for all generic test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import e
    from tests.base import TestsFlextServiceBase
    from tests.constants import TestsFlextConstants, TestsFlextConstants as c
    from tests.models import TestsFlextModels, TestsFlextModels as m
    from tests.protocols import TestsFlextProtocols, TestsFlextProtocols as p
    from tests.typings import TestsFlextTypes, TestsFlextTypes as t
    from tests.utilities import TestsFlextUtilities, TestsFlextUtilities as u

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestsFlextConstants": ("tests.constants", "TestsFlextConstants"),
    "TestsFlextModels": ("tests.models", "TestsFlextModels"),
    "TestsFlextProtocols": ("tests.protocols", "TestsFlextProtocols"),
    "TestsFlextServiceBase": ("tests.base", "TestsFlextServiceBase"),
    "TestsFlextTypes": ("tests.typings", "TestsFlextTypes"),
    "TestsFlextUtilities": ("tests.utilities", "TestsFlextUtilities"),
    "c": ("tests.constants", "TestsFlextConstants"),
    "e": ("flext_core", "e"),
    "m": ("tests.models", "TestsFlextModels"),
    "p": ("tests.protocols", "TestsFlextProtocols"),
    "t": ("tests.typings", "TestsFlextTypes"),
    "u": ("tests.utilities", "TestsFlextUtilities"),
}

__all__ = [
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "c",
    "e",
    "m",
    "p",
    "t",
    "u",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
