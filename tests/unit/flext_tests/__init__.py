# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Unit tests for flext_tests namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes

    from .test_docker import TestDocker
    from .test_domains import TestFlextTestsDomains
    from .test_factories import TestFactoriesHelpers
    from .test_files import TestFlextTestsFiles
    from .test_matchers import TestFlextTestsMatchers
    from .test_utilities import TestUtilities

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestDocker": ("tests.unit.flext_tests.test_docker", "TestDocker"),
    "TestFactoriesHelpers": ("tests.unit.flext_tests.test_factories", "TestFactoriesHelpers"),
    "TestFlextTestsDomains": ("tests.unit.flext_tests.test_domains", "TestFlextTestsDomains"),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsMatchers": ("tests.unit.flext_tests.test_matchers", "TestFlextTestsMatchers"),
    "TestUtilities": ("tests.unit.flext_tests.test_utilities", "TestUtilities"),
}

__all__ = [
    "TestDocker",
    "TestFactoriesHelpers",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestUtilities",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
