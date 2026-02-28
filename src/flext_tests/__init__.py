"""FLEXT Tests - Shared test utilities and fixtures package.

Provides comprehensive test infrastructure for the FLEXT ecosystem including
common test utilities, matchers, domain objects, factories, builders, Docker
container management, file operations, and integration with core FLEXT components.

Scope: Public API exports for all flext_tests modules including test utilities,
factories, builders, matchers, domain objects, Docker container management,
file operations, and re-exports of core FLEXT components for testing purposes.
All classes and utilities are designed for reuse across FLEXT test suites with
consistent patterns and comprehensive functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_tests import (
        FlextTestsBuilders,
        FlextTestsConstants,
        FlextTestsConstants as c,
        FlextTestsDocker,
        FlextTestsDomains,
        FlextTestsFactories,
        FlextTestsFiles,
        FlextTestsMatchers,
        FlextTestsModels,
        FlextTestsModels as m,
        FlextTestsProtocols,
        FlextTestsProtocols as p,
        FlextTestsServiceBase,
        FlextTestsTypes,
        FlextTestsTypes as t,
        FlextTestsUtilities,
        FlextTestsUtilities as u,
        FlextTestsUtilityBase,
        FlextTestsValidator,
        s,
        tb,
        tf,
        tm,
        tt,
        tv,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextTestsBuilders": ("flext_tests.builders", "FlextTestsBuilders"),
    "FlextTestsConstants": ("flext_tests.constants", "FlextTestsConstants"),
    "FlextTestsDocker": ("flext_tests.docker", "FlextTestsDocker"),
    "FlextTestsDomains": ("flext_tests.domains", "FlextTestsDomains"),
    "FlextTestsFactories": ("flext_tests.factories", "FlextTestsFactories"),
    "FlextTestsFiles": ("flext_tests.files", "FlextTestsFiles"),
    "FlextTestsMatchers": ("flext_tests.matchers", "FlextTestsMatchers"),
    "FlextTestsModels": ("flext_tests.models", "FlextTestsModels"),
    "FlextTestsProtocols": ("flext_tests.protocols", "FlextTestsProtocols"),
    "FlextTestsServiceBase": ("flext_tests.base", "FlextTestsServiceBase"),
    "FlextTestsTypes": ("flext_tests.typings", "FlextTestsTypes"),
    "FlextTestsUtilities": ("flext_tests.utilities", "FlextTestsUtilities"),
    "FlextTestsUtilityBase": ("flext_tests.base", "FlextTestsUtilityBase"),
    "FlextTestsValidator": ("flext_tests.validator", "FlextTestsValidator"),
    "c": ("flext_tests.constants", "FlextTestsConstants"),
    "m": ("flext_tests.models", "FlextTestsModels"),
    "p": ("flext_tests.protocols", "FlextTestsProtocols"),
    "s": ("flext_tests.base", "s"),
    "t": ("flext_tests.typings", "FlextTestsTypes"),
    "tb": ("flext_tests.builders", "tb"),
    "tf": ("flext_tests.files", "tf"),
    "tm": ("flext_tests.matchers", "tm"),
    "tt": ("flext_tests.factories", "tt"),
    "tv": ("flext_tests.validator", "tv"),
    "u": ("flext_tests.utilities", "FlextTestsUtilities"),
}

__all__ = [
    "FlextTestsBuilders",
    "FlextTestsConstants",
    "FlextTestsDocker",
    "FlextTestsDomains",
    "FlextTestsFactories",
    "FlextTestsFiles",
    "FlextTestsMatchers",
    "FlextTestsModels",
    "FlextTestsProtocols",
    "FlextTestsServiceBase",
    "FlextTestsTypes",
    "FlextTestsUtilities",
    "FlextTestsUtilityBase",
    "FlextTestsValidator",
    "c",
    "m",
    "p",
    "s",
    "t",
    "tb",
    "tf",
    "tm",
    "tt",
    "tv",
    "u",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
