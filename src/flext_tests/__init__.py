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

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_tests._validator.bypass import FlextValidatorBypass
    from flext_tests._validator.imports import FlextValidatorImports
    from flext_tests._validator.layer import FlextValidatorLayer
    from flext_tests._validator.models import FlextValidatorModels
    from flext_tests._validator.settings import FlextValidatorSettings
    from flext_tests._validator.tests import FlextValidatorTests
    from flext_tests._validator.types import FlextValidatorTypes
    from flext_tests.base import FlextTestsServiceBase, FlextTestsUtilityBase, s
    from flext_tests.builders import FlextTestsBuilders, tb
    from flext_tests.constants import FlextTestsConstants, FlextTestsConstants as c
    from flext_tests.docker import FlextTestsDocker
    from flext_tests.domains import FlextTestsDomains
    from flext_tests.factories import FlextTestsFactories, tt
    from flext_tests.files import FlextTestsFiles, tf
    from flext_tests.matchers import FlextTestsMatchers, tm
    from flext_tests.models import FlextTestsModels, FlextTestsModels as m
    from flext_tests.protocols import FlextTestsProtocols, FlextTestsProtocols as p
    from flext_tests.typings import FlextTestsTypes, FlextTestsTypes as t
    from flext_tests.utilities import FlextTestsUtilities, FlextTestsUtilities as u
    from flext_tests.validator import FlextTestsValidator, tv

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
    "FlextValidatorBypass": ("flext_tests._validator.bypass", "FlextValidatorBypass"),
    "FlextValidatorImports": (
        "flext_tests._validator.imports",
        "FlextValidatorImports",
    ),
    "FlextValidatorLayer": ("flext_tests._validator.layer", "FlextValidatorLayer"),
    "FlextValidatorModels": ("flext_tests._validator.models", "FlextValidatorModels"),
    "FlextValidatorSettings": (
        "flext_tests._validator.settings",
        "FlextValidatorSettings",
    ),
    "FlextValidatorTests": ("flext_tests._validator.tests", "FlextValidatorTests"),
    "FlextValidatorTypes": ("flext_tests._validator.types", "FlextValidatorTypes"),
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
    "FlextValidatorBypass",
    "FlextValidatorImports",
    "FlextValidatorLayer",
    "FlextValidatorModels",
    "FlextValidatorSettings",
    "FlextValidatorTests",
    "FlextValidatorTypes",
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
