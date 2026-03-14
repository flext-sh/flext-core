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
    from tests.unit.flext_tests.test_builders import TestFlextTestsBuilders
    from tests.unit.flext_tests.test_docker import (
        TestContainerInfo,
        TestContainerStatus,
        TestFlextTestsDocker,
        TestFlextTestsDockerWorkerId,
        TestFlextTestsDockerWorkspaceRoot,
    )
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_factories import (
        TestConfig,
        TestFactoriesHelpers,
        TestFlextTestsFactoriesModernAPI,
        TestService,
        TestService as s,
        TestsFlextTestsFactoriesDict,
        TestsFlextTestsFactoriesGeneric,
        TestsFlextTestsFactoriesList,
        TestsFlextTestsFactoriesModel,
        TestsFlextTestsFactoriesRes,
        TestUser,
    )
    from tests.unit.flext_tests.test_files import (
        TestAssertExists,
        TestBatchOperations,
        TestCreateInStatic,
        TestFileInfo,
        TestFileInfoFromModels,
        TestFileInfoFromModels as m,
        TestFlextTestsFiles,
        TestFlextTestsFilesNewApi,
        TestInfoWithContentMeta,
        TestShortAlias,
    )
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import (
        TestFlextTestsUtilitiesFactory,
        TestFlextTestsUtilitiesResult,
        TestFlextTestsUtilitiesResult as r,
        TestFlextTestsUtilitiesResultCompat,
        TestFlextTestsUtilitiesTestContext,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestAssertExists": ("tests.unit.flext_tests.test_files", "TestAssertExists"),
    "TestBatchOperations": ("tests.unit.flext_tests.test_files", "TestBatchOperations"),
    "TestConfig": ("tests.unit.flext_tests.test_factories", "TestConfig"),
    "TestContainerInfo": ("tests.unit.flext_tests.test_docker", "TestContainerInfo"),
    "TestContainerStatus": (
        "tests.unit.flext_tests.test_docker",
        "TestContainerStatus",
    ),
    "TestCreateInStatic": ("tests.unit.flext_tests.test_files", "TestCreateInStatic"),
    "TestFactoriesHelpers": (
        "tests.unit.flext_tests.test_factories",
        "TestFactoriesHelpers",
    ),
    "TestFileInfo": ("tests.unit.flext_tests.test_files", "TestFileInfo"),
    "TestFileInfoFromModels": (
        "tests.unit.flext_tests.test_files",
        "TestFileInfoFromModels",
    ),
    "TestFlextTestsBuilders": (
        "tests.unit.flext_tests.test_builders",
        "TestFlextTestsBuilders",
    ),
    "TestFlextTestsDocker": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDocker",
    ),
    "TestFlextTestsDockerWorkerId": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkerId",
    ),
    "TestFlextTestsDockerWorkspaceRoot": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkspaceRoot",
    ),
    "TestFlextTestsDomains": (
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ),
    "TestFlextTestsFactoriesModernAPI": (
        "tests.unit.flext_tests.test_factories",
        "TestFlextTestsFactoriesModernAPI",
    ),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsFilesNewApi": (
        "tests.unit.flext_tests.test_files",
        "TestFlextTestsFilesNewApi",
    ),
    "TestFlextTestsMatchers": (
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ),
    "TestFlextTestsUtilitiesFactory": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesFactory",
    ),
    "TestFlextTestsUtilitiesResult": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResult",
    ),
    "TestFlextTestsUtilitiesResultCompat": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResultCompat",
    ),
    "TestFlextTestsUtilitiesTestContext": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesTestContext",
    ),
    "TestInfoWithContentMeta": (
        "tests.unit.flext_tests.test_files",
        "TestInfoWithContentMeta",
    ),
    "TestService": ("tests.unit.flext_tests.test_factories", "TestService"),
    "TestShortAlias": ("tests.unit.flext_tests.test_files", "TestShortAlias"),
    "TestUser": ("tests.unit.flext_tests.test_factories", "TestUser"),
    "TestsFlextTestsFactoriesDict": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesDict",
    ),
    "TestsFlextTestsFactoriesGeneric": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesGeneric",
    ),
    "TestsFlextTestsFactoriesList": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesList",
    ),
    "TestsFlextTestsFactoriesModel": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesModel",
    ),
    "TestsFlextTestsFactoriesRes": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesRes",
    ),
    "m": ("tests.unit.flext_tests.test_files", "TestFileInfoFromModels"),
    "r": ("tests.unit.flext_tests.test_utilities", "TestFlextTestsUtilitiesResult"),
    "s": ("tests.unit.flext_tests.test_factories", "TestService"),
}

__all__ = [
    "TestAssertExists",
    "TestBatchOperations",
    "TestConfig",
    "TestContainerInfo",
    "TestContainerStatus",
    "TestCreateInStatic",
    "TestFactoriesHelpers",
    "TestFileInfo",
    "TestFileInfoFromModels",
    "TestFlextTestsBuilders",
    "TestFlextTestsDocker",
    "TestFlextTestsDockerWorkerId",
    "TestFlextTestsDockerWorkspaceRoot",
    "TestFlextTestsDomains",
    "TestFlextTestsFactoriesModernAPI",
    "TestFlextTestsFiles",
    "TestFlextTestsFilesNewApi",
    "TestFlextTestsMatchers",
    "TestFlextTestsUtilitiesFactory",
    "TestFlextTestsUtilitiesResult",
    "TestFlextTestsUtilitiesResultCompat",
    "TestFlextTestsUtilitiesTestContext",
    "TestInfoWithContentMeta",
    "TestService",
    "TestShortAlias",
    "TestUser",
    "TestsFlextTestsFactoriesDict",
    "TestsFlextTestsFactoriesGeneric",
    "TestsFlextTestsFactoriesList",
    "TestsFlextTestsFactoriesModel",
    "TestsFlextTestsFactoriesRes",
    "m",
    "r",
    "s",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
