# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext tests package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.unit.flext_tests import (
        test_docker as test_docker,
        test_domains as test_domains,
        test_files as test_files,
        test_matchers as test_matchers,
        test_utilities as test_utilities,
    )
    from tests.unit.flext_tests.test_docker import TestDocker as TestDocker
    from tests.unit.flext_tests.test_domains import (
        TestFlextTestsDomains as TestFlextTestsDomains,
    )
    from tests.unit.flext_tests.test_files import (
        TestFlextTestsFiles as TestFlextTestsFiles,
    )
    from tests.unit.flext_tests.test_matchers import (
        TestFlextTestsMatchers as TestFlextTestsMatchers,
    )
    from tests.unit.flext_tests.test_utilities import TestUtilities as TestUtilities

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "TestDocker": ["tests.unit.flext_tests.test_docker", "TestDocker"],
    "TestFlextTestsDomains": [
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ],
    "TestFlextTestsFiles": ["tests.unit.flext_tests.test_files", "TestFlextTestsFiles"],
    "TestFlextTestsMatchers": [
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ],
    "TestUtilities": ["tests.unit.flext_tests.test_utilities", "TestUtilities"],
    "test_docker": ["tests.unit.flext_tests.test_docker", ""],
    "test_domains": ["tests.unit.flext_tests.test_domains", ""],
    "test_files": ["tests.unit.flext_tests.test_files", ""],
    "test_matchers": ["tests.unit.flext_tests.test_matchers", ""],
    "test_utilities": ["tests.unit.flext_tests.test_utilities", ""],
}

_EXPORTS: Sequence[str] = [
    "TestDocker",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestUtilities",
    "test_docker",
    "test_domains",
    "test_files",
    "test_matchers",
    "test_utilities",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
