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
        test_docker,
        test_domains,
        test_files,
        test_matchers,
        test_utilities,
    )
    from tests.unit.flext_tests.test_docker import *
    from tests.unit.flext_tests.test_domains import *
    from tests.unit.flext_tests.test_files import *
    from tests.unit.flext_tests.test_matchers import *
    from tests.unit.flext_tests.test_utilities import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "TestDocker": "tests.unit.flext_tests.test_docker",
    "TestFlextTestsDomains": "tests.unit.flext_tests.test_domains",
    "TestFlextTestsFiles": "tests.unit.flext_tests.test_files",
    "TestFlextTestsMatchers": "tests.unit.flext_tests.test_matchers",
    "TestUtilities": "tests.unit.flext_tests.test_utilities",
    "test_docker": "tests.unit.flext_tests.test_docker",
    "test_domains": "tests.unit.flext_tests.test_domains",
    "test_files": "tests.unit.flext_tests.test_files",
    "test_matchers": "tests.unit.flext_tests.test_matchers",
    "test_utilities": "tests.unit.flext_tests.test_utilities",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
