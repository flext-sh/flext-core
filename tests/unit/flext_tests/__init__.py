# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import tests.unit.flext_tests.test_docker as _tests_unit_flext_tests_test_docker

    test_docker = _tests_unit_flext_tests_test_docker
    import tests.unit.flext_tests.test_domains as _tests_unit_flext_tests_test_domains
    from tests.unit.flext_tests.test_docker import TestDocker

    test_domains = _tests_unit_flext_tests_test_domains
    import tests.unit.flext_tests.test_files as _tests_unit_flext_tests_test_files
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains

    test_files = _tests_unit_flext_tests_test_files
    import tests.unit.flext_tests.test_matchers as _tests_unit_flext_tests_test_matchers
    from tests.unit.flext_tests.test_files import TestFlextTestsFiles

    test_matchers = _tests_unit_flext_tests_test_matchers
    import tests.unit.flext_tests.test_utilities as _tests_unit_flext_tests_test_utilities
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers

    test_utilities = _tests_unit_flext_tests_test_utilities
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
    from tests.unit.flext_tests.test_utilities import TestUtilities
_LAZY_IMPORTS = {
    "TestDocker": ("tests.unit.flext_tests.test_docker", "TestDocker"),
    "TestFlextTestsDomains": (
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsMatchers": (
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ),
    "TestUtilities": ("tests.unit.flext_tests.test_utilities", "TestUtilities"),
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "test_docker": "tests.unit.flext_tests.test_docker",
    "test_domains": "tests.unit.flext_tests.test_domains",
    "test_files": "tests.unit.flext_tests.test_files",
    "test_matchers": "tests.unit.flext_tests.test_matchers",
    "test_utilities": "tests.unit.flext_tests.test_utilities",
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
    "TestDocker",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestUtilities",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "test_docker",
    "test_domains",
    "test_files",
    "test_matchers",
    "test_utilities",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
