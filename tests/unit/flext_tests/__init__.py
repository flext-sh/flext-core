# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext tests package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
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
    from tests.unit.flext_tests import (
        test_docker,
        test_domains,
        test_files,
        test_matchers,
        test_utilities,
    )
    from tests.unit.flext_tests.test_docker import TestDocker
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_files import TestFlextTestsFiles
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import TestUtilities

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "TestDocker": "tests.unit.flext_tests.test_docker",
    "TestFlextTestsDomains": "tests.unit.flext_tests.test_domains",
    "TestFlextTestsFiles": "tests.unit.flext_tests.test_files",
    "TestFlextTestsMatchers": "tests.unit.flext_tests.test_matchers",
    "TestUtilities": "tests.unit.flext_tests.test_utilities",
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
