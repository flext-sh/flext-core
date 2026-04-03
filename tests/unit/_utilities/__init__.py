# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Utilities package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports
from tests.unit._utilities.test_guards import TestFlextUtilitiesGuards
from tests.unit._utilities.test_mapper import TestFlextUtilitiesMapper

if _t.TYPE_CHECKING:
    import tests.unit._utilities.test_guards as _tests_unit__utilities_test_guards

    test_guards = _tests_unit__utilities_test_guards
    import tests.unit._utilities.test_mapper as _tests_unit__utilities_test_mapper

    test_mapper = _tests_unit__utilities_test_mapper

    _ = (
        TestFlextUtilitiesGuards,
        TestFlextUtilitiesMapper,
        test_guards,
        test_mapper,
    )
_LAZY_IMPORTS = {
    "TestFlextUtilitiesGuards": "tests.unit._utilities.test_guards",
    "TestFlextUtilitiesMapper": "tests.unit._utilities.test_mapper",
    "test_guards": "tests.unit._utilities.test_guards",
    "test_mapper": "tests.unit._utilities.test_mapper",
}

__all__ = [
    "TestFlextUtilitiesGuards",
    "TestFlextUtilitiesMapper",
    "test_guards",
    "test_mapper",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
