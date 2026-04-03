# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Utilities package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.unit._utilities import test_guards, test_mapper
    from tests.unit._utilities.test_guards import TestFlextUtilitiesGuards
    from tests.unit._utilities.test_mapper import TestFlextUtilitiesMapper

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "TestFlextUtilitiesGuards": "tests.unit._utilities.test_guards",
    "TestFlextUtilitiesMapper": "tests.unit._utilities.test_mapper",
    "test_guards": "tests.unit._utilities.test_guards",
    "test_mapper": "tests.unit._utilities.test_mapper",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
