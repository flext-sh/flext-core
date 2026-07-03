# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core.tests.unit._utilities.test_guards import (
        TestsFlextUtilitiesGuards as TestsFlextUtilitiesGuards,
    )
    from flext_core.tests.unit._utilities.test_mapper import (
        TestsFlextUtilitiesMapper as TestsFlextUtilitiesMapper,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_guards": ("TestsFlextUtilitiesGuards",),
        ".test_mapper": ("TestsFlextUtilitiesMapper",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
