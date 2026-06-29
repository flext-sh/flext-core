"""Lazy export map part 04."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_04 = build_lazy_import_map({
    "flext_tests": (
        "c",
        "d",
        "e",
        "h",
        "m",
        "p",
        "r",
        "s",
        "t",
        "td",
        "tf",
        "tk",
        "tm",
        "tv",
        "u",
        "x",
    ),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_04"]
