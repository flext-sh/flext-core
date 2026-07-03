"""Lazy export registry."""

from __future__ import annotations

from flext_core.lazy import merge_lazy_imports
from tests.unit._exports_lazy_part_01 import TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_01
from tests.unit._exports_lazy_part_02 import TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_02
from tests.unit._exports_lazy_part_03 import TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_03
from tests.unit._exports_lazy_part_04 import TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_04

_LOCAL_LAZY_IMPORTS = {
    **TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_01,
    **TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_02,
    **TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_03,
    **TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_04,
}

TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS = merge_lazy_imports(
    ("._models", "._utilities"),
    _LOCAL_LAZY_IMPORTS,
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name="tests.unit",
)

__all__: list[str] = ["TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS"]
