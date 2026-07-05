# AUTO-GENERATED FILE — Regenerate with: make gen
"""Lazy export registry."""

from __future__ import annotations

from examples._exports_lazy_part_01 import EXAMPLES_LAZY_IMPORTS_PART_01
from examples._exports_lazy_part_02 import EXAMPLES_LAZY_IMPORTS_PART_02
from flext_core.lazy import merge_lazy_imports

_LOCAL_LAZY_IMPORTS = {
    **EXAMPLES_LAZY_IMPORTS_PART_01,
    **EXAMPLES_LAZY_IMPORTS_PART_02,
}

EXAMPLES_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._models",
        "._shared_parts",
    ),
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
    module_name="examples",
)

__all__: list[str] = ["EXAMPLES_LAZY_IMPORTS"]
