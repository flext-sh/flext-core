# AUTO-GENERATED FILE — Regenerate with: make gen
"""Lazy export registry."""

from __future__ import annotations

from flext_core._utilities._exports_lazy_part_01 import (
    FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_01,
)
from flext_core._utilities._exports_lazy_part_02 import (
    FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_02,
)
from flext_core._utilities._exports_lazy_part_03 import (
    FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_03,
)
from flext_core.lazy import merge_lazy_imports

_LOCAL_LAZY_IMPORTS = {
    **FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_01,
    **FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_02,
    **FLEXT_CORE__UTILITIES_LAZY_IMPORTS_PART_03,
}

FLEXT_CORE__UTILITIES_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._beartype",
        "._checker_parts",
        "._enforcement_collect_parts",
        "._enforcement_parts",
        "._generators_parts",
        "._guards_parts",
        "._logging_config_parts",
        "._logging_context_parts",
        "._mapper_access_parts",
        "._mapper_extract_parts",
        "._parser_targets_parts",
        "._project_metadata_parts",
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
    module_name="flext_core._utilities",
)

__all__: list[str] = ["FLEXT_CORE__UTILITIES_LAZY_IMPORTS"]
