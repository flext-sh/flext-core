"""Root lazy-export names that should be excluded from dynamic re-exports."""

from __future__ import annotations

from typing import Final

ROOT_EXCLUDE_NAMES: Final[tuple[str, ...]] = (
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
)

__all__: list[str] = ["ROOT_EXCLUDE_NAMES"]
