"""Pytest dispatcher facade for the cross-layer enforcement catalog."""

from __future__ import annotations

from flext_tests._fixtures._enforcement_parts.config import (
    active_rules,
    discover_workspace_root,
    pytest_configure,
    split_csv,
)
from flext_tests._fixtures._enforcement_parts.hooks import (
    pytest_collection_modifyitems,
    pytest_sessionstart,
    pytest_terminal_summary,
    pytest_warning_recorded,
)
from flext_tests._fixtures._enforcement_parts.items import (
    EnforcementCollector,
    EnforcementItem,
    EnforcementViolationError,
)
from flext_tests.enforcement_plugin import pytest_addoption

__all__: list[str] = [
    "EnforcementCollector",
    "EnforcementItem",
    "EnforcementViolationError",
    "active_rules",
    "discover_workspace_root",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_sessionstart",
    "pytest_terminal_summary",
    "pytest_warning_recorded",
    "split_csv",
]
