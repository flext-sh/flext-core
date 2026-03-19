# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Namespace validator test fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes

    from .rule0_loose_items import Rule0LooseItemsFixture
    from .rule0_multiple_classes import Rule0MultipleClassesFixture
    from .rule0_no_class import MAX_VALUE, helper
    from .rule0_wrong_prefix import RandomConstants
    from .rule1_loose_constant import DEFAULT_TIMEOUT, MAX_RETRIES
    from .rule1_loose_enum import Rule1LooseEnumFixture, Status
    from .rule1_magic_number import FlextTestUtilities, u
    from .rule1_valid_constants import FlextTestConstants, c
    from .rule2_typevar_wrong_module import FlextTestModels, m
    from .rule2_valid_types import FlextTestTypes, t
    from .typings import LooseTypeAlias

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_TIMEOUT": (
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "DEFAULT_TIMEOUT",
    ),
    "FlextTestConstants": (
        "tests.fixtures.namespace_validator.rule1_valid_constants",
        "FlextTestConstants",
    ),
    "FlextTestModels": (
        "tests.fixtures.namespace_validator.rule2_typevar_wrong_module",
        "FlextTestModels",
    ),
    "FlextTestTypes": (
        "tests.fixtures.namespace_validator.rule2_valid_types",
        "FlextTestTypes",
    ),
    "FlextTestUtilities": (
        "tests.fixtures.namespace_validator.rule1_magic_number",
        "FlextTestUtilities",
    ),
    "LooseTypeAlias": ("tests.fixtures.namespace_validator.typings", "LooseTypeAlias"),
    "MAX_RETRIES": (
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "MAX_RETRIES",
    ),
    "MAX_VALUE": ("tests.fixtures.namespace_validator.rule0_no_class", "MAX_VALUE"),
    "RandomConstants": (
        "tests.fixtures.namespace_validator.rule0_wrong_prefix",
        "RandomConstants",
    ),
    "Rule0LooseItemsFixture": (
        "tests.fixtures.namespace_validator.rule0_loose_items",
        "Rule0LooseItemsFixture",
    ),
    "Rule0MultipleClassesFixture": (
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "Rule0MultipleClassesFixture",
    ),
    "Rule1LooseEnumFixture": (
        "tests.fixtures.namespace_validator.rule1_loose_enum",
        "Rule1LooseEnumFixture",
    ),
    "Status": ("tests.fixtures.namespace_validator.rule1_loose_enum", "Status"),
    "c": ("tests.fixtures.namespace_validator.rule1_valid_constants", "c"),
    "helper": ("tests.fixtures.namespace_validator.rule0_no_class", "helper"),
    "m": ("tests.fixtures.namespace_validator.rule2_typevar_wrong_module", "m"),
    "t": ("tests.fixtures.namespace_validator.rule2_valid_types", "t"),
    "u": ("tests.fixtures.namespace_validator.rule1_magic_number", "u"),
}

__all__ = [
    "DEFAULT_TIMEOUT",
    "MAX_RETRIES",
    "MAX_VALUE",
    "FlextTestConstants",
    "FlextTestModels",
    "FlextTestTypes",
    "FlextTestUtilities",
    "LooseTypeAlias",
    "RandomConstants",
    "Rule0LooseItemsFixture",
    "Rule0MultipleClassesFixture",
    "Rule1LooseEnumFixture",
    "Status",
    "c",
    "helper",
    "m",
    "t",
    "u",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
