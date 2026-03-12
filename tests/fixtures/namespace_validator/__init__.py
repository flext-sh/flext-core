# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Namespace validator test fixtures."""

from __future__ import annotations

from typing import Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

# Lazy import mapping: export_name -> (module_path, attr_name)
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
    "MAX_RETRIES": (
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "MAX_RETRIES",
    ),
    "MAX_VALUE": ("tests.fixtures.namespace_validator.rule0_no_class", "MAX_VALUE"),
    "RandomConstants": (
        "tests.fixtures.namespace_validator.rule0_wrong_prefix",
        "RandomConstants",
    ),
    "Status": ("tests.fixtures.namespace_validator.rule1_loose_enum", "Status"),
    "c": ("tests.fixtures.namespace_validator.rule1_valid_constants", "c"),
    "helper": ("tests.fixtures.namespace_validator.rule0_no_class", "helper"),
    "m": ("tests.fixtures.namespace_validator.rule2_typevar_wrong_module", "m"),
    "t": ("tests.fixtures.namespace_validator.rule2_valid_types", "t"),
    "u": ("tests.fixtures.namespace_validator.rule1_magic_number", "u"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(_LAZY_IMPORTS)
    return sorted(_LAZY_IMPORTS)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
