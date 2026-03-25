# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Check package."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_infra.check.services import (
        DEFAULT_GATES,
        PyreflyConfigFixer,
        WorkspaceChecker,
        run_cli,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "DEFAULT_GATES": ["flext_infra.check.services", "DEFAULT_GATES"],
    "PyreflyConfigFixer": ["flext_infra.check.services", "PyreflyConfigFixer"],
    "WorkspaceChecker": ["flext_infra.check.services", "WorkspaceChecker"],
    "run_cli": ["flext_infra.check.services", "run_cli"],
}

__all__ = [
    "DEFAULT_GATES",
    "PyreflyConfigFixer",
    "WorkspaceChecker",
    "run_cli",
]


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.
    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)