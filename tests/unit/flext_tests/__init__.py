# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Flext tests package."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.unit.flext_tests.test_docker import TestDocker
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_files import TestFlextTestsFiles
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import TestUtilities

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "TestDocker": ["tests.unit.flext_tests.test_docker", "TestDocker"],
    "TestFlextTestsDomains": ["tests.unit.flext_tests.test_domains", "TestFlextTestsDomains"],
    "TestFlextTestsFiles": ["tests.unit.flext_tests.test_files", "TestFlextTestsFiles"],
    "TestFlextTestsMatchers": ["tests.unit.flext_tests.test_matchers", "TestFlextTestsMatchers"],
    "TestUtilities": ["tests.unit.flext_tests.test_utilities", "TestUtilities"],
}

__all__ = [
    "TestDocker",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestUtilities",
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