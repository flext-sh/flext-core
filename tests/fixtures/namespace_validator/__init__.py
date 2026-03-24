# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Namespace validator test fixtures."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.fixtures.namespace_validator.rule0_loose_items import (
        Rule0LooseItemsFixture,
    )
    from tests.fixtures.namespace_validator.rule0_multiple_classes import (
        FlextTestConstants,
        Rule0MultipleClassesFixture,
    )
    from tests.fixtures.namespace_validator.rule0_no_class import MAX_VALUE, helper
    from tests.fixtures.namespace_validator.rule0_wrong_prefix import RandomConstants
    from tests.fixtures.namespace_validator.rule1_loose_constant import (
        DEFAULT_TIMEOUT,
        MAX_RETRIES,
    )
    from tests.fixtures.namespace_validator.rule1_loose_enum import (
        FlextTestModels,
        Rule1LooseEnumFixture,
        Status,
    )
    from tests.fixtures.namespace_validator.rule1_magic_number import (
        FlextTestUtilities,
        u,
    )
    from tests.fixtures.namespace_validator.rule1_method_in_constants import c
    from tests.fixtures.namespace_validator.rule2_composite_type_loose import m
    from tests.fixtures.namespace_validator.rule2_protocol_in_types import (
        FlextTestTypes,
        t,
    )
    from tests.fixtures.namespace_validator.typings import LooseTypeAlias

    from flext_core import FlextTypes

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "DEFAULT_TIMEOUT": ["tests.fixtures.namespace_validator.rule1_loose_constant", "DEFAULT_TIMEOUT"],
    "FlextTestConstants": ["tests.fixtures.namespace_validator.rule0_multiple_classes", "FlextTestConstants"],
    "FlextTestModels": ["tests.fixtures.namespace_validator.rule1_loose_enum", "FlextTestModels"],
    "FlextTestTypes": ["tests.fixtures.namespace_validator.rule2_protocol_in_types", "FlextTestTypes"],
    "FlextTestUtilities": ["tests.fixtures.namespace_validator.rule1_magic_number", "FlextTestUtilities"],
    "LooseTypeAlias": ["tests.fixtures.namespace_validator.typings", "LooseTypeAlias"],
    "MAX_RETRIES": ["tests.fixtures.namespace_validator.rule1_loose_constant", "MAX_RETRIES"],
    "MAX_VALUE": ["tests.fixtures.namespace_validator.rule0_no_class", "MAX_VALUE"],
    "RandomConstants": ["tests.fixtures.namespace_validator.rule0_wrong_prefix", "RandomConstants"],
    "Rule0LooseItemsFixture": ["tests.fixtures.namespace_validator.rule0_loose_items", "Rule0LooseItemsFixture"],
    "Rule0MultipleClassesFixture": ["tests.fixtures.namespace_validator.rule0_multiple_classes", "Rule0MultipleClassesFixture"],
    "Rule1LooseEnumFixture": ["tests.fixtures.namespace_validator.rule1_loose_enum", "Rule1LooseEnumFixture"],
    "Status": ["tests.fixtures.namespace_validator.rule1_loose_enum", "Status"],
    "c": ["tests.fixtures.namespace_validator.rule1_method_in_constants", "c"],
    "helper": ["tests.fixtures.namespace_validator.rule0_no_class", "helper"],
    "m": ["tests.fixtures.namespace_validator.rule2_composite_type_loose", "m"],
    "t": ["tests.fixtures.namespace_validator.rule2_protocol_in_types", "t"],
    "u": ["tests.fixtures.namespace_validator.rule1_magic_number", "u"],
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
