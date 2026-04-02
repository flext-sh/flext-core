# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Namespace validator test fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from tests.fixtures.namespace_validator import (
        rule0_loose_items,
        rule0_multiple_classes,
        rule0_no_class,
        rule0_valid,
        rule0_wrong_prefix,
        rule1_loose_constant,
        rule1_loose_enum,
        rule1_magic_number,
        rule1_method_in_constants,
        rule1_valid_constants,
        rule2_composite_type_loose,
        rule2_protocol_in_types,
        rule2_typevar_in_class,
        rule2_typevar_wrong_module,
        rule2_valid_types,
        typings,
    )
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

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "DEFAULT_TIMEOUT": "tests.fixtures.namespace_validator.rule1_loose_constant",
    "FlextTestConstants": "tests.fixtures.namespace_validator.rule0_multiple_classes",
    "FlextTestModels": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "FlextTestTypes": "tests.fixtures.namespace_validator.rule2_protocol_in_types",
    "FlextTestUtilities": "tests.fixtures.namespace_validator.rule1_magic_number",
    "LooseTypeAlias": "tests.fixtures.namespace_validator.typings",
    "MAX_RETRIES": "tests.fixtures.namespace_validator.rule1_loose_constant",
    "MAX_VALUE": "tests.fixtures.namespace_validator.rule0_no_class",
    "RandomConstants": "tests.fixtures.namespace_validator.rule0_wrong_prefix",
    "Rule0LooseItemsFixture": "tests.fixtures.namespace_validator.rule0_loose_items",
    "Rule0MultipleClassesFixture": "tests.fixtures.namespace_validator.rule0_multiple_classes",
    "Rule1LooseEnumFixture": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "Status": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "c": "tests.fixtures.namespace_validator.rule1_method_in_constants",
    "helper": "tests.fixtures.namespace_validator.rule0_no_class",
    "m": "tests.fixtures.namespace_validator.rule2_composite_type_loose",
    "rule0_loose_items": "tests.fixtures.namespace_validator.rule0_loose_items",
    "rule0_multiple_classes": "tests.fixtures.namespace_validator.rule0_multiple_classes",
    "rule0_no_class": "tests.fixtures.namespace_validator.rule0_no_class",
    "rule0_valid": "tests.fixtures.namespace_validator.rule0_valid",
    "rule0_wrong_prefix": "tests.fixtures.namespace_validator.rule0_wrong_prefix",
    "rule1_loose_constant": "tests.fixtures.namespace_validator.rule1_loose_constant",
    "rule1_loose_enum": "tests.fixtures.namespace_validator.rule1_loose_enum",
    "rule1_magic_number": "tests.fixtures.namespace_validator.rule1_magic_number",
    "rule1_method_in_constants": "tests.fixtures.namespace_validator.rule1_method_in_constants",
    "rule1_valid_constants": "tests.fixtures.namespace_validator.rule1_valid_constants",
    "rule2_composite_type_loose": "tests.fixtures.namespace_validator.rule2_composite_type_loose",
    "rule2_protocol_in_types": "tests.fixtures.namespace_validator.rule2_protocol_in_types",
    "rule2_typevar_in_class": "tests.fixtures.namespace_validator.rule2_typevar_in_class",
    "rule2_typevar_wrong_module": "tests.fixtures.namespace_validator.rule2_typevar_wrong_module",
    "rule2_valid_types": "tests.fixtures.namespace_validator.rule2_valid_types",
    "t": "tests.fixtures.namespace_validator.rule2_protocol_in_types",
    "typings": "tests.fixtures.namespace_validator.typings",
    "u": "tests.fixtures.namespace_validator.rule1_magic_number",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
