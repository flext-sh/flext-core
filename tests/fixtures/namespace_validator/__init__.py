# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Namespace validator test fixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core.lazy import install_lazy_exports

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "DEFAULT_TIMEOUT": [
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "DEFAULT_TIMEOUT",
    ],
    "FlextTestConstants": [
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "FlextTestConstants",
    ],
    "FlextTestModels": [
        "tests.fixtures.namespace_validator.rule1_loose_enum",
        "FlextTestModels",
    ],
    "FlextTestTypes": [
        "tests.fixtures.namespace_validator.rule2_protocol_in_types",
        "FlextTestTypes",
    ],
    "FlextTestUtilities": [
        "tests.fixtures.namespace_validator.rule1_magic_number",
        "FlextTestUtilities",
    ],
    "LooseTypeAlias": ["tests.fixtures.namespace_validator.typings", "LooseTypeAlias"],
    "MAX_RETRIES": [
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "MAX_RETRIES",
    ],
    "MAX_VALUE": ["tests.fixtures.namespace_validator.rule0_no_class", "MAX_VALUE"],
    "RandomConstants": [
        "tests.fixtures.namespace_validator.rule0_wrong_prefix",
        "RandomConstants",
    ],
    "Rule0LooseItemsFixture": [
        "tests.fixtures.namespace_validator.rule0_loose_items",
        "Rule0LooseItemsFixture",
    ],
    "Rule0MultipleClassesFixture": [
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "Rule0MultipleClassesFixture",
    ],
    "Rule1LooseEnumFixture": [
        "tests.fixtures.namespace_validator.rule1_loose_enum",
        "Rule1LooseEnumFixture",
    ],
    "Status": ["tests.fixtures.namespace_validator.rule1_loose_enum", "Status"],
    "c": ["tests.fixtures.namespace_validator.rule1_method_in_constants", "c"],
    "helper": ["tests.fixtures.namespace_validator.rule0_no_class", "helper"],
    "m": ["tests.fixtures.namespace_validator.rule2_composite_type_loose", "m"],
    "rule0_loose_items": ["tests.fixtures.namespace_validator.rule0_loose_items", ""],
    "rule0_multiple_classes": [
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "",
    ],
    "rule0_no_class": ["tests.fixtures.namespace_validator.rule0_no_class", ""],
    "rule0_valid": ["tests.fixtures.namespace_validator.rule0_valid", ""],
    "rule0_wrong_prefix": ["tests.fixtures.namespace_validator.rule0_wrong_prefix", ""],
    "rule1_loose_constant": [
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "",
    ],
    "rule1_loose_enum": ["tests.fixtures.namespace_validator.rule1_loose_enum", ""],
    "rule1_magic_number": ["tests.fixtures.namespace_validator.rule1_magic_number", ""],
    "rule1_method_in_constants": [
        "tests.fixtures.namespace_validator.rule1_method_in_constants",
        "",
    ],
    "rule1_valid_constants": [
        "tests.fixtures.namespace_validator.rule1_valid_constants",
        "",
    ],
    "rule2_composite_type_loose": [
        "tests.fixtures.namespace_validator.rule2_composite_type_loose",
        "",
    ],
    "rule2_protocol_in_types": [
        "tests.fixtures.namespace_validator.rule2_protocol_in_types",
        "",
    ],
    "rule2_typevar_in_class": [
        "tests.fixtures.namespace_validator.rule2_typevar_in_class",
        "",
    ],
    "rule2_typevar_wrong_module": [
        "tests.fixtures.namespace_validator.rule2_typevar_wrong_module",
        "",
    ],
    "rule2_valid_types": ["tests.fixtures.namespace_validator.rule2_valid_types", ""],
    "t": ["tests.fixtures.namespace_validator.rule2_protocol_in_types", "t"],
    "typings": ["tests.fixtures.namespace_validator.typings", ""],
    "u": ["tests.fixtures.namespace_validator.rule1_magic_number", "u"],
}

_EXPORTS: Sequence[str] = [
    "DEFAULT_TIMEOUT",
    "FlextTestConstants",
    "FlextTestModels",
    "FlextTestTypes",
    "FlextTestUtilities",
    "LooseTypeAlias",
    "MAX_RETRIES",
    "MAX_VALUE",
    "RandomConstants",
    "Rule0LooseItemsFixture",
    "Rule0MultipleClassesFixture",
    "Rule1LooseEnumFixture",
    "Status",
    "c",
    "helper",
    "m",
    "rule0_loose_items",
    "rule0_multiple_classes",
    "rule0_no_class",
    "rule0_valid",
    "rule0_wrong_prefix",
    "rule1_loose_constant",
    "rule1_loose_enum",
    "rule1_magic_number",
    "rule1_method_in_constants",
    "rule1_valid_constants",
    "rule2_composite_type_loose",
    "rule2_protocol_in_types",
    "rule2_typevar_in_class",
    "rule2_typevar_wrong_module",
    "rule2_valid_types",
    "t",
    "typings",
    "u",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
