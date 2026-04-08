# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Namespace validator package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import tests.fixtures.namespace_validator.rule0_loose_items as _tests_fixtures_namespace_validator_rule0_loose_items

    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.protocols import FlextProtocols as p
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u

    rule0_loose_items = _tests_fixtures_namespace_validator_rule0_loose_items
    import tests.fixtures.namespace_validator.rule0_multiple_classes as _tests_fixtures_namespace_validator_rule0_multiple_classes

    rule0_multiple_classes = _tests_fixtures_namespace_validator_rule0_multiple_classes
    import tests.fixtures.namespace_validator.rule0_no_class as _tests_fixtures_namespace_validator_rule0_no_class

    rule0_no_class = _tests_fixtures_namespace_validator_rule0_no_class
    import tests.fixtures.namespace_validator.rule0_valid as _tests_fixtures_namespace_validator_rule0_valid

    rule0_valid = _tests_fixtures_namespace_validator_rule0_valid
    import tests.fixtures.namespace_validator.rule0_wrong_prefix as _tests_fixtures_namespace_validator_rule0_wrong_prefix

    rule0_wrong_prefix = _tests_fixtures_namespace_validator_rule0_wrong_prefix
    import tests.fixtures.namespace_validator.rule1_loose_constant as _tests_fixtures_namespace_validator_rule1_loose_constant

    rule1_loose_constant = _tests_fixtures_namespace_validator_rule1_loose_constant
    import tests.fixtures.namespace_validator.rule1_loose_enum as _tests_fixtures_namespace_validator_rule1_loose_enum

    rule1_loose_enum = _tests_fixtures_namespace_validator_rule1_loose_enum
    import tests.fixtures.namespace_validator.rule1_magic_number as _tests_fixtures_namespace_validator_rule1_magic_number

    rule1_magic_number = _tests_fixtures_namespace_validator_rule1_magic_number
    import tests.fixtures.namespace_validator.rule1_method_in_constants as _tests_fixtures_namespace_validator_rule1_method_in_constants

    rule1_method_in_constants = (
        _tests_fixtures_namespace_validator_rule1_method_in_constants
    )
    import tests.fixtures.namespace_validator.rule1_valid_constants as _tests_fixtures_namespace_validator_rule1_valid_constants

    rule1_valid_constants = _tests_fixtures_namespace_validator_rule1_valid_constants
    import tests.fixtures.namespace_validator.rule2_composite_type_loose as _tests_fixtures_namespace_validator_rule2_composite_type_loose

    rule2_composite_type_loose = (
        _tests_fixtures_namespace_validator_rule2_composite_type_loose
    )
    import tests.fixtures.namespace_validator.rule2_protocol_in_types as _tests_fixtures_namespace_validator_rule2_protocol_in_types

    rule2_protocol_in_types = (
        _tests_fixtures_namespace_validator_rule2_protocol_in_types
    )
    import tests.fixtures.namespace_validator.rule2_typevar_in_class as _tests_fixtures_namespace_validator_rule2_typevar_in_class

    rule2_typevar_in_class = _tests_fixtures_namespace_validator_rule2_typevar_in_class
    import tests.fixtures.namespace_validator.rule2_typevar_wrong_module as _tests_fixtures_namespace_validator_rule2_typevar_wrong_module

    rule2_typevar_wrong_module = (
        _tests_fixtures_namespace_validator_rule2_typevar_wrong_module
    )
    import tests.fixtures.namespace_validator.rule2_valid_types as _tests_fixtures_namespace_validator_rule2_valid_types

    rule2_valid_types = _tests_fixtures_namespace_validator_rule2_valid_types
    import tests.fixtures.namespace_validator.typings as _tests_fixtures_namespace_validator_typings

    typings = _tests_fixtures_namespace_validator_typings
_LAZY_IMPORTS = {
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
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
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "typings": "tests.fixtures.namespace_validator.typings",
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
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
    "s",
    "t",
    "typings",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
