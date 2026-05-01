"""Enforcement rule dispatcher — predicate routing via data-driven MRO visitors.

Engine combines visitor mixins for field + model, attributes, methods, classes,
modules, imports, and deprecated syntax checks. Helper methods live in
FlextUtilitiesBeartypeHelpers; visitors in domain-specific mixin classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import ClassVar, no_type_check, override

# Side-effect: monkey-patch beartype cave so typing_extensions.TypeAliasType
# (used by pydantic.JsonValue et al.) is accepted as a PEP-695 alias.
from flext_core._utilities.beartype_typingext_patch import (
    FlextUtilitiesBeartypeTypingExtPatch as _FlextUtilitiesBeartypeTypingExtPatch,
)

_: type[_FlextUtilitiesBeartypeTypingExtPatch]
from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.attr_visitor import (
    FlextUtilitiesBeartypeAttrVisitor,
)
from flext_core._utilities._beartype.class_visitor import (
    FlextUtilitiesBeartypeClassVisitor,
)
from flext_core._utilities._beartype.deprecated_visitor import (
    FlextUtilitiesBeartypeDeprecatedVisitor,
)
from flext_core._utilities._beartype.field_visitor import (
    FlextUtilitiesBeartypeFieldVisitor,
)
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers,
)
from flext_core._utilities._beartype.import_visitor import (
    FlextUtilitiesBeartypeImportVisitor,
)
from flext_core._utilities._beartype.method_visitor import (
    FlextUtilitiesBeartypeMethodVisitor,
)
from flext_core._utilities._beartype.module_visitor import (
    FlextUtilitiesBeartypeModuleVisitor,
)

_NO_VIOLATION: t.StrMapping | None = None


@no_type_check
class FlextUtilitiesBeartypeEngine(
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesBeartypeFieldVisitor,
    FlextUtilitiesBeartypeAttrVisitor,
    FlextUtilitiesBeartypeMethodVisitor,
    FlextUtilitiesBeartypeClassVisitor,
    FlextUtilitiesBeartypeModuleVisitor,
    FlextUtilitiesBeartypeImportVisitor,
    FlextUtilitiesBeartypeDeprecatedVisitor,
):
    """Annotation inspection + per-tag rule predicates via data-driven visitors."""

    @staticmethod
    def defined_inside(inner_cls: type, outer_qualname: str) -> bool:
        return getattr(inner_cls, "__qualname__", "").startswith(f"{outer_qualname}.")

    @staticmethod
    def defined_in_function_scope(target: type) -> bool:
        return "<locals>" in getattr(target, "__qualname__", "")

    @staticmethod
    def attr_accept_constants(name: str, value: object) -> bool:
        if name.startswith("_") or name in c.ENFORCEMENT_CONSTANTS_SKIP_ATTRS:
            return False
        if isinstance(value, (type, classmethod, staticmethod, property)):
            return False
        return not callable(value)

    @staticmethod
    def attr_accept_public(name: str) -> bool:
        return not name.startswith("_")

    @staticmethod
    def attr_accept_utility(name: str) -> bool:
        return (
            name not in c.ENFORCEMENT_UTILITIES_EXEMPT_METHODS
        ) and not name.startswith("_")

    @staticmethod
    def contains_any(hint: t.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeHelpers.contains_any_recursive(
            hint,
            seen=set(),
        )

    @override
    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        return FlextUtilitiesBeartypeHelpers.has_forbidden_collection_origin(
            hint,
            forbidden,
        )

    @override
    @staticmethod
    def has_runtime_protocol_marker(value: type) -> bool:
        return FlextUtilitiesBeartypeHelpers.has_runtime_protocol_marker(value)

    @override
    @staticmethod
    def has_nested_namespace(value: type) -> bool:
        return FlextUtilitiesBeartypeHelpers.has_nested_namespace(value)

    @override
    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        return FlextUtilitiesBeartypeHelpers.count_union_members(hint)

    @override
    @staticmethod
    def matches_str_none_union(hint: t.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeHelpers.matches_str_none_union(hint)

    @override
    @staticmethod
    def alias_contains_any(alias_value: t.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeHelpers.alias_contains_any(alias_value)

    @classmethod
    def apply(
        cls, kind: c.EnforcementPredicateKind, params: mp.BaseModel, *args: object
    ) -> t.StrMapping | None:
        """Dispatch a rule predicate to its visitor by ``predicate_kind``."""
        visitor = cls._VISITORS.get(kind)
        return _NO_VIOLATION if visitor is None else visitor(params, *args)

    _VISITORS: ClassVar[
        t.MappingKV[c.EnforcementPredicateKind, Callable[..., t.StrMapping | None]]
    ] = MappingProxyType({
        c.EnforcementPredicateKind.FIELD_SHAPE: FlextUtilitiesBeartypeFieldVisitor.v_field_shape,
        c.EnforcementPredicateKind.MODEL_CONFIG: FlextUtilitiesBeartypeFieldVisitor.v_model_config,
        c.EnforcementPredicateKind.ATTR_SHAPE: FlextUtilitiesBeartypeAttrVisitor.v_attr_shape,
        c.EnforcementPredicateKind.METHOD_SHAPE: FlextUtilitiesBeartypeMethodVisitor.v_method_shape,
        c.EnforcementPredicateKind.CLASS_PLACEMENT: FlextUtilitiesBeartypeClassVisitor.v_class_placement,
        c.EnforcementPredicateKind.PROTOCOL_TREE: FlextUtilitiesBeartypeClassVisitor.v_protocol_tree,
        c.EnforcementPredicateKind.MRO_SHAPE: FlextUtilitiesBeartypeClassVisitor.v_mro_shape,
        c.EnforcementPredicateKind.LOOSE_SYMBOL: FlextUtilitiesBeartypeClassVisitor.v_loose_symbol,
        c.EnforcementPredicateKind.WRAPPER: FlextUtilitiesBeartypeDeprecatedVisitor.v_wrapper,
        c.EnforcementPredicateKind.IMPORT_BLACKLIST: FlextUtilitiesBeartypeImportVisitor.v_import_blacklist,
        c.EnforcementPredicateKind.ALIAS_REBIND: FlextUtilitiesBeartypeImportVisitor.v_alias_rebind,
        c.EnforcementPredicateKind.LIBRARY_IMPORT: FlextUtilitiesBeartypeImportVisitor.v_library_import,
        c.EnforcementPredicateKind.LOC_CAP: FlextUtilitiesBeartypeModuleVisitor.v_loc_cap,
        c.EnforcementPredicateKind.DUPLICATE_SYMBOL: FlextUtilitiesBeartypeModuleVisitor.v_duplicate_symbol,
        c.EnforcementPredicateKind.DEPRECATED_SYNTAX: FlextUtilitiesBeartypeDeprecatedVisitor.v_deprecated_syntax,
    })


ube = FlextUtilitiesBeartypeEngine

__all__: list[str] = ["FlextUtilitiesBeartypeEngine", "ube"]
