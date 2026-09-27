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
from typing import ClassVar, TypeAliasType, override

from pydantic.fields import FieldInfo

from .._constants.enforcement import FlextConstantsEnforcement as c
from .._models.enforcement import FlextModelsEnforcement as me
from .._models.pydantic import FlextModelsPydantic as mp
from .._protocols.base import FlextProtocolsBase as p
from .._typings.base import FlextTypingBase as t
from ._beartype._helpers_parts.helpers_part_03 import FlextUtilitiesBeartypeHelpers
from ._beartype.attr_visitor import FlextUtilitiesBeartypeAttrVisitor
from ._beartype.class_visitor import FlextUtilitiesBeartypeClassVisitor
from ._beartype.deprecated_visitor import FlextUtilitiesBeartypeDeprecatedVisitor
from ._beartype.field_visitor import FlextUtilitiesBeartypeFieldVisitor
from ._beartype.import_visitor import FlextUtilitiesBeartypeImportVisitor
from ._beartype.method_visitor import FlextUtilitiesBeartypeMethodVisitor
from ._beartype.module_visitor import FlextUtilitiesBeartypeModuleVisitor
from ._beartype.type_aliases import FlextUtilitiesBeartypeTypeAliases
from .beartype_typingext_patch import (
    FlextUtilitiesBeartypeTypingExtPatch as _FlextUtilitiesBeartypeTypingExtPatch,
)

_NO_VIOLATION: t.StrMapping | None = None
# Side-effect: monkey-patch beartype cave so typing_extensions.TypeAliasType
# (used by pydantic.JsonValue et al.) is accepted as a PEP-695 alias.
_FlextUtilitiesBeartypeTypingExtPatch.apply()


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
    def attr_accept_constants(name: str, value: p.AttributeProbe) -> bool:
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
        return FlextUtilitiesBeartypeHelpers.contains_any_recursive(hint, seen=set())

    @staticmethod
    def deferred_aliases(
        params: mp.BaseModel, owner: type, *args: p.AttributeProbe
    ) -> tuple[me.DeferredAlias, ...]:
        """Account for unavailable alias values before a value-dependent rule."""
        if isinstance(params, me.AttrShapeParams):
            if params.forbid_any_in_alias:
                match args:
                    case (_, alias) if isinstance(alias, TypeAliasType):
                        return FlextUtilitiesBeartypeTypeAliases.deferred(
                            alias, recursive=True, owner=owner
                        )
                    case _:
                        return ()
            return ()
        if not isinstance(params, me.FieldShapeParams) or not args:
            return ()
        info = args[-1]
        if not isinstance(info, FieldInfo) or params.require_description:
            return ()
        if params.forbid_any or params.forbid_bare_collection:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation, recursive=params.forbid_any, owner=owner
            )
        if params.forbid_mutable_default:
            return ()
        if (
            params.forbid_raw_default_factory
            and FlextUtilitiesBeartypeHelpers.mutable_default_factory_kind(
                info.default_factory
            )
            is not None
        ):
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation, unwrap_annotated=True, inspect_origin=True, owner=owner
            )
        if params.forbid_str_none_empty:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation, owner=owner
            )
        return ()

    @override
    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None, forbidden: frozenset[str]
    ) -> tuple[bool, str]:
        return FlextUtilitiesBeartypeHelpers.has_forbidden_collection_origin(
            hint, forbidden
        )

    @classmethod
    def apply(
        cls,
        kind: c.EnforcementPredicateKind,
        params: mp.BaseModel,
        *args: p.AttributeProbe,
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
        c.EnforcementPredicateKind.CLASSVAR_CONSTANT: FlextUtilitiesBeartypeAttrVisitor.v_classvar_constant,
        c.EnforcementPredicateKind.METHOD_SHAPE: FlextUtilitiesBeartypeMethodVisitor.v_method_shape,
        c.EnforcementPredicateKind.CLASS_PLACEMENT: FlextUtilitiesBeartypeClassVisitor.v_class_placement,
        c.EnforcementPredicateKind.PROTOCOL_TREE: FlextUtilitiesBeartypeClassVisitor.v_protocol_tree,
        c.EnforcementPredicateKind.MRO_SHAPE: FlextUtilitiesBeartypeClassVisitor.v_mro_shape,
        c.EnforcementPredicateKind.LOOSE_SYMBOL: FlextUtilitiesBeartypeClassVisitor.v_loose_symbol,
        c.EnforcementPredicateKind.WRAPPER: FlextUtilitiesBeartypeDeprecatedVisitor.v_wrapper,
        c.EnforcementPredicateKind.IMPORT_BLACKLIST: FlextUtilitiesBeartypeImportVisitor.v_import_blacklist,
        c.EnforcementPredicateKind.FOREIGN_CANONICAL_ALIAS_IMPORT: FlextUtilitiesBeartypeImportVisitor.v_foreign_canonical_alias_import,
        c.EnforcementPredicateKind.ALIAS_REBIND: FlextUtilitiesBeartypeImportVisitor.v_alias_rebind,
        c.EnforcementPredicateKind.COMPATIBILITY_ALIAS: FlextUtilitiesBeartypeImportVisitor.v_compatibility_alias,
        c.EnforcementPredicateKind.LIBRARY_IMPORT: FlextUtilitiesBeartypeImportVisitor.v_library_import,
        c.EnforcementPredicateKind.LOC_CAP: FlextUtilitiesBeartypeModuleVisitor.v_loc_cap,
        c.EnforcementPredicateKind.MODULE_ALIAS: FlextUtilitiesBeartypeModuleVisitor.v_module_alias,
        c.EnforcementPredicateKind.DUPLICATE_SYMBOL: FlextUtilitiesBeartypeModuleVisitor.v_duplicate_symbol,
        c.EnforcementPredicateKind.DEPRECATED_SYNTAX: FlextUtilitiesBeartypeDeprecatedVisitor.v_deprecated_syntax,
    })


__all__: list[str] = ["FlextUtilitiesBeartypeEngine"]
