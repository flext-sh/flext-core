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

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
    FlextUtilitiesBeartypeHelpers,
)
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
from flext_core._utilities._beartype.import_visitor import (
    FlextUtilitiesBeartypeImportVisitor,
)
from flext_core._utilities._beartype.method_visitor import (
    FlextUtilitiesBeartypeMethodVisitor,
)
from flext_core._utilities._beartype.module_visitor import (
    FlextUtilitiesBeartypeModuleVisitor,
)
from flext_core._utilities._beartype.type_aliases import (
    FlextUtilitiesBeartypeTypeAliases,
)
from flext_core._utilities.beartype_typingext_patch import (
    FlextUtilitiesBeartypeTypingExtPatch as _FlextUtilitiesBeartypeTypingExtPatch,
)

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

    @classmethod
    def deferred_aliases(
        cls,
        params: mp.BaseModel,
        owner: type,
        *args: p.AttributeProbe,
    ) -> tuple[me.DeferredAlias, ...]:
        """Account for unavailable alias values before a value-dependent rule.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if isinstance(params, me.AttrShapeParams):
            return cls._attr_shape_aliases(params, owner, args)
        if not isinstance(params, me.FieldShapeParams) or not args:
            return ()
        info = args[-1]
        if not isinstance(info, FieldInfo) or params.require_description:
            return ()
        return cls._field_shape_aliases(params, owner, info)

    @staticmethod
    def _attr_shape_aliases(
        params: me.AttrShapeParams,
        owner: type,
        args: tuple[p.AttributeProbe, ...],
    ) -> tuple[me.DeferredAlias, ...]:
        """Collect the alias probes required by the attribute-shape params.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if not params.forbid_any_in_alias:
            return ()
        match args:
            case (_, alias) if isinstance(alias, TypeAliasType):
                return FlextUtilitiesBeartypeTypeAliases.deferred(
                    alias,
                    recursive=True,
                    owner=owner,
                )
            case _:
                return ()

    @staticmethod
    def _field_shape_aliases(
        params: me.FieldShapeParams,
        owner: type,
        info: FieldInfo,
    ) -> tuple[me.DeferredAlias, ...]:
        """Collect the alias probes required by the field-shape params.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if params.forbid_any or params.forbid_bare_collection:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation,
                recursive=params.forbid_any,
                owner=owner,
            )
        if params.forbid_mutable_default:
            return ()
        if (
            params.forbid_raw_default_factory
            and FlextUtilitiesBeartypeHelpers.mutable_default_factory_kind(
                info.default_factory,
            )
            is not None
        ):
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation,
                unwrap_annotated=True,
                inspect_origin=True,
                owner=owner,
            )
        if params.forbid_str_none_empty:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation,
                owner=owner,
            )
        return ()

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

    @classmethod
    def apply(
        cls,
        kind: c.EnforcementPredicateKind,
        params: mp.BaseModel,
        *args: p.AttributeProbe,
    ) -> t.StrMapping | None:
        """Dispatch a rule predicate to its visitor; an unmapped kind raises.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return cls._VISITORS[kind](params, *args)

    _VISITORS: ClassVar[
        t.MappingKV[c.EnforcementPredicateKind, Callable[..., t.StrMapping | None]]
    ] = MappingProxyType({
        c.EnforcementPredicateKind.FIELD_SHAPE: (
            FlextUtilitiesBeartypeFieldVisitor.v_field_shape
        ),
        c.EnforcementPredicateKind.MODEL_CONFIG: (
            FlextUtilitiesBeartypeFieldVisitor.v_model_config
        ),
        c.EnforcementPredicateKind.ATTR_SHAPE: (
            FlextUtilitiesBeartypeAttrVisitor.v_attr_shape
        ),
        c.EnforcementPredicateKind.CLASSVAR_CONSTANT: (
            FlextUtilitiesBeartypeAttrVisitor.v_classvar_constant
        ),
        c.EnforcementPredicateKind.METHOD_SHAPE: (
            FlextUtilitiesBeartypeMethodVisitor.v_method_shape
        ),
        c.EnforcementPredicateKind.CLASS_PLACEMENT: (
            FlextUtilitiesBeartypeClassVisitor.v_class_placement
        ),
        c.EnforcementPredicateKind.PROTOCOL_TREE: (
            FlextUtilitiesBeartypeClassVisitor.v_protocol_tree
        ),
        c.EnforcementPredicateKind.MRO_SHAPE: (
            FlextUtilitiesBeartypeClassVisitor.v_mro_shape
        ),
        c.EnforcementPredicateKind.LOOSE_SYMBOL: (
            FlextUtilitiesBeartypeClassVisitor.v_loose_symbol
        ),
        c.EnforcementPredicateKind.IMPORT_BLACKLIST: (
            FlextUtilitiesBeartypeImportVisitor.v_import_blacklist
        ),
        c.EnforcementPredicateKind.ALIAS_REBIND: (
            FlextUtilitiesBeartypeImportVisitor.v_alias_rebind
        ),
        c.EnforcementPredicateKind.COMPATIBILITY_ALIAS: (
            FlextUtilitiesBeartypeImportVisitor.v_compatibility_alias
        ),
        c.EnforcementPredicateKind.LIBRARY_IMPORT: (
            FlextUtilitiesBeartypeImportVisitor.v_library_import
        ),
        c.EnforcementPredicateKind.LOC_CAP: (
            FlextUtilitiesBeartypeModuleVisitor.v_loc_cap
        ),
        c.EnforcementPredicateKind.MODULE_ALIAS: (
            FlextUtilitiesBeartypeModuleVisitor.v_module_alias
        ),
        c.EnforcementPredicateKind.DUPLICATE_SYMBOL: (
            FlextUtilitiesBeartypeModuleVisitor.v_duplicate_symbol
        ),
        c.EnforcementPredicateKind.DEPRECATED_SYNTAX: (
            FlextUtilitiesBeartypeDeprecatedVisitor.v_deprecated_syntax
        ),
    })


__all__: list[str] = ["FlextUtilitiesBeartypeEngine"]
