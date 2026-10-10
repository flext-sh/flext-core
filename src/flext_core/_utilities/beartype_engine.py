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

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsEnforcement, FlextModelsPydantic
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import FlextUtilitiesBeartypeTypeAliases
from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_03 import (
    FlextUtilitiesBeartypeClassVisitor,
)
from flext_core._utilities._beartype._helpers_parts.helpers_part_03 import (
    FlextUtilitiesBeartypeHelpers,
)
from flext_core._utilities._beartype.attr_visitor import (
    FlextUtilitiesBeartypeAttrVisitor,
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
    def attr_accept_constants(
        name: str,
        value: FlextProtocolsBase.AttributeProbe,
    ) -> bool:
        if (
            name.startswith("_")
            or name in FlextConstantsEnforcement.ENFORCEMENT_CONSTANTS_SKIP_ATTRS
        ):
            return False
        if isinstance(value, (type, classmethod, staticmethod, property)):
            return False
        return not callable(value)

    @staticmethod
    def attr_accept_public(name: str) -> bool:
        return not name.startswith("_")

    @staticmethod
    def attr_accept_utility(target: type, name: str) -> bool:
        if name.startswith("_"):
            return False
        if name in FlextConstantsEnforcement.ENFORCEMENT_UTILITIES_EXEMPT_METHODS:
            return False
        return target.__name__ not in (
            FlextConstantsEnforcement.ENFORCEMENT_UTILITIES_STATEFUL_ADAPTERS
        )

    @staticmethod
    def contains_any(hint: FlextTypingBase.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeHelpers.contains_any_recursive(hint, seen=set())

    @staticmethod
    def deferred_aliases(
        params: FlextModelsPydantic.BaseModel,
        owner: type,
        *args: FlextProtocolsBase.AttributeProbe,
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Account for unavailable alias values before a value-dependent rule.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if isinstance(params, FlextModelsEnforcement.AttrShapeParams):
            return FlextUtilitiesBeartypeEngine._attr_shape_deferred(
                params,
                owner,
                args,
            )
        if isinstance(params, FlextModelsEnforcement.FieldShapeParams):
            return FlextUtilitiesBeartypeEngine._field_shape_deferred(
                params,
                owner,
                args,
            )
        return ()

    @staticmethod
    def _attr_shape_deferred(
        params: FlextModelsEnforcement.AttrShapeParams,
        owner: type,
        args: tuple[FlextProtocolsBase.AttributeProbe, ...],
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Compute deferred aliases for attribute-shape params.

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
    def _field_shape_deferred(
        params: FlextModelsEnforcement.FieldShapeParams,
        owner: type,
        args: tuple[FlextProtocolsBase.AttributeProbe, ...],
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Compute deferred aliases for field-shape params.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if not args:
            return ()
        info = args[-1]
        if not isinstance(info, FieldInfo) or params.require_description:
            return ()
        if params.forbid_any or params.forbid_bare_collection:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation,
                recursive=params.forbid_any,
                owner=owner,
            )
        return FlextUtilitiesBeartypeEngine._field_shape_tail(info, params, owner)

    @staticmethod
    def _field_shape_tail(
        info: FieldInfo,
        params: FlextModelsEnforcement.FieldShapeParams,
        owner: type,
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Compute deferred aliases for factory/string-shape field flags.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        if params.forbid_mutable_default or params.forbid_raw_default_factory:
            return ()
        if params.forbid_str_none_empty:
            return FlextUtilitiesBeartypeTypeAliases.deferred(
                info.annotation,
                owner=owner,
            )
        return ()

    @override
    @staticmethod
    def has_forbidden_collection_origin(
        hint: FlextTypingBase.TypeFormSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        return FlextUtilitiesBeartypeHelpers.has_forbidden_collection_origin(
            hint,
            forbidden,
        )

    @classmethod
    def apply(
        cls,
        kind: FlextConstantsEnforcement.EnforcementPredicateKind,
        params: FlextModelsPydantic.BaseModel,
        *args: FlextProtocolsBase.AttributeProbe,
    ) -> FlextTypingBase.StrMapping | None:
        """Dispatch a rule predicate to its visitor; an unmapped kind raises.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        return cls._VISITORS[kind](params, *args)

    _VISITORS: ClassVar[
        FlextTypingBase.MappingKV[
            FlextConstantsEnforcement.EnforcementPredicateKind,
            Callable[..., FlextTypingBase.StrMapping | None],
        ]
    ] = MappingProxyType({
        FlextConstantsEnforcement.EnforcementPredicateKind.FIELD_SHAPE: (
            FlextUtilitiesBeartypeFieldVisitor.v_field_shape
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.MODEL_CONFIG: (
            FlextUtilitiesBeartypeFieldVisitor.v_model_config
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.ATTR_SHAPE: (
            FlextUtilitiesBeartypeAttrVisitor.v_attr_shape
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.CLASSVAR_CONSTANT: (
            FlextUtilitiesBeartypeAttrVisitor.v_classvar_constant
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.METHOD_SHAPE: (
            FlextUtilitiesBeartypeMethodVisitor.v_method_shape
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.CLASS_PLACEMENT: (
            FlextUtilitiesBeartypeClassVisitor.v_class_placement
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.PROTOCOL_TREE: (
            FlextUtilitiesBeartypeClassVisitor.v_protocol_tree
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.MRO_SHAPE: (
            FlextUtilitiesBeartypeClassVisitor.v_mro_shape
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.LOOSE_SYMBOL: (
            FlextUtilitiesBeartypeClassVisitor.v_loose_symbol
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.IMPORT_BLACKLIST: (
            FlextUtilitiesBeartypeImportVisitor.v_import_blacklist
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.ALIAS_REBIND: (
            FlextUtilitiesBeartypeImportVisitor.v_alias_rebind
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.COMPATIBILITY_ALIAS: (
            FlextUtilitiesBeartypeImportVisitor.v_compatibility_alias
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.LIBRARY_IMPORT: (
            FlextUtilitiesBeartypeImportVisitor.v_library_import
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.LOC_CAP: (
            FlextUtilitiesBeartypeModuleVisitor.v_loc_cap
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.MODULE_ALIAS: (
            FlextUtilitiesBeartypeModuleVisitor.v_module_alias
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.DUPLICATE_SYMBOL: (
            FlextUtilitiesBeartypeModuleVisitor.v_duplicate_symbol
        ),
        FlextConstantsEnforcement.EnforcementPredicateKind.DEPRECATED_SYNTAX: (
            FlextUtilitiesBeartypeDeprecatedVisitor.v_deprecated_syntax
        ),
    })


__all__: list[str] = ["FlextUtilitiesBeartypeEngine"]
