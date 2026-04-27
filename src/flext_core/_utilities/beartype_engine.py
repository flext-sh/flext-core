"""Enforcement rule dispatcher — predicate routing via data-driven MRO visitors.

Engine combines visitor mixins for field + model, attributes, methods, classes,
modules, imports, and deprecated syntax checks. Helper methods live in
FlextUtilitiesBeartypeHelpers; visitors in domain-specific mixin classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, MutableSet
from types import MappingProxyType, UnionType
from typing import ClassVar, Union, get_args, get_origin, no_type_check

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
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers
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
    FlextUtilitiesBeartypeFieldVisitor,
    FlextUtilitiesBeartypeAttrVisitor,
    FlextUtilitiesBeartypeMethodVisitor,
    FlextUtilitiesBeartypeClassVisitor,
    FlextUtilitiesBeartypeModuleVisitor,
    FlextUtilitiesBeartypeImportVisitor,
    FlextUtilitiesBeartypeDeprecatedVisitor,
):
    """Annotation inspection + per-tag rule predicates via data-driven visitors."""

    unwrap_type_alias = staticmethod(FlextUtilitiesBeartypeHelpers.unwrap_type_alias)
    unwrap_annotated = staticmethod(FlextUtilitiesBeartypeHelpers.unwrap_annotated)
    runtime_module_for = staticmethod(FlextUtilitiesBeartypeHelpers.runtime_module_for)
    runtime_wrapper_module_for = staticmethod(FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for)
    iter_module_callables = staticmethod(FlextUtilitiesBeartypeHelpers.iter_module_callables)
    function_param_names = staticmethod(FlextUtilitiesBeartypeHelpers.function_param_names)
    is_pass_through_bytecode = staticmethod(FlextUtilitiesBeartypeHelpers.is_pass_through_bytecode)
    has_call_to_global = staticmethod(FlextUtilitiesBeartypeHelpers.has_call_to_global)
    has_attribute_call = staticmethod(FlextUtilitiesBeartypeHelpers.has_attribute_call)
    has_private_attr_probe = staticmethod(FlextUtilitiesBeartypeHelpers.has_private_attr_probe)
    has_forbidden_collection_origin = staticmethod(FlextUtilitiesBeartypeHelpers.has_forbidden_collection_origin)

    @staticmethod
    def contains_any(hint: t.TypeHintSpecifier | None) -> bool:
        return FlextUtilitiesBeartypeHelpers.contains_any_recursive(hint, seen=set())

    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        h = FlextUtilitiesBeartypeEngine
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(h2) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: t.TypeHintSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeEngine
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return False
        return str in (a := get_args(h2)) and type(None) in a

    @staticmethod
    def alias_contains_any(alias_value: t.TypeHintSpecifier | None) -> bool:
        try:
            return FlextUtilitiesBeartypeEngine.contains_any(alias_value)
        except (TypeError, AttributeError, RuntimeError, RecursionError):  # noqa: BLE001
            return "Any" in str(alias_value)

    @staticmethod
    def mutable_kind(value: object) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return str(kind.__name__)
        return None

    @staticmethod
    def mutable_default_factory_kind(factory: type | Callable[..., object] | None) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: t.TypeHintSpecifier | None, factory: type | Callable[..., object] | None
    ) -> bool:
        h = FlextUtilitiesBeartypeEngine
        expected_by = {list: MutableSequence, dict: MutableMapping, set: MutableSet}
        mk = h.mutable_default_factory_kind(factory)
        if mk is None:
            return False
        exp = expected_by.get(mk)
        norm = h.unwrap_annotated(hint)
        if norm is None:
            return False
        if isinstance(norm, str):
            en = exp.__name__ if exp else ""
            return bool(en) and (norm == en or norm.startswith((f"{en}[", f"typing.{en}[", f"collections.abc.{en}[")))
        org = get_origin(norm)
        tgt = org or norm
        return exp is not None and tgt is exp

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__)

    @staticmethod
    def has_runtime_protocol_marker(value: type) -> bool:
        return bool(getattr(value, "_is_protocol", False))

    @staticmethod
    def has_abstract_contract(value: type) -> bool:
        return bool(getattr(value, "__abstractmethods__", None)) or any(getattr(b, "__name__", "") == "ABC" for b in value.__mro__)

    @staticmethod
    def has_nested_namespace(value: type) -> bool:
        for base in value.__mro__:
            if base is not object and any(isinstance(v, type) and not n.startswith("_") for n, v in vars(base).items()):
                return True
        return False

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
        return name not in c.ENFORCEMENT_UTILITIES_EXEMPT_METHODS and not name.startswith("_")

    @classmethod
    def apply(
        cls, kind: c.EnforcementPredicateKind, params: mp.BaseModel, *args: object
    ) -> t.StrMapping | None:
        """Dispatch a rule predicate to its visitor by ``predicate_kind``."""
        visitor = cls._VISITORS.get(kind)
        return _NO_VIOLATION if visitor is None else visitor(params, *args)

    # noqa: SLF001 - private visitor methods are intentional; part of data-driven dispatch contract
    _VISITORS: ClassVar[Mapping[c.EnforcementPredicateKind, Callable[..., t.StrMapping | None]]] = (
        MappingProxyType({
            c.EnforcementPredicateKind.FIELD_SHAPE: FlextUtilitiesBeartypeFieldVisitor._v_field_shape,
            c.EnforcementPredicateKind.MODEL_CONFIG: FlextUtilitiesBeartypeFieldVisitor._v_model_config,
            c.EnforcementPredicateKind.ATTR_SHAPE: FlextUtilitiesBeartypeAttrVisitor._v_attr_shape,
            c.EnforcementPredicateKind.METHOD_SHAPE: FlextUtilitiesBeartypeMethodVisitor._v_method_shape,
            c.EnforcementPredicateKind.CLASS_PLACEMENT: FlextUtilitiesBeartypeClassVisitor._v_class_placement,
            c.EnforcementPredicateKind.PROTOCOL_TREE: FlextUtilitiesBeartypeClassVisitor._v_protocol_tree,
            c.EnforcementPredicateKind.MRO_SHAPE: FlextUtilitiesBeartypeClassVisitor._v_mro_shape,
            c.EnforcementPredicateKind.LOOSE_SYMBOL: FlextUtilitiesBeartypeClassVisitor._v_loose_symbol,
            c.EnforcementPredicateKind.WRAPPER: FlextUtilitiesBeartypeDeprecatedVisitor._v_wrapper,
            c.EnforcementPredicateKind.IMPORT_BLACKLIST: FlextUtilitiesBeartypeImportVisitor._v_import_blacklist,
            c.EnforcementPredicateKind.ALIAS_REBIND: FlextUtilitiesBeartypeImportVisitor._v_alias_rebind,
            c.EnforcementPredicateKind.LIBRARY_IMPORT: FlextUtilitiesBeartypeImportVisitor._v_library_import,
            c.EnforcementPredicateKind.LOC_CAP: FlextUtilitiesBeartypeModuleVisitor._v_loc_cap,
            c.EnforcementPredicateKind.DUPLICATE_SYMBOL: FlextUtilitiesBeartypeModuleVisitor._v_duplicate_symbol,
            c.EnforcementPredicateKind.DEPRECATED_SYNTAX: FlextUtilitiesBeartypeDeprecatedVisitor._v_deprecated_syntax,
        })
    )


ube = FlextUtilitiesBeartypeEngine
__all__: list[str] = ["FlextUtilitiesBeartypeEngine", "ube"]
