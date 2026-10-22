"""Type and module introspection helpers — annotation inspection + bytecode analysis.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import types as _types_mod
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from types import UnionType
from typing import TYPE_CHECKING, ForwardRef, TypeAliasType, Union, get_args, get_origin

from flext_core import c
from flext_core._utilities._beartype import FlextUtilitiesBeartypeTypeAliases
from flext_core._utilities._beartype._helpers_parts import helpers_part_02

# Import directly from base modules to avoid a circular load through the public
# flext_core facade while this module is still being initialized.

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsBase
    from flext_core._typings import FlextTypingBase


class FlextUtilitiesBeartypeHelpersPart03(
    helpers_part_02.FlextUtilitiesBeartypeHelpersPart02
):
    @staticmethod
    def module_filename_for(module: _types_mod.ModuleType) -> str | None:
        filename = getattr(module, "__file__", None)
        return filename if isinstance(filename, str) else None

    @staticmethod
    def object_module_for(
        obj: FlextProtocolsBase.AttributeProbe,
    ) -> _types_mod.ModuleType | None:
        module = inspect.getmodule(obj)
        return module if isinstance(module, _types_mod.ModuleType) else None

    @staticmethod
    def object_module_name_for(obj: FlextProtocolsBase.AttributeProbe) -> str | None:
        module = inspect.getmodule(obj)
        name = getattr(module, "__name__", None)
        return name if isinstance(name, str) else None

    @staticmethod
    def count_union_members(hint: FlextTypingBase.TypeHintSpecifier | None) -> int:
        h = FlextUtilitiesBeartypeHelpersPart03
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(h2) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: FlextTypingBase.TypeFormSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeHelpersPart03
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return False
        return str in (a := get_args(h2)) and type(None) in a

    @staticmethod
    def alias_contains_any(
        alias_value: FlextTypingBase.TypeFormSpecifier | None,
        *,
        owner: _types_mod.ModuleType | type | None = None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpersPart03
        return h.contains_any_recursive(alias_value, seen=set(), owner=owner)

    @staticmethod
    def mutable_kind(value: FlextProtocolsBase.AttributeProbe) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def mutable_default_factory_kind(
        factory: type | Callable[..., FlextProtocolsBase.AttributeProbe] | None,
    ) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: FlextTypingBase.TypeFormSpecifier | None,
        factory: type | Callable[..., FlextProtocolsBase.AttributeProbe] | None,
        *,
        owner: _types_mod.ModuleType | type | None = None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpersPart03
        mk = h.mutable_default_factory_kind(factory)
        if mk is list:
            exp = MutableSequence
        elif mk is dict:
            exp = MutableMapping
        elif mk is set:
            exp = MutableSet
        else:
            return False
        if isinstance(hint, (str, ForwardRef)):
            hint = FlextUtilitiesBeartypeTypeAliases.declared_origin(hint, owner=owner)
        norm = h.unwrap_annotated(hint, owner=owner)
        if norm is None:
            return False
        org = get_origin(norm)
        if isinstance(org, TypeAliasType):
            resolved = h.resolve_type_alias_value(org, owner=owner)
            if resolved is None:
                return False
            org = get_origin(resolved) or resolved
        tgt = org or norm
        return tgt is exp

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )


__all__: list[str] = ["FlextUtilitiesBeartypeHelpersPart03"]
