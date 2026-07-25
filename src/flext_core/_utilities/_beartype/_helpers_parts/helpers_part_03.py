"""Type and module introspection helpers — annotation inspection + bytecode analysis."""

from __future__ import annotations

import dis
import inspect
import types as _types_mod
from collections.abc import (
    Callable,
    MutableMapping,
    MutableSequence,
    MutableSet,
)
from types import UnionType
from typing import (
    TYPE_CHECKING,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

# Import directly from base modules to avoid a circular load through the public
# flext_core facade while this module is still being initialized.
from flext_core._constants.enforcement import FlextConstantsEnforcement as c

from .helpers_part_02 import (
    FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpersPart02,
)

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase as p
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeHelpers(FlextUtilitiesBeartypeHelpersPart02):
    @staticmethod
    def has_attribute_call(
        fn: _types_mod.FunctionType,
        attr_name: str,
    ) -> dis.Instruction | None:
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_ATTR" and ins.argval == attr_name:
                return ins
        return None

    @staticmethod
    def has_private_attr_probe(
        fn: _types_mod.FunctionType,
        builtins_set: frozenset[str],
    ) -> t.StrPair | None:
        last_builtin: str | None = None
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval in builtins_set:
                last_builtin = ins.argval
            elif ins.opname == "LOAD_CONST" and last_builtin is not None:
                value = ins.argval
                if (
                    isinstance(value, str)
                    and value.startswith("_")
                    and not value.startswith("__")
                ):
                    return last_builtin, value
            elif ins.opname in {"CALL", "CALL_FUNCTION"}:
                last_builtin = None
        return None

    @staticmethod
    def module_filename_for(module: _types_mod.ModuleType) -> str | None:
        filename = getattr(module, "__file__", None)
        return filename if isinstance(filename, str) else None

    @staticmethod
    def object_module_for(obj: p.AttributeProbe) -> _types_mod.ModuleType | None:
        module = inspect.getmodule(obj)
        return module if isinstance(module, _types_mod.ModuleType) else None

    @staticmethod
    def object_module_name_for(obj: p.AttributeProbe) -> str | None:
        module = inspect.getmodule(obj)
        name = getattr(module, "__name__", None)
        return name if isinstance(name, str) else None

    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(h2) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: t.TypeHintSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return False
        return str in (a := get_args(h2)) and type(None) in a

    @staticmethod
    def alias_contains_any(alias_value: t.TypeHintSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        try:
            return h.contains_any_recursive(alias_value, seen=set())
        except (TypeError, AttributeError, RuntimeError, RecursionError):
            return "Any" in str(alias_value)

    @staticmethod
    def mutable_kind(value: p.AttributeProbe) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def mutable_default_factory_kind(
        factory: type | Callable[..., p.AttributeProbe] | None,
    ) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: t.TypeHintSpecifier | None,
        factory: type | Callable[..., p.AttributeProbe] | None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        mk = h.mutable_default_factory_kind(factory)
        if mk is list:
            exp = MutableSequence
        elif mk is dict:
            exp = MutableMapping
        elif mk is set:
            exp = MutableSet
        else:
            return False
        norm = h.unwrap_annotated(hint)
        if norm is None:
            return False
        if isinstance(norm, str):
            en = exp.__name__
            return bool(en) and (
                norm == en
                or norm.startswith((
                    f"{en}[",
                    f"{en}Of[",
                    f"typing.{en}[",
                    f"typing.{en}Of[",
                    f"collections.abc.{en}[",
                    f"t.{en}Of[",
                ))
            )
        org = get_origin(norm)
        if isinstance(org, TypeAliasType):
            org = get_origin(org.__value__) or org.__value__
        tgt = org or norm
        return tgt is exp

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
