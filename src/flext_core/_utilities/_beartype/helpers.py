"""Type and module introspection helpers — annotation inspection + bytecode analysis."""

from __future__ import annotations

import dis
import functools
import inspect
import types as _types_mod
from collections.abc import (
    Callable,
    Iterator,
    MutableMapping,
    MutableSequence,
    MutableSet,
)
from types import UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeHelpers:
    """Annotation + bytecode inspection helpers."""

    @staticmethod
    def unwrap_type_alias(
        hint: t.TypeHintSpecifier | None,
    ) -> t.TypeHintSpecifier | None:
        current = hint
        seen: set[int] = set()
        while isinstance(current, TypeAliasType):
            current_id = id(current)
            if current_id in seen:
                return current
            seen.add(current_id)
            current = current.__value__
        return current

    @staticmethod
    def contains_any_recursive(
        hint: t.TypeHintSpecifier | None,
        *,
        seen: set[int],
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        hint = h.unwrap_type_alias(hint)
        if hint is None:
            return False
        hint_id = id(hint)
        if hint_id in seen:
            return False
        seen.add(hint_id)
        if hint is Any or hint is object:
            return True
        return any(
            h.contains_any_recursive(child, seen=seen) for child in get_args(hint)
        )

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None,
        forbidden: frozenset[str],
    ) -> tuple[bool, str]:
        h = FlextUtilitiesBeartypeHelpers
        hint = h.unwrap_type_alias(hint)
        if hint is None:
            return False, ""
        origin = get_origin(hint)
        if origin is None or not hasattr(origin, "__name__"):
            return False, ""
        name: str = origin.__name__
        return (True, name) if name in forbidden else (False, "")

    @staticmethod
    def has_runtime_protocol_marker(value: type) -> bool:
        return bool(getattr(value, "_is_protocol", False))

    @staticmethod
    def has_abstract_contract(value: type) -> bool:
        return bool(getattr(value, "__abstractmethods__", None)) or any(
            getattr(base, "__name__", "") == "ABC" for base in value.__mro__
        )

    @staticmethod
    def has_nested_namespace(value: type) -> bool:
        for base in value.__mro__:
            if base is not object and any(
                isinstance(member, type) and not name.startswith("_")
                for name, member in vars(base).items()
            ):
                return True
        return False

    @staticmethod
    def unwrap_annotated(
        hint: t.TypeHintSpecifier | None,
    ) -> t.TypeHintSpecifier | None:
        h = FlextUtilitiesBeartypeHelpers
        current = hint
        while current is not None:
            current = h.unwrap_type_alias(current)
            match current:
                case ForwardRef():
                    current = current.__forward_arg__
                case str():
                    stripped = current.strip()
                    annotated_arg = next(
                        (
                            stripped.removeprefix(prefix)[:-1]
                            for prefix in (
                                "Annotated[",
                                "typing.Annotated[",
                                "typing_extensions.Annotated[",
                            )
                            if stripped.startswith(prefix) and stripped.endswith("]")
                        ),
                        None,
                    )
                    if annotated_arg is None:
                        return stripped
                    current = h.first_top_level_arg(annotated_arg)
                case _ if get_origin(current) is Annotated:
                    args = get_args(current)
                    current = current if not args else args[0]
                    if not args:
                        return current
                case _:
                    return current
        return current

    @staticmethod
    def first_top_level_arg(annotation_text: str) -> str:
        depth = 0
        for index, char in enumerate(annotation_text):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            elif char == "," and depth == 0:
                return annotation_text[:index].strip()
        return annotation_text.strip()

    @staticmethod
    @functools.cache
    def runtime_module_for(target: type) -> _types_mod.ModuleType | None:
        if "." in target.__qualname__:
            return None
        module = inspect.getmodule(target)
        if module is None or getattr(module, target.__name__, None) is not target:
            return None
        src_file = getattr(module, "__file__", None)
        if not isinstance(src_file, str):
            return None
        return (
            None
            if any(m in src_file for m in c.ENFORCE_NON_WORKSPACE_PATH_MARKERS)
            else module
        )

    @staticmethod
    def runtime_wrapper_module_for(target: type) -> _types_mod.ModuleType | None:
        h = FlextUtilitiesBeartypeHelpers
        module = h.runtime_module_for(target)
        if module is None:
            return None
        src_file = str(getattr(module, "__file__", "") or "")
        normalized = src_file.replace("\\", "/")
        if any(
            s in normalized for s in ("/tests/", "/examples/", "/scripts/")
        ) and not normalized.endswith("/__init__.py"):
            return module
        return None

    @staticmethod
    def iter_module_callables(
        module: _types_mod.ModuleType,
    ) -> Iterator[_types_mod.FunctionType]:
        module_name = module.__name__
        for value in vars(module).values():
            if isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            code = getattr(value, "__code__", None)
            if (
                isinstance(value, _types_mod.FunctionType)
                and isinstance(code, _types_mod.CodeType)
                and getattr(inspect.getmodule(value), "__name__", None) == module_name
            ):
                yield value

    @staticmethod
    def function_param_names(fn: _types_mod.FunctionType) -> tuple[str, ...]:
        code = getattr(fn, "__code__", None)
        return (
            tuple(name for name in code.co_varnames[: code.co_argcount])
            if isinstance(code, _types_mod.CodeType)
            else ()
        )

    @staticmethod
    def is_pass_through_bytecode(
        fn: _types_mod.FunctionType,
        param_names: tuple[str, ...],
    ) -> bool:
        instructions = [
            ins
            for ins in dis.get_instructions(fn)
            if ins.opname not in {"RESUME", "CACHE", "PUSH_NULL", "COPY_FREE_VARS"}
        ]
        if not instructions or instructions[0].opname not in {
            "LOAD_GLOBAL",
            "LOAD_DEREF",
            "LOAD_FAST",
            "LOAD_NAME",
        }:
            return False
        consumed = 1
        for expected_arg in param_names:
            if (
                consumed >= len(instructions)
                or instructions[consumed].opname != "LOAD_FAST"
                or instructions[consumed].argval != expected_arg
            ):
                return False
            consumed += 1
        return (
            consumed + 1 < len(instructions)
            and instructions[consumed].opname in {"CALL", "CALL_FUNCTION"}
            and instructions[consumed + 1].opname == "RETURN_VALUE"
        )

    @staticmethod
    def has_call_to_global(
        fn: _types_mod.FunctionType,
        target_name: str,
    ) -> dis.Instruction | None:
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval == target_name:
                return ins
        return None

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
    ) -> tuple[str, str] | None:
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

    # --- Canonical module/object introspection helpers ---

    @staticmethod
    def module_filename_for(module: _types_mod.ModuleType) -> str | None:
        filename = getattr(module, "__file__", None)
        return filename if isinstance(filename, str) else None

    @staticmethod
    def object_module_for(obj: object) -> _types_mod.ModuleType | None:
        module = inspect.getmodule(obj)
        return module if isinstance(module, _types_mod.ModuleType) else None

    @staticmethod
    def object_module_name_for(obj: object) -> str | None:
        module = inspect.getmodule(obj)
        name = getattr(module, "__name__", None)
        return name if isinstance(name, str) else None

    # --- Type-shape helpers moved from beartype_engine so visitors
    #     can import helpers top-level without cyclic imports ---

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
    def mutable_kind(value: object) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def mutable_default_factory_kind(
        factory: type | Callable[..., object] | None,
    ) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: t.TypeHintSpecifier | None,
        factory: type | Callable[..., object] | None,
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
                    f"typing.{en}[",
                    f"collections.abc.{en}[",
                ))
            )
        org = get_origin(norm)
        tgt = org or norm
        return tgt is exp

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )
