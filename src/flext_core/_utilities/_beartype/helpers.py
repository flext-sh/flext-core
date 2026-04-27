"""Type and module introspection helpers — annotation inspection + bytecode analysis."""

from __future__ import annotations

import dis
import functools
import types as _types_mod
from collections.abc import Callable, Iterator
from typing import Annotated, ForwardRef, TypeAliasType, get_args, get_origin

from beartype._util.func.utilfunccodeobj import (
    get_func_code_object_or_none,
)
from beartype._util.func.utilfunctest import is_func_python
from beartype._util.module.utilmodget import (
    get_module_filename_or_none,
    get_object_module_name_or_none,
    get_object_module_or_none,
)

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from typing import Any


class FlextUtilitiesBeartypeHelpers:
    """Annotation + bytecode inspection helpers."""

    @staticmethod
    def unwrap_type_alias(hint: t.TypeHintSpecifier | None) -> t.TypeHintSpecifier | None:
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
        return any(h.contains_any_recursive(child, seen=seen) for child in get_args(hint))

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
    def unwrap_annotated(hint: t.TypeHintSpecifier | None) -> t.TypeHintSpecifier | None:
        h = FlextUtilitiesBeartypeHelpers
        current = hint
        while current is not None:
            current = h.unwrap_type_alias(current)
            if isinstance(current, ForwardRef):
                current = current.__forward_arg__
                continue
            if isinstance(current, str):
                stripped = current.strip()
                for prefix in ("Annotated[", "typing.Annotated[", "typing_extensions.Annotated["):
                    if stripped.startswith(prefix) and stripped.endswith("]"):
                        current = h._first_top_level_arg(stripped.removeprefix(prefix)[:-1])
                        break
                else:
                    return stripped
                continue
            if get_origin(current) is not Annotated:
                return current
            args = get_args(current)
            current = args[0] if args else current
            if not args:
                return current
        return current

    @staticmethod
    def _first_top_level_arg(annotation_text: str) -> str:
        depth = 0
        for index, char in enumerate(annotation_text):
            if char in "[":
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
        module = get_object_module_or_none(target)
        if module is None or getattr(module, target.__name__, None) is not target:
            return None
        src_file = get_module_filename_or_none(module)
        if src_file is None:
            return None
        return None if any(m in src_file for m in c.ENFORCE_NON_WORKSPACE_PATH_MARKERS) else module

    @staticmethod
    def runtime_wrapper_module_for(target: type) -> _types_mod.ModuleType | None:
        h = FlextUtilitiesBeartypeHelpers
        module = h.runtime_module_for(target)
        if module is None:
            return None
        src_file = get_module_filename_or_none(module) or ""
        normalized = src_file.replace("\\", "/")
        if any(s in normalized for s in ("/tests/", "/examples/", "/scripts/")) and not normalized.endswith("/__init__.py"):
            return module
        return None

    @staticmethod
    def iter_module_callables(module: _types_mod.ModuleType) -> Iterator[Callable[..., tp.JsonValue]]:
        module_name = module.__name__
        for value in vars(module).values():
            if isinstance(value, (classmethod, staticmethod)):
                value = value.__func__
            if is_func_python(value) and get_object_module_name_or_none(value) == module_name:
                yield value

    @staticmethod
    def function_param_names(fn: Callable[..., tp.JsonValue]) -> tuple[str, ...]:
        code = get_func_code_object_or_none(fn)
        return tuple(str(name) for name in code.co_varnames[: code.co_argcount]) if code else ()

    @staticmethod
    def is_pass_through_bytecode(fn: Callable[..., tp.JsonValue], param_names: tuple[str, ...]) -> bool:
        instructions = [
            ins
            for ins in dis.get_instructions(fn)
            if ins.opname not in {"RESUME", "CACHE", "PUSH_NULL", "COPY_FREE_VARS"}
        ]
        if not instructions or instructions[0].opname not in ("LOAD_GLOBAL", "LOAD_DEREF", "LOAD_FAST", "LOAD_NAME"):
            return False
        consumed = 1
        for expected_arg in param_names:
            if consumed >= len(instructions) or instructions[consumed].opname != "LOAD_FAST" or instructions[consumed].argval != expected_arg:
                return False
            consumed += 1
        return consumed + 1 < len(instructions) and instructions[consumed].opname in {"CALL", "CALL_FUNCTION"} and instructions[consumed + 1].opname == "RETURN_VALUE"

    @staticmethod
    def has_call_to_global(fn: Callable[..., tp.JsonValue], target_name: str) -> dis.Instruction | None:
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval == target_name:
                return ins
        return None

    @staticmethod
    def has_attribute_call(fn: Callable[..., tp.JsonValue], attr_name: str) -> dis.Instruction | None:
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_ATTR" and ins.argval == attr_name:
                return ins
        return None

    @staticmethod
    def has_private_attr_probe(
        fn: Callable[..., tp.JsonValue],
        builtins_set: frozenset[str],
    ) -> tuple[str, str] | None:
        last_builtin: str | None = None
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval in builtins_set:
                last_builtin = ins.argval
            elif ins.opname == "LOAD_CONST" and last_builtin is not None:
                value = ins.argval
                if isinstance(value, str) and value.startswith("_") and not value.startswith("__"):
                    return last_builtin, value
            elif ins.opname in {"CALL", "CALL_FUNCTION"}:
                last_builtin = None
        return None
