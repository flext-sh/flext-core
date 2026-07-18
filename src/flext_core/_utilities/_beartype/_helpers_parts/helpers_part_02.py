"""Type and module introspection helpers — annotation inspection + bytecode analysis."""

from __future__ import annotations

import dis
import inspect
import types as _types_mod
from typing import Annotated, ClassVar, ForwardRef, get_args, get_origin

from flext_core._constants.enforcement import FlextConstantsEnforcement as c

from .helpers_part_01 import (
    FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpersPart01,
)

from collections.abc import Iterator
from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeHelpers(FlextUtilitiesBeartypeHelpersPart01):
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

    _RUNTIME_MODULE_CACHE: ClassVar[dict[type, _types_mod.ModuleType]] = {}

    @staticmethod
    def runtime_module_for(target: type) -> _types_mod.ModuleType | None:
        """Resolve the defining module once the class is bound to it.

        Misses are NOT cached: during ``__init_subclass__`` the class is not
        yet assigned to its module namespace, so a cached miss would poison
        every later post-import check for the same class.
        """
        cached = FlextUtilitiesBeartypeHelpers._RUNTIME_MODULE_CACHE.get(target)
        if cached is not None:
            return cached
        if "." in target.__qualname__:
            return None
        module = inspect.getmodule(target)
        if module is None or getattr(module, target.__name__, None) is not target:
            return None
        src_file = getattr(module, "__file__", None)
        if not isinstance(src_file, str):
            return None
        if any(m in src_file for m in c.ENFORCE_NON_WORKSPACE_PATH_MARKERS):
            return None
        FlextUtilitiesBeartypeHelpers._RUNTIME_MODULE_CACHE[target] = module
        return module

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
    def function_param_names(fn: _types_mod.FunctionType) -> t.StrSequence:
        code = getattr(fn, "__code__", None)
        return (
            tuple(name for name in code.co_varnames[: code.co_argcount])
            if isinstance(code, _types_mod.CodeType)
            else ()
        )

    @staticmethod
    def is_pass_through_bytecode(
        fn: _types_mod.FunctionType, param_names: t.StrSequence
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
        fn: _types_mod.FunctionType, target_name: str
    ) -> dis.Instruction | None:
        for ins in dis.get_instructions(fn):
            if ins.opname == "LOAD_GLOBAL" and ins.argval == target_name:
                return ins
        return None


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
