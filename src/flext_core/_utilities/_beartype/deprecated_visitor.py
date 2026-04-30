"""Deprecated syntax detection via bytecode + module introspection."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import TypeAlias

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_NO_VIOLATION: t.StrMapping | None = None
_typing_TypeAlias = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.


class FlextUtilitiesBeartypeDeprecatedVisitor:
    """DEPRECATED_SYNTAX + WRAPPER visitors via bytecode introspection."""

    @staticmethod
    def v_wrapper(
        _params: me.WrapperParams,
        target: type,
    ) -> t.StrMapping | None:
        """WRAPPER — pass-through wrapper detection via bytecode (ENFORCE-043)."""
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        for fn in _ubh.iter_module_callables(module):
            param_names = _ubh.function_param_names(fn)
            if not param_names:
                continue
            if _ubh.is_pass_through_bytecode(fn, param_names):
                return {"name": fn.__name__, "file": Path(src_file).name}
        return _NO_VIOLATION

    @staticmethod
    def v_deprecated_syntax(
        params: me.DeprecatedSyntaxParams,
        target: type,
    ) -> t.StrMapping | None:
        """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``."""
        shape = params.ast_shape
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        file_name = Path(src_file).name
        violation = _NO_VIOLATION
        match shape:
            case "AnnAssign[TypeAlias]":
                try:
                    module_annotations = inspect.get_annotations(module, eval_str=False)
                except (TypeError, NameError):
                    pass
                else:
                    violation = next(
                        (
                            {"file": file_name, "line": "?"}
                            for annotation in module_annotations.values()
                            if annotation is _typing_TypeAlias
                        ),
                        _NO_VIOLATION,
                    )
            case "cast_outside_core":
                if not any(
                    marker in src_file for marker in c.ENFORCE_FLEXT_CORE_PATH_MARKERS
                ):
                    cast_target = c.EnforceAstHookSymbol.CAST_CALL.value
                    violation = next(
                        (
                            {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
                            for fn in _ubh.iter_module_callables(module)
                            if _ubh.has_call_to_global(fn, cast_target) is not None
                        ),
                        _NO_VIOLATION,
                    )
            case "model_rebuild_call":
                attr = c.EnforceAstHookSymbol.MODEL_REBUILD_ATTR.value
                violation = next(
                    (
                        {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
                        for fn in _ubh.iter_module_callables(module)
                        if _ubh.has_attribute_call(fn, attr) is not None
                    ),
                    _NO_VIOLATION,
                )
            case "private_attr_probe":
                probes = c.ENFORCE_PRIVATE_PROBE_BUILTINS
                violation = next(
                    (
                        {"probe": builtin, "name": attr, "file": file_name}
                        for fn in _ubh.iter_module_callables(module)
                        if (hit := _ubh.has_private_attr_probe(fn, probes)) is not None
                        for builtin, attr in (hit,)
                    ),
                    _NO_VIOLATION,
                )
            case "no_core_tests_namespace":
                wrapper_module = _ubh.runtime_wrapper_module_for(target)
                if wrapper_module is not None:
                    wrapper_file_name = Path(
                        _ubh.module_filename_for(wrapper_module) or ""
                    ).name
                    violation = next(
                        (
                            {
                                "symbol": f"{alias_name}.Core.Tests",
                                "file": wrapper_file_name,
                                "line": "<runtime>",
                            }
                            for alias_name in _ubh.runtime_alias_names(
                                wrapper_module.__name__.split(".", 1)[0]
                            )
                            if (
                                alias_value := getattr(wrapper_module, alias_name, None)
                            )
                            is not None
                            and (core := getattr(alias_value, "Core", None)) is not None
                            and hasattr(core, "Tests")
                        ),
                        _NO_VIOLATION,
                    )
            case "no_wrapper_root_alias_import":
                wrapper_module = _ubh.runtime_wrapper_module_for(target)
                if wrapper_module is not None:
                    wrapper_file_name = Path(
                        _ubh.module_filename_for(wrapper_module) or ""
                    ).name
                    package_name = wrapper_module.__name__.split(".", 1)[0]
                    wrapper_submodules = _ubh.facade_module_names(package_name)
                    violation = _NO_VIOLATION
                    try:
                        source = Path(
                            _ubh.module_filename_for(wrapper_module) or ""
                        ).read_text(encoding="utf-8")
                    except OSError:
                        source = ""
                    if source:
                        try:
                            syntax = ast.parse(source)
                        except SyntaxError:
                            syntax = None
                        if syntax is not None:
                            violation = next(
                                (
                                    {
                                        "file": wrapper_file_name,
                                        "line": str(node.lineno),
                                        "statement": f"from {node.module} import {alias.name}",
                                    }
                                    for node in ast.walk(syntax)
                                    if isinstance(node, ast.ImportFrom)
                                    and isinstance(node.module, str)
                                    and node.module.split(".")[0]
                                    in {
                                        "tests",
                                        "examples",
                                        "scripts",
                                    }
                                    and "." in node.module
                                    for alias in node.names
                                ),
                                _NO_VIOLATION,
                            )
                    if violation is _NO_VIOLATION:
                        violation = next(
                            (
                                {
                                    "file": wrapper_file_name,
                                    "line": "<runtime>",
                                    "statement": f"from {origin} import {alias_name}",
                                }
                                for alias_name in _ubh.runtime_alias_names(package_name)
                                if (
                                    alias_value := getattr(
                                        wrapper_module, alias_name, None
                                    )
                                )
                                is not None
                                and (
                                    origin := _ubh.object_module_name_for(alias_value)
                                    or ""
                                )
                                for parent, _, child in (origin.partition("."),)
                                if parent in {"tests", "examples", "scripts"}
                                and child in wrapper_submodules
                            ),
                            _NO_VIOLATION,
                        )
            case _:
                pass
        return violation
