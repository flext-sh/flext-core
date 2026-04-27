"""Deprecated syntax detection via bytecode + module introspection."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from beartype._util.module.utilmodget import (
    get_module_filename_or_none,
    get_object_module_name_or_none,
)

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cp
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_typing_TypeAlias = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.


class FlextUtilitiesBeartypeDeprecatedVisitor:
    """DEPRECATED_SYNTAX + WRAPPER visitors via bytecode introspection."""

    @staticmethod
    def v_wrapper(
        params: me.WrapperParams,  # noqa: ARG004 — params reserved for future
        target: type,
    ) -> t.StrMapping | None:
        """WRAPPER — pass-through wrapper detection via bytecode (ENFORCE-043)."""
        from flext_core._utilities.beartype_engine import ube

        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        for fn in ube.iter_module_callables(module):
            param_names = ube.function_param_names(fn)
            if not param_names:
                continue
            if ube.is_pass_through_bytecode(fn, param_names):
                return {"name": fn.__name__, "file": Path(src_file).name}
        return _NO_VIOLATION

    @staticmethod
    def v_deprecated_syntax(
        params: me.DeprecatedSyntaxParams,
        target: type,
    ) -> t.StrMapping | None:
        """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``."""
        import inspect

        from flext_core._utilities.beartype_engine import ube

        shape = params.ast_shape
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        file_name = Path(src_file).name
        if shape == "AnnAssign[TypeAlias]":
            try:
                module_annotations = inspect.get_annotations(module, eval_str=False)
            except (TypeError, NameError):
                return _NO_VIOLATION
            for ann in module_annotations.values():
                if ann is _typing_TypeAlias:
                    return {"file": file_name, "line": "?"}
            return _NO_VIOLATION
        if shape == "cast_outside_core":
            if any(marker in src_file for marker in c.ENFORCE_FLEXT_CORE_PATH_MARKERS):
                return _NO_VIOLATION
            cast_target = c.EnforceAstHookSymbol.CAST_CALL.value
            for fn in ube.iter_module_callables(module):
                hit = ube.has_call_to_global(fn, cast_target)
                if hit is not None:
                    return {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
            return _NO_VIOLATION
        if shape == "model_rebuild_call":
            attr = c.EnforceAstHookSymbol.MODEL_REBUILD_ATTR.value
            for fn in ube.iter_module_callables(module):
                hit = ube.has_attribute_call(fn, attr)
                if hit is not None:
                    return {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
            return _NO_VIOLATION
        if shape == "private_attr_probe":
            probes = c.ENFORCE_PRIVATE_PROBE_BUILTINS
            for fn in ube.iter_module_callables(module):
                hit = ube.has_private_attr_probe(fn, probes)
                if hit is not None:
                    builtin, attr = hit
                    return {"probe": builtin, "name": attr, "file": file_name}
            return _NO_VIOLATION
        if shape == "no_core_tests_namespace":
            wrapper_module = ube.runtime_wrapper_module_for(target)
            if wrapper_module is None:
                return _NO_VIOLATION
            for alias_name in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(wrapper_module, alias_name, None)
                if alias_value is None:
                    continue
                core = getattr(alias_value, "Core", None)
                if core is None:
                    continue
                if hasattr(core, "Tests"):
                    return {
                        "symbol": f"{alias_name}.Core.Tests",
                        "file": Path(
                            get_module_filename_or_none(wrapper_module) or ""
                        ).name,
                        "line": "<runtime>",
                    }
            return _NO_VIOLATION
        if shape == "no_wrapper_root_alias_import":
            wrapper_module = ube.runtime_wrapper_module_for(target)
            if wrapper_module is None:
                return _NO_VIOLATION
            wrapper_submodules = cp.FACADE_MODULE_NAMES
            for alias_name in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(wrapper_module, alias_name, None)
                if alias_value is None:
                    continue
                origin = get_object_module_name_or_none(alias_value) or ""
                parent, dot, child = origin.partition(".")
                if (
                    dot
                    and parent in {"tests", "examples", "scripts"}
                    and child in wrapper_submodules
                ):
                    return {
                        "file": Path(
                            get_module_filename_or_none(wrapper_module) or ""
                        ).name,
                        "line": "<runtime>",
                        "statement": f"from {origin} import {alias_name}",
                    }
            return _NO_VIOLATION
        return _NO_VIOLATION
