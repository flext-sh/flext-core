"""Deprecated syntax detection via bytecode + module introspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from flext_core._constants import FlextConstantsEnforcement, FlextConstantsRegex
from flext_core._models import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import FlextUtilitiesBeartypeHelpers

if TYPE_CHECKING:
    from types import ModuleType

_NO_VIOLATION: FlextTypingBase.StrMapping | None = None
_TYPING_TYPE_ALIAS = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.


def _wrapper_file_name(target: type) -> str | None:
    """Resolve the wrapper module file name for ``target``, if any.

    Returns:
        The resulting ``str | None``.

    """
    wrapper_module = FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target)
    if wrapper_module is None:
        return None
    return Path(
        FlextUtilitiesBeartypeHelpers.module_filename_for(wrapper_module) or "",
    ).name


def _v_type_alias(
    module: ModuleType,
    file_name: str,
) -> FlextTypingBase.StrMapping | None:
    """Detect ``X: TypeAlias = Y`` declarations on the module.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    try:
        has_type_alias = any(
            annotation is _TYPING_TYPE_ALIAS
            for annotation in inspect.get_annotations(module, eval_str=False).values()
        )
    except (TypeError, NameError):
        has_type_alias = False
    if has_type_alias:
        return {"file": file_name, "line": "?"}
    return _NO_VIOLATION


def _v_cast_outside_core(
    module: ModuleType,
    src_file: str,
    file_name: str,
) -> FlextTypingBase.StrMapping | None:
    """Detect ``cast(...)`` calls in modules outside the core source tree.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    if any(
        marker in src_file
        for marker in FlextConstantsEnforcement.ENFORCE_FLEXT_CORE_PATH_MARKERS
    ):
        return _NO_VIOLATION
    cast_target = FlextConstantsEnforcement.EnforceAstHookSymbol.CAST_CALL.value
    return next(
        (
            {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
            for fn in FlextUtilitiesBeartypeHelpers.iter_module_callables(module)
            if FlextUtilitiesBeartypeHelpers.has_call_to_global(fn, cast_target)
            is not None
        ),
        _NO_VIOLATION,
    )


def _v_no_core_tests_namespace(
    target: type,
) -> FlextTypingBase.StrMapping | None:
    """Detect ``<alias>.Core.Tests`` wrapper access for a non-wrapper package.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    wrapper_module = FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target)
    wrapper_file_name = _wrapper_file_name(target)
    if wrapper_module is None or wrapper_file_name is None:
        return _NO_VIOLATION
    return next(
        (
            {
                "symbol": f"{alias_name}.Core.Tests",
                "file": wrapper_file_name,
                "line": "<runtime>",
            }
            for alias_name in FlextUtilitiesBeartypeHelpers.runtime_alias_names(
                wrapper_module.__name__.split(".", 1)[0],
            )
            if (alias_value := getattr(wrapper_module, alias_name, None)) is not None
            and (core := getattr(alias_value, "Core", None)) is not None
            and hasattr(core, "Tests")
        ),
        _NO_VIOLATION,
    )


def _v_wrapper_import_regex(
    wrapper_module: ModuleType | None,
    wrapper_file_name: str,
) -> FlextTypingBase.StrMapping | None:
    """Scan wrapper source text for forbidden facade-import statements.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    # A module without a source file has no text to scan; a
    # declared source that cannot be read raises.
    wrapper_file = (
        FlextUtilitiesBeartypeHelpers.module_filename_for(wrapper_module)
        if wrapper_module is not None
        else None
    )
    source = (
        Path(wrapper_file).read_text(encoding="utf-8")
        if wrapper_file is not None
        else ""
    )
    if not source:
        return _NO_VIOLATION
    return next(
        (
            {
                "file": wrapper_file_name,
                "line": str(source.count("\n", 0, match.start()) + 1),
                "statement": (
                    f"from {match.group(1)}.{match.group(2)} import {first_alias}"
                ),
            }
            for match in FlextConstantsRegex.FORBIDDEN_FACADE_IMPORT_RE.finditer(
                source,
            )
            for first_alias in (
                [n.strip() for n in match.group(3).split(",") if n.strip()][:1]
            )
        ),
        _NO_VIOLATION,
    )


def _v_wrapper_runtime_origin(
    target: type,
) -> FlextTypingBase.StrMapping | None:
    """Detect wrapper alias attributes whose origin is a tests/examples/scripts module.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    wrapper_module = FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target)
    wrapper_file_name = _wrapper_file_name(target)
    if wrapper_module is None or wrapper_file_name is None:
        return _NO_VIOLATION
    package_name = wrapper_module.__name__.split(".", 1)[0]
    wrapper_submodules = FlextUtilitiesBeartypeHelpers.facade_module_names(package_name)
    return next(
        (
            {
                "file": wrapper_file_name,
                "line": "<runtime>",
                "statement": f"from {origin} import {alias_name}",
            }
            for alias_name in FlextUtilitiesBeartypeHelpers.runtime_alias_names(
                package_name,
            )
            if (alias_value := getattr(wrapper_module, alias_name, None)) is not None
            and (
                origin := FlextUtilitiesBeartypeHelpers.object_module_name_for(
                    alias_value,
                )
                or ""
            )
            for parent, _, child in (origin.partition("."),)
            if parent in {"tests", "examples", "scripts"}
            and child in wrapper_submodules
        ),
        _NO_VIOLATION,
    )


def _v_no_wrapper_root_alias_import(
    target: type,
) -> FlextTypingBase.StrMapping | None:
    """Detect forbidden facade imports or tests-origin aliases in the wrapper.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    regex_violation = _v_wrapper_import_regex(
        FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target),
        _wrapper_file_name(target) or "",
    )
    if regex_violation is not _NO_VIOLATION:
        return regex_violation
    return _v_wrapper_runtime_origin(target)


class FlextUtilitiesBeartypeDeprecatedVisitor:
    """DEPRECATED_SYNTAX visitor via runtime introspection."""

    @staticmethod
    def v_deprecated_syntax(
        params: FlextModelsEnforcement.DeprecatedSyntaxParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``.

        Returns:
            The resulting ``t.StrMapping | None``.

        Raises:
            ValueError: If unknown deprecated-syntax shape.

        """
        shape = params.ast_shape
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = FlextUtilitiesBeartypeHelpers.module_filename_for(module) or ""
        file_name = Path(src_file).name
        handlers: FlextTypingBase.MappingKV[
            str,
            Callable[[], FlextTypingBase.StrMapping | None],
        ] = {
            "AnnAssign[TypeAlias]": lambda: _v_type_alias(module, file_name),
            "cast_outside_core": lambda: _v_cast_outside_core(
                module,
                src_file,
                file_name,
            ),
            "no_core_tests_namespace": lambda: _v_no_core_tests_namespace(target),
            "no_wrapper_root_alias_import": lambda: _v_no_wrapper_root_alias_import(
                target,
            ),
        }
        handler = handlers.get(shape)
        if handler is None:
            msg = f"unknown deprecated-syntax shape {shape!r}"
            raise ValueError(msg)
        return handler()


__all__: list[str] = ["FlextUtilitiesBeartypeDeprecatedVisitor"]
