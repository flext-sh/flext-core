"""Deprecated syntax detection via bytecode + module introspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.regex import FlextConstantsRegex as cre
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_NO_VIOLATION: t.StrMapping | None = None
_TYPING_TYPE_ALIAS = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.


class FlextUtilitiesBeartypeDeprecatedVisitor:
    """DEPRECATED_SYNTAX visitor via runtime introspection."""

    @classmethod
    def v_deprecated_syntax(
        cls,
        params: me.DeprecatedSyntaxParams,
        target: type,
    ) -> t.StrMapping | None:
        """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``.

        Returns:
            The resulting ``t.StrMapping | None``.

        Raises:
            ValueError: If unknown deprecated-syntax shape.

        """
        shape = params.ast_shape
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        file_name = Path(src_file).name
        violation = _NO_VIOLATION
        match shape:
            case "AnnAssign[TypeAlias]":
                violation = cls._shape_annassign_type_alias(module, file_name)
            case "cast_outside_core":
                violation = cls._shape_cast_outside_core(module, src_file, file_name)
            case "no_core_tests_namespace":
                violation = cls._shape_no_core_tests_namespace(target)
            case "no_wrapper_root_alias_import":
                violation = cls._shape_no_wrapper_root_alias_import(target)
            case _:
                msg = f"unknown deprecated-syntax shape {shape!r}"
                raise ValueError(msg)
        return violation

    @classmethod
    def _shape_annassign_type_alias(
        cls,
        module: ModuleType,
        file_name: str,
    ) -> t.StrMapping | None:
        """Flag a typing.TypeAlias declaration in the runtime module.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        try:
            has_type_alias = any(
                annotation is _TYPING_TYPE_ALIAS
                for annotation in inspect.get_annotations(
                    module,
                    eval_str=False,
                ).values()
            )
        except (TypeError, NameError):
            has_type_alias = False
        return {"file": file_name, "line": "?"} if has_type_alias else _NO_VIOLATION

    @classmethod
    def _shape_cast_outside_core(
        cls,
        module: ModuleType,
        src_file: str,
        file_name: str,
    ) -> t.StrMapping | None:
        """Flag a cast() call in a module outside the core path markers.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if any(marker in src_file for marker in c.ENFORCE_FLEXT_CORE_PATH_MARKERS):
            return _NO_VIOLATION
        cast_target = c.EnforceAstHookSymbol.CAST_CALL.value
        return next(
            (
                {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
                for fn in _ubh.iter_module_callables(module)
                if _ubh.has_call_to_global(fn, cast_target) is not None
            ),
            _NO_VIOLATION,
        )

    @classmethod
    def _shape_no_core_tests_namespace(cls, target: type) -> t.StrMapping | None:
        """Flag a wrapper root alias that still declares Core.Tests.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        wrapper_module = _ubh.runtime_wrapper_module_for(target)
        if wrapper_module is None:
            return _NO_VIOLATION
        wrapper_file_name = Path(
            _ubh.module_filename_for(wrapper_module) or "",
        ).name
        return next(
            (
                {
                    "symbol": f"{alias_name}.Core.Tests",
                    "file": wrapper_file_name,
                    "line": "<runtime>",
                }
                for alias_name in _ubh.runtime_alias_names(
                    wrapper_module.__name__.split(".", 1)[0],
                )
                if (alias_value := getattr(wrapper_module, alias_name, None))
                is not None
                and (core := getattr(alias_value, "Core", None)) is not None
                and hasattr(core, "Tests")
            ),
            _NO_VIOLATION,
        )

    @classmethod
    def _shape_no_wrapper_root_alias_import(cls, target: type) -> t.StrMapping | None:
        """Flag a wrapper facade import that bypasses the root alias.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        wrapper_module = _ubh.runtime_wrapper_module_for(target)
        if wrapper_module is None:
            return _NO_VIOLATION
        wrapper_file_name = Path(
            _ubh.module_filename_for(wrapper_module) or "",
        ).name
        package_name = wrapper_module.__name__.split(".", 1)[0]
        wrapper_submodules = _ubh.facade_module_names(package_name)
        wrapper_file = _ubh.module_filename_for(wrapper_module)
        source = (
            Path(wrapper_file).read_text(encoding="utf-8")
            if wrapper_file is not None
            else ""
        )
        # A module without a source file has no text to scan; a
        # declared source that cannot be read raises.
        if source:
            return next(
                (
                    {
                        "file": wrapper_file_name,
                        "line": str(
                            source.count("\n", 0, match.start()) + 1,
                        ),
                        "statement": (
                            f"from {match.group(1)}.{match.group(2)} "
                            f"import {first_alias}"
                        ),
                    }
                    for match in cre.FORBIDDEN_FACADE_IMPORT_RE.finditer(
                        source,
                    )
                    for first_alias in (
                        [n.strip() for n in match.group(3).split(",") if n.strip()][:1]
                    )
                ),
                _NO_VIOLATION,
            )
        return next(
            (
                {
                    "file": wrapper_file_name,
                    "line": "<runtime>",
                    "statement": f"from {origin} import {alias_name}",
                }
                for alias_name in _ubh.runtime_alias_names(package_name)
                if (
                    alias_value := getattr(
                        wrapper_module,
                        alias_name,
                        None,
                    )
                )
                is not None
                and (origin := _ubh.object_module_name_for(alias_value) or "")
                for parent, _, child in (origin.partition("."),)
                if parent in {"tests", "examples", "scripts"}
                and child in wrapper_submodules
            ),
            _NO_VIOLATION,
        )
