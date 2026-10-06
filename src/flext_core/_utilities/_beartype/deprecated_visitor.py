"""Deprecated syntax detection via bytecode + module introspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TypeAlias

from flext_core._constants.enforcement import FlextConstantsEnforcement
from flext_core._constants.regex import FlextConstantsRegex
from flext_core._models.enforcement import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers

_NO_VIOLATION: FlextTypingBase.StrMapping | None = None
_TYPING_TYPE_ALIAS = TypeAlias  # sentinel for ``X: TypeAlias = Y`` annotation match.


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
        violation = _NO_VIOLATION
        match shape:
            case "AnnAssign[TypeAlias]":
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
                if has_type_alias:
                    violation = {"file": file_name, "line": "?"}
            case "cast_outside_core":
                if not any(
                    marker in src_file
                    for marker in FlextConstantsEnforcement.ENFORCE_FLEXT_CORE_PATH_MARKERS
                ):
                    cast_target = (
                        FlextConstantsEnforcement.EnforceAstHookSymbol.CAST_CALL.value
                    )
                    violation = next(
                        (
                            {"file": file_name, "line": str(fn.__code__.co_firstlineno)}
                            for fn in FlextUtilitiesBeartypeHelpers.iter_module_callables(
                                module,
                            )
                            if FlextUtilitiesBeartypeHelpers.has_call_to_global(
                                fn,
                                cast_target,
                            )
                            is not None
                        ),
                        _NO_VIOLATION,
                    )
            case "no_core_tests_namespace":
                wrapper_module = (
                    FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target)
                )
                if wrapper_module is not None:
                    wrapper_file_name = Path(
                        FlextUtilitiesBeartypeHelpers.module_filename_for(
                            wrapper_module,
                        )
                        or "",
                    ).name
                    violation = next(
                        (
                            {
                                "symbol": f"{alias_name}.Core.Tests",
                                "file": wrapper_file_name,
                                "line": "<runtime>",
                            }
                            for alias_name in FlextUtilitiesBeartypeHelpers.runtime_alias_names(
                                wrapper_module.__name__.split(".", 1)[0],
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
                wrapper_module = (
                    FlextUtilitiesBeartypeHelpers.runtime_wrapper_module_for(target)
                )
                if wrapper_module is not None:
                    wrapper_file_name = Path(
                        FlextUtilitiesBeartypeHelpers.module_filename_for(
                            wrapper_module,
                        )
                        or "",
                    ).name
                    package_name = wrapper_module.__name__.split(".", 1)[0]
                    wrapper_submodules = (
                        FlextUtilitiesBeartypeHelpers.facade_module_names(package_name)
                    )
                    violation = _NO_VIOLATION
                    # A module without a source file has no text to scan; a
                    # declared source that cannot be read raises.
                    wrapper_file = FlextUtilitiesBeartypeHelpers.module_filename_for(
                        wrapper_module,
                    )
                    source = (
                        Path(wrapper_file).read_text(encoding="utf-8")
                        if wrapper_file is not None
                        else ""
                    )
                    if source:
                        violation = next(
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
                                for match in FlextConstantsRegex.FORBIDDEN_FACADE_IMPORT_RE.finditer(
                                    source,
                                )
                                for first_alias in (
                                    [
                                        n.strip()
                                        for n in match.group(3).split(",")
                                        if n.strip()
                                    ][:1]
                                )
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
                                for alias_name in FlextUtilitiesBeartypeHelpers.runtime_alias_names(
                                    package_name,
                                )
                                if (
                                    alias_value := getattr(
                                        wrapper_module,
                                        alias_name,
                                        None,
                                    )
                                )
                                is not None
                                and (
                                    origin
                                    := FlextUtilitiesBeartypeHelpers.object_module_name_for(
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
            case _:
                msg = f"unknown deprecated-syntax shape {shape!r}"
                raise ValueError(msg)
        return violation
