"""MRO_SHAPE redundant-inner + self-ref analysis sidecar.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, cast

from flext_core._models import FlextModelsEnforcement
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype import FlextUtilitiesBeartypeHelpers
from flext_core._utilities._beartype._class_visitor.visitor_base import NO_VIOLATION

if TYPE_CHECKING:
    from flext_core import t


def redundant_inner_violation(
    target: type,
    alias_violation: FlextTypingBase.StrMapping | None,
    *,
    forbid_redundant_inner: bool,
) -> FlextTypingBase.StrMapping | None:
    """Compute the redundant-inner-namespace violation.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    outer_name, separator, _ = target.__qualname__.partition(".")
    has_only_dunder_attrs = all(
        key.startswith("__") and key.endswith("__") for key in vars(target)
    )
    has_redundant_inner = all((
        alias_violation is None,
        forbid_redundant_inner,
        bool(separator),
        getattr(target.__bases__[0], "__qualname__", "") == outer_name,
        has_only_dunder_attrs,
    ))
    return {"class": target.__qualname__} if has_redundant_inner else NO_VIOLATION


def self_ref_violation(
    target: type,
    violation: FlextTypingBase.StrMapping | None,
    params: FlextModelsEnforcement.MroShapeParams,
) -> FlextTypingBase.StrMapping | None:
    """Compute the utilities.py self-root import violation.

    Returns:
        The resulting ``t.StrMapping | None``.

    """
    if violation is not None or not params.require_explicit_class_when_self_ref:
        return NO_VIOLATION
    maybe_module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
    if maybe_module is None:
        return NO_VIOLATION

    module = maybe_module
    src_file = str(getattr(module, "__file__", "") or "")
    normalized_values: list[FunctionType] = []
    members = cast("dict[str, t.ModuleGlobalValue]", vars(target))
    # Snapshot the values: reading a member can execute a lazy facade whose
    # module import binds more class attributes, mutating this very mapping
    # mid-iteration (RuntimeError: dictionary changed size during iteration).
    for name, value in tuple(members.items()):
        if name.startswith("__pydantic_"):
            continue
        function = getattr(value, "__func__", value)
        if type(function) is FunctionType:
            normalized_values.append(function)
    first_name = getattr(target.__bases__[0], "__name__", "")
    base_count = len(target.__bases__)
    min_multi_parent = 2
    return (
        {"class": target.__name__, "first_base": "u"}
        if all((
            violation is None,
            src_file.endswith("utilities.py"),
            not getattr(target, "__flext_pattern_b__", False),
            base_count >= min_multi_parent,
            first_name == "u",
        ))
        and any("u" in value.__code__.co_names for value in normalized_values)
        else NO_VIOLATION
    )
