"""MRO_SHAPE redundant-inner + self-ref analysis sidecar."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_01 import (
    NO_VIOLATION,
)
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers as ubh

if TYPE_CHECKING:
    from flext_core._protocols.enforcement import FlextProtocolsEnforcement as pe
    from flext_core._typings.base import FlextTypingBase as t


def redundant_inner_violation(
    target: type, alias_violation: t.StrMapping | None, *, forbid_redundant_inner: bool
) -> t.StrMapping | None:
    """Compute the redundant-inner-namespace violation."""
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
    target: type, violation: t.StrMapping | None, params: pe.MroShapeParams
) -> t.StrMapping | None:
    """Compute the utilities.py self-root import violation."""
    if violation is not None or not params.require_explicit_class_when_self_ref:
        return NO_VIOLATION
    maybe_module = ubh.runtime_module_for(target)
    if maybe_module is None:
        return NO_VIOLATION

    module = maybe_module
    src_file = str(getattr(module, "__file__", "") or "")
    normalized_values = (
        value.__func__ if isinstance(value, (classmethod, staticmethod)) else value
        for value in vars(target).values()
    )
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
        and any(
            (code := getattr(value, "__code__", None)) is not None
            and "u" in code.co_names
            for value in normalized_values
            if callable(value)
        )
        else NO_VIOLATION
    )
