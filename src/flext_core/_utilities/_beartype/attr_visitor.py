"""Class-attribute governance — constants, aliases, TypeAdapters."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, get_origin

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp

from .helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}

_CONSTANT_LITERAL_TYPES: tuple[type, ...] = (
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
)
"""Scalar literal types accepted as canonical constant values."""

_CONSTANT_CONTAINER_TYPES: tuple[type, ...] = (
    frozenset,
    tuple,
    dict,
    MappingProxyType,
    Path,
)
"""Container/call types accepted as canonical constant values."""

_CLASSVAR_EXEMPT_NAMES: frozenset[str] = frozenset({"model_config", "logger"})
"""ClassVar attribute names that are framework idioms and stay in place."""


class FlextUtilitiesBeartypeAttrVisitor:
    """ATTR_SHAPE — class-attribute shape governance."""

    @staticmethod
    def v_attr_shape(
        params: me.AttrShapeParams,
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        """ATTR_SHAPE — class-attribute governance (constants / aliases / TypeAdapters)."""
        if params.forbid_mutable_value:
            mk = _ubh.mutable_kind(value)
            if mk is not None:
                return {"kind": mk}
        if params.require_uppercase_name and name != name.upper():
            return _BARE_VIOLATION
        if params.forbid_any_in_alias and _ubh.alias_contains_any(
            getattr(value, "__value__", None),
        ):
            return _BARE_VIOLATION
        if (
            params.require_typeadapter_naming
            and type(value).__name__ == "TypeAdapter"
            and not (name.startswith("ADAPTER_") or name.upper() == name)
        ):
            return {"name": name, "upper_name": name.upper()}
        return _NO_VIOLATION

    @staticmethod
    def _is_constant_value(value: object) -> bool:
        """Return True when a class attribute value looks like a constant."""
        if value is None:
            return True
        return isinstance(value, _CONSTANT_LITERAL_TYPES + _CONSTANT_CONTAINER_TYPES)

    @staticmethod
    def _has_classvar_annotation(target: type, name: str) -> bool:
        """Return True when ``name`` is annotated as ClassVar on ``target``."""
        try:
            annotations = inspect.get_annotations(target, eval_str=True)
        except Exception:
            annotations: dict[str, object] = {}
        ann = annotations.get(name)
        if ann is ClassVar:
            return True
        if ann is not None:
            return get_origin(ann) is ClassVar
        # Fallback for string annotations whose defining module is unavailable
        # (e.g. synthetic test classes) — keep detection strict but do not fail.
        try:
            raw_annotations = inspect.get_annotations(target, eval_str=False)
        except Exception:
            return False
        raw = raw_annotations.get(name)
        return isinstance(raw, str) and (
            raw.startswith("ClassVar") or "ClassVar[" in raw
        )

    @staticmethod
    def v_classvar_constant(
        params: me.ClassVarConstantParams,
        target: type,
    ) -> t.StrMapping | None:
        """CLASSVAR_CONSTANT — flag ClassVar constants declared outside _constants."""
        _ = params
        module_name = getattr(target, "__module__", "") or ""
        if module_name.endswith("._constants") or "._constants." in module_name:
            return _NO_VIOLATION
        for name, value in vars(target).items():
            if name.startswith("_") or name != name.upper():
                continue
            if name in _CLASSVAR_EXEMPT_NAMES:
                continue
            if not FlextUtilitiesBeartypeAttrVisitor._has_classvar_annotation(
                target,
                name,
            ):
                continue
            if not FlextUtilitiesBeartypeAttrVisitor._is_constant_value(value):
                continue
            return {
                "name": name,
                "module": module_name.rsplit(".", 1)[-1],
            }
        return _NO_VIOLATION
