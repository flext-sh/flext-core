"""Class-attribute governance — constants, aliases, TypeAdapters.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, get_origin

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}

_CONSTANT_LITERAL_TYPES: t.VariadicTuple[type] = (
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
)
"""Scalar literal types accepted as canonical constant values."""

_CONSTANT_CONTAINER_TYPES: t.VariadicTuple[type] = (
    frozenset,
    tuple,
    dict,
    list,
    set,
    MappingProxyType,
    Path,
)
"""Container/call types accepted as canonical constant values."""


class FlextUtilitiesBeartypeAttrVisitor:
    """ATTR_SHAPE — class-attribute shape governance."""

    @staticmethod
    def v_attr_shape(
        params: me.AttrShapeParams,
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        """ATTR_SHAPE — class-attribute governance (constants / aliases / TypeAdapters).

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if params.forbid_mutable_value:
            mk = _ubh.mutable_kind(value)
            if mk is not None:
                return {"kind": mk}
        if params.require_uppercase_name and name != name.upper():
            return _BARE_VIOLATION
        if params.forbid_any_in_alias and _ubh.alias_contains_any(
            _ubh.resolve_type_alias_value(value),
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
        """Return True when a class attribute value looks like a constant.

        Returns:
            True when a class attribute value looks like a constant.

        """
        if value is None:
            return True
        return isinstance(value, _CONSTANT_LITERAL_TYPES + _CONSTANT_CONTAINER_TYPES)

    @staticmethod
    def _has_classvar_annotation(target: type, name: str) -> bool:
        """Return True when ``name`` is annotated as ClassVar on ``target``.

        Returns:
            True when ``name`` is annotated as ClassVar on ``target``.

        """
        annotations_map: dict[str, object]
        try:
            annotations_map = inspect.get_annotations(target, eval_str=True)
        except (NameError, AttributeError, TypeError):
            annotations_map = {}
        ann = annotations_map.get(name)
        if ann is not None:
            return ann is ClassVar or get_origin(ann) is ClassVar
        # String annotations whose defining module is unavailable (synthetic
        # test classes) are read unevaluated; any failure escapes with its cause.
        raw = inspect.get_annotations(target, eval_str=False).get(name)
        return isinstance(raw, str) and (
            raw.startswith("ClassVar") or "ClassVar[" in raw
        )

    @staticmethod
    def _is_implicit_constant(
        params: me.ClassVarConstantParams,
        target: type,
        name: str,
        value: object,
    ) -> bool:
        """Return True for UPPER_CASE attributes lacking ``ClassVar``.

        Such attributes look like constants but are not declared as
        ``ClassVar``.

        Returns:
            True when an UPPER_CASE attribute looks like a constant but lacks ClassVar.

        """
        if not params.detect_implicit_constants:
            return False
        if FlextUtilitiesBeartypeAttrVisitor._has_classvar_annotation(target, name):
            return False
        return FlextUtilitiesBeartypeAttrVisitor._is_constant_value(value)

    @staticmethod
    def v_classvar_constant(
        params: me.ClassVarConstantParams,
        target: type,
        name: str,
        value: object,
    ) -> t.StrMapping | None:
        """CLASSVAR_CONSTANT — flag one constant declared outside _constants.

        Granularity belongs to the iterator (one item per public attribute,
        see ``_ns_classvar_constants``); this visitor judges exactly one
        attribute per call, per the one-detail-per-call engine contract.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        module_name = getattr(target, "__module__", "") or ""
        if module_name.endswith("._constants") or "._constants." in module_name:
            return None
        if name in c.ENFORCEMENT_CLASSVAR_EXEMPT_NAMES:
            return None
        has_classvar = FlextUtilitiesBeartypeAttrVisitor._has_classvar_annotation(
            target,
            name,
        )
        is_implicit = (
            not has_classvar
            and FlextUtilitiesBeartypeAttrVisitor._is_implicit_constant(
                params,
                target,
                name,
                value,
            )
        )
        if not (has_classvar or is_implicit):
            return None
        if not FlextUtilitiesBeartypeAttrVisitor._is_constant_value(value):
            return None
        project = module_name.split(".", 1)[0] or "project"
        return {
            "name": name,
            "module": module_name.rsplit(".", 1)[-1],
            "class_name": getattr(target, "__qualname__", "<class>"),
            "full_module": module_name,
            "implicit": str(is_implicit),
            "suggested_name": name,
            "suggested_target": f"{project}._constants",
        }
