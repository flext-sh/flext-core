"""Field + model annotation governance via Pydantic inspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import inspect
from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin

from pydantic.fields import FieldInfo

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)


class FlextUtilitiesBeartypeFieldVisitor:
    """FIELD_SHAPE + MODEL_CONFIG visitors via Pydantic introspection."""

    @staticmethod
    def _ast_union_members(node: ast.expr) -> tuple[ast.expr, ...]:
        """Return only the top-level members written in a union expression.

        Returns:
            Only the top-level members written in a union expression.

        """
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return (
                *FlextUtilitiesBeartypeFieldVisitor._ast_union_members(node.left),
                *FlextUtilitiesBeartypeFieldVisitor._ast_union_members(node.right),
            )
        if isinstance(node, ast.Subscript) and (
            (isinstance(node.value, ast.Name) and node.value.id == "Union")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "Union")
        ):
            return (
                tuple(node.slice.elts)
                if isinstance(node.slice, ast.Tuple)
                else (node.slice,)
            )
        return (node,)

    @classmethod
    def _declared_union_members_from_string(cls, declared: str) -> int:
        """Count union arms in a string annotation via its parsed syntax tree.

        Returns:
            The resulting ``int``.

        """
        unwrapped = _ubh.unwrap_annotated(declared)
        if not isinstance(unwrapped, str):
            return 0
        try:
            expression = ast.parse(unwrapped, mode="eval").body
        except SyntaxError:
            expression = None
        if expression is None:
            return 0
        members = cls._ast_union_members(expression)
        if len(members) == 1:
            return 0
        return sum(
            1
            for member in members
            if not (isinstance(member, ast.Name) and member.id == "None")
            and not (isinstance(member, ast.Constant) and member.value is None)
        )

    @classmethod
    def _declared_union_members(cls, annotation: object | None) -> int:
        """Count union arms from declaration syntax without expanding aliases.

        Returns:
            The resulting ``int``.

        """
        declared = annotation
        if isinstance(declared, str):
            return cls._declared_union_members_from_string(declared)
        if get_origin(declared) is Annotated:
            args = get_args(declared)
            declared = args[0] if args else declared
        if isinstance(declared, TypeAliasType):
            return 0
        if get_origin(declared) not in {UnionType, Union}:
            return 0
        return sum(1 for member in get_args(declared) if member is not type(None))

    @staticmethod
    def _field_description_violation(
        model_type: type,
        name: str,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        # Cheap evidence first: class-wide annotation evaluation runs once per
        # field, so it is reached only when no description is declared plainly.
        if name.startswith("_") or info.description:
            return None
        raw_annotations = vars(model_type).get("__annotations__", {})
        raw_annotation = raw_annotations.get(name)
        if isinstance(raw_annotation, str) and "description=" in raw_annotation:
            return None
        resolved_annotation = inspect.get_annotations(model_type, eval_str=False).get(
            name,
        )
        if isinstance(resolved_annotation, str):
            try:
                resolved = inspect.get_annotations(model_type, eval_str=True).get(name)
            except (NameError, TypeError):
                resolved = None
            else:
                resolved_annotation = resolved
        has_annotated_description = False
        if get_origin(resolved_annotation) is Annotated:
            has_annotated_description = any(
                isinstance(meta, FieldInfo) and meta.description
                for meta in get_args(resolved_annotation)[1:]
            )
        return None if has_annotated_description else {}

    @staticmethod
    def _field_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
        *,
        declared_annotation: object | None = None,
    ) -> t.StrMapping | None:
        visitor = FlextUtilitiesBeartypeFieldVisitor
        violation = visitor._any_violation(params, info)
        if violation is None:
            violation = visitor._collection_violation(params, info)
        if violation is None:
            violation = visitor._mutable_default_violation(params, info)
        if violation is None:
            violation = visitor._default_factory_violation(params, info)
        if violation is None:
            violation = visitor._str_none_empty_violation(params, info)
        if violation is None:
            violation = visitor._inline_union_violation(params, declared_annotation)
        return violation

    @staticmethod
    def _any_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Report the forbid-any violation when the flag and annotation match.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if params.forbid_any and _ubh.contains_any_recursive(
            info.annotation,
            seen=set(),
        ):
            return {}
        return None

    @staticmethod
    def _collection_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Report the bare-collection violation with its replacement hint.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_bare_collection:
            return None
        bad, origin = _ubh.has_forbidden_collection_origin(
            info.annotation,
            c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS,
        )
        if not bad:
            return None
        replacement = next(
            (
                repl
                for key, repl in c.ENFORCEMENT_FORBIDDEN_COLLECTIONS.items()
                if key.__name__ == origin
            ),
            origin,
        )
        return {"kind": origin, "replacement": replacement}

    @staticmethod
    def _mutable_default_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Report the mutable-default violation when the flag and default match.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_mutable_default:
            return None
        mutable_kind = _ubh.mutable_kind(info.default)
        if mutable_kind is not None and info.default:
            return {"kind": mutable_kind}
        return None

    @staticmethod
    def _default_factory_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Report the raw default-factory violation when the shapes match.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not (
            params.forbid_raw_default_factory
            and info.default_factory is not None
            and not _ubh.allows_mutable_default_factory(
                info.annotation,
                info.default_factory,
            )
        ):
            return None
        factory_kind = _ubh.mutable_default_factory_kind(info.default_factory)
        if factory_kind is not None:
            return {"kind": factory_kind.__name__}
        return None

    @staticmethod
    def _str_none_empty_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        """Report the str|None empty-default violation when the shapes match.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if (
            params.forbid_str_none_empty
            and _ubh.matches_str_none_union(info.annotation)
            and isinstance(info.default, str)
            and not info.default
        ):
            return {}
        return None

    @staticmethod
    def _inline_union_violation(
        params: me.FieldShapeParams,
        declared_annotation: object | None,
    ) -> t.StrMapping | None:
        """Report the declared-union arm-count violation when it exceeds the cap.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_inline_union:
            return None
        inline_union_arms = FlextUtilitiesBeartypeFieldVisitor._declared_union_members(
            declared_annotation,
        )
        if inline_union_arms > params.max_union_arms:
            return {"arms": str(inline_union_arms)}
        return None

    @classmethod
    def v_field_shape(
        cls: type[FlextUtilitiesBeartypeFieldVisitor],
        params: me.FieldShapeParams,
        *args: type | str | FieldInfo,
    ) -> t.StrMapping | None:
        """FIELD_SHAPE — Pydantic field annotation governance via flags.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        match args:
            case (model_type, name, info):
                if not (
                    isinstance(model_type, type)
                    and isinstance(name, str)
                    and isinstance(info, FieldInfo)
                ):
                    return None
                if params.require_description:
                    return cls._field_description_violation(model_type, name, info)
                declared_annotation = (
                    vars(model_type).get("__annotations__", {}).get(name)
                )
                return cls._field_violation(
                    params,
                    info,
                    declared_annotation=declared_annotation,
                )
            case (info,):
                if not isinstance(info, FieldInfo):
                    return None
                return cls._field_violation(params, info)
            case _:
                return None

    @staticmethod
    def v_model_config(
        params: me.ModelConfigParams,
        target: type,
    ) -> t.StrMapping | None:
        """MODEL_CONFIG — Pydantic model_config governance via flags.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        violation: t.StrMapping | None = None
        has_v1_config = params.forbid_v1_config and isinstance(
            target.__dict__.get("Config"),
            type,
        )
        if has_v1_config:
            violation = {}
        elif issubclass(target, mp.BaseModel) and not _ubh.has_relaxed_extra_base(
            target,
        ):
            extra = target.model_config.get("extra")
            local = target.__dict__.get("model_config", {})
            if params.require_extra_forbid and extra is None:
                violation = {}
            elif (
                params.allowed_extra_values
                and extra not in {None, "forbid", *params.allowed_extra_values}
                and "extra" in local
            ):
                violation = {"extra": str(extra)}
            elif (
                params.require_frozen_for_value_objects
                and any(
                    b.__name__ in c.ENFORCEMENT_VALUE_OBJECT_BASES
                    for b in target.__mro__
                )
                and not target.model_config.get("frozen", False)
            ):
                violation = {}
        return violation
