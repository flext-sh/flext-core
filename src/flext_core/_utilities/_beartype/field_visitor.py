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

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsEnforcement, FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import FlextUtilitiesBeartypeHelpers


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
    def _declared_union_members(cls, annotation: object | None) -> int:
        """Count union arms from declaration syntax without expanding aliases.

        Returns:
            The resulting ``int``.

        """
        declared = annotation
        if isinstance(declared, str):
            return cls._string_union_members(declared)
        if get_origin(declared) is Annotated:
            args = get_args(declared)
            declared = args[0] if args else declared
        if isinstance(declared, TypeAliasType):
            return 0
        if get_origin(declared) not in {UnionType, Union}:
            return 0
        return sum(1 for member in get_args(declared) if member is not type(None))

    @classmethod
    def _string_union_members(cls, declared: str) -> int:
        """Count union arms inside a string annotation.

        Returns:
            The resulting ``int``.

        """
        unwrapped = FlextUtilitiesBeartypeHelpers.unwrap_annotated(declared)
        if not isinstance(unwrapped, str):
            return 0
        try:
            expression = ast.parse(unwrapped, mode="eval").body
        except SyntaxError:
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

    @staticmethod
    def _field_description_violation(
        model_type: type,
        name: str,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
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
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
        *,
        declared_annotation: object | None = None,
    ) -> FlextTypingBase.StrMapping | None:
        """Run field-shape checks in canonical order and return the first hit.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        checks = (
            FlextUtilitiesBeartypeFieldVisitor._check_forbid_any,
            FlextUtilitiesBeartypeFieldVisitor._check_bare_collection,
            FlextUtilitiesBeartypeFieldVisitor._check_mutable_default,
            FlextUtilitiesBeartypeFieldVisitor._check_raw_default_factory,
            FlextUtilitiesBeartypeFieldVisitor._check_str_none_empty,
        )
        for check in checks:
            violation = check(params, info)
            if violation is not None:
                return violation
        return FlextUtilitiesBeartypeFieldVisitor._check_inline_union(
            params,
            declared_annotation,
        )

    @staticmethod
    def _check_forbid_any(
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_any flag against the field annotation.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if params.forbid_any and FlextUtilitiesBeartypeHelpers.contains_any_recursive(
            info.annotation,
            seen=set(),
        ):
            return {}
        return None

    @staticmethod
    def _check_bare_collection(
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_bare_collection flag against the field annotation.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_bare_collection:
            return None
        bad, origin = FlextUtilitiesBeartypeHelpers.has_forbidden_collection_origin(
            info.annotation,
            FlextConstantsEnforcement.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS,
        )
        if not bad:
            return None
        forbidden = FlextConstantsEnforcement.ENFORCEMENT_FORBIDDEN_COLLECTIONS
        replacement = next(
            (repl for key, repl in forbidden.items() if key.__name__ == origin),
            origin,
        )
        return {"kind": origin, "replacement": replacement}

    @staticmethod
    def _check_mutable_default(
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_mutable_default flag against the field default.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_mutable_default:
            return None
        mutable_kind = FlextUtilitiesBeartypeHelpers.mutable_kind(info.default)
        if mutable_kind is not None and info.default:
            return {"kind": mutable_kind}
        return None

    @staticmethod
    def _check_raw_default_factory(
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_raw_default_factory flag against the factory.

        A collection field's empty default is ``u.empty`` of the field's own
        contract; a raw collection constructor, bare or specialized, is a
        violation. The contract's agreement with the annotation is proven
        statically by the type checkers, so this check never reads it.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_raw_default_factory:
            return None
        kind = FlextUtilitiesBeartypeHelpers.raw_collection_factory_kind(
            info.default_factory,
        )
        return None if kind is None else {"kind": kind.__name__}

    @staticmethod
    def _check_str_none_empty(
        params: FlextModelsEnforcement.FieldShapeParams,
        info: FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_str_none_empty flag against the field default.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if (
            params.forbid_str_none_empty
            and FlextUtilitiesBeartypeHelpers.matches_str_none_union(info.annotation)
            and isinstance(info.default, str)
            and not info.default
        ):
            return {}
        return None

    @classmethod
    def _check_inline_union(
        cls,
        params: FlextModelsEnforcement.FieldShapeParams,
        declared_annotation: object | None,
    ) -> FlextTypingBase.StrMapping | None:
        """Check the forbid_inline_union flag against the declared annotation.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.forbid_inline_union:
            return None
        inline_union_arms = cls._declared_union_members(declared_annotation)
        if inline_union_arms > params.max_union_arms:
            return {"arms": str(inline_union_arms)}
        return None

    @classmethod
    def v_field_shape(
        cls: type[FlextUtilitiesBeartypeFieldVisitor],
        params: FlextModelsEnforcement.FieldShapeParams,
        *args: type | str | FieldInfo,
    ) -> FlextTypingBase.StrMapping | None:
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
        params: FlextModelsEnforcement.ModelConfigParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
        """MODEL_CONFIG — Pydantic model_config governance via flags.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        violation: FlextTypingBase.StrMapping | None = None
        has_v1_config = params.forbid_v1_config and isinstance(
            target.__dict__.get("Config"),
            type,
        )
        if has_v1_config:
            violation = {}
        elif issubclass(
            target,
            FlextModelsPydantic.BaseModel,
        ) and not FlextUtilitiesBeartypeHelpers.has_relaxed_extra_base(
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
                    b.__name__
                    in FlextConstantsEnforcement.ENFORCEMENT_VALUE_OBJECT_BASES
                    for b in target.__mro__
                )
                and not target.model_config.get("frozen", False)
            ):
                violation = {}
        return violation
