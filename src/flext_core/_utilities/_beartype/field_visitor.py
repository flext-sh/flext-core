"""Field + model annotation governance via Pydantic inspection."""

from __future__ import annotations

import inspect
from typing import Annotated, get_args, get_origin

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
    def _field_description_violation(
        model_type: type,
        name: str,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        raw_annotations = vars(model_type).get("__annotations__", {})
        raw_annotation = raw_annotations.get(name)
        resolved_annotation = inspect.get_annotations(model_type, eval_str=False).get(
            name
        )
        has_annotated_description = False
        if get_origin(resolved_annotation) is Annotated:
            has_annotated_description = any(
                isinstance(meta, FieldInfo) and meta.description
                for meta in get_args(resolved_annotation)[1:]
            )
        has_description = any((
            name.startswith("_"),
            bool(info.description),
            isinstance(raw_annotation, str) and "description=" in raw_annotation,
            has_annotated_description,
        ))
        return None if has_description else {}

    @staticmethod
    def _field_violation(
        params: me.FieldShapeParams,
        info: FieldInfo,
    ) -> t.StrMapping | None:
        violation: t.StrMapping | None = None
        if params.forbid_any and _ubh.contains_any_recursive(
            info.annotation, seen=set()
        ):
            violation = {}
        elif params.forbid_bare_collection:
            bad, origin = _ubh.has_forbidden_collection_origin(
                info.annotation, c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS
            )
            if bad:
                replacement = next(
                    (
                        repl
                        for key, repl in c.ENFORCEMENT_FORBIDDEN_COLLECTIONS.items()
                        if key.__name__ == origin
                    ),
                    origin,
                )
                violation = {"kind": origin, "replacement": replacement}
        elif params.forbid_mutable_default:
            mutable_kind = _ubh.mutable_kind(info.default)
            if mutable_kind is not None and info.default:
                violation = {"kind": mutable_kind}
        elif (
            params.forbid_raw_default_factory
            and info.default_factory is not None
            and not _ubh.allows_mutable_default_factory(
                info.annotation, info.default_factory
            )
        ):
            factory_kind = _ubh.mutable_default_factory_kind(info.default_factory)
            if factory_kind is not None:
                violation = {"kind": factory_kind.__name__}
        elif (
            params.forbid_str_none_empty
            and _ubh.matches_str_none_union(info.annotation)
            and isinstance(info.default, str)
            and not info.default
        ):
            violation = {}
        elif params.forbid_inline_union:
            inline_union_arms = _ubh.count_union_members(info.annotation)
            if inline_union_arms > params.max_union_arms:
                violation = {"arms": str(inline_union_arms)}
        return violation

    @classmethod
    def v_field_shape(
        cls: type[FlextUtilitiesBeartypeFieldVisitor],
        params: me.FieldShapeParams,
        *args: type | str | FieldInfo,
    ) -> t.StrMapping | None:
        """FIELD_SHAPE — Pydantic field annotation governance via flags."""
        match args:
            case (model_type, name, info) if params.require_description:
                if not (
                    isinstance(model_type, type)
                    and isinstance(name, str)
                    and isinstance(info, FieldInfo)
                ):
                    return None
                return cls._field_description_violation(model_type, name, info)
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
        """MODEL_CONFIG — Pydantic model_config governance via flags."""
        violation: t.StrMapping | None = None
        has_v1_config = params.forbid_v1_config and isinstance(
            target.__dict__.get("Config"), type
        )
        if has_v1_config:
            violation = {}
        elif issubclass(target, mp.BaseModel) and not _ubh.has_relaxed_extra_base(
            target
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
