"""Field + model annotation governance via Pydantic inspection."""

from __future__ import annotations

import inspect
from typing import Annotated, get_args, get_origin

from pydantic.fields import FieldInfo

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_FIELD_DESCRIPTION_ARITY: int = 3


class FlextUtilitiesBeartypeFieldVisitor:
    """FIELD_SHAPE + MODEL_CONFIG visitors via Pydantic introspection."""

    @staticmethod
    def v_field_shape(
        params: me.FieldShapeParams,
        *args: object,
    ) -> t.StrMapping | None:
        """FIELD_SHAPE — Pydantic field annotation governance via flags."""
        # Import here to avoid circular dependency
        from flext_core._utilities.beartype_engine import ube

        if params.require_description and len(args) == _FIELD_DESCRIPTION_ARITY:
            model_type, name, info = args
            if not (
                isinstance(model_type, type)
                and isinstance(name, str)
                and isinstance(info, FieldInfo)
            ):
                return _NO_VIOLATION
            if name.startswith("_") or info.description:
                return _NO_VIOLATION
            raw_ann = vars(model_type).get("__annotations__", {})
            raw = raw_ann.get(name)
            if isinstance(raw, str) and "description=" in raw:
                return _NO_VIOLATION
            resolved = inspect.get_annotations(model_type, eval_str=False)
            ann = resolved.get(name)
            if get_origin(ann) is Annotated:
                for meta in get_args(ann)[1:]:
                    if isinstance(meta, FieldInfo) and meta.description:
                        return _NO_VIOLATION
            return _BARE_VIOLATION
        if len(args) != 1:
            return _NO_VIOLATION
        info = args[0]
        if not isinstance(info, FieldInfo):
            return _NO_VIOLATION
        if params.forbid_any and ube.contains_any(info.annotation):
            return _BARE_VIOLATION
        if params.forbid_bare_collection:
            bad, origin = ube.has_forbidden_collection_origin(
                info.annotation, c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS
            )
            if bad:
                replacement = next(
                    (
                        repl
                        for k, repl in c.ENFORCEMENT_FORBIDDEN_COLLECTIONS.items()
                        if k.__name__ == origin
                    ),
                    origin,
                )
                return {"kind": origin, "replacement": replacement}
        if params.forbid_mutable_default:
            mk = ube.mutable_kind(info.default)
            if mk is not None and info.default:
                return {"kind": mk}
        if (
            params.forbid_raw_default_factory
            and info.default_factory is not None
            and not ube.allows_mutable_default_factory(
                info.annotation, info.default_factory
            )
        ):
            fk = ube.mutable_default_factory_kind(info.default_factory)
            if fk is not None:
                return {"kind": fk.__name__}
        if (
            params.forbid_str_none_empty
            and ube.matches_str_none_union(info.annotation)
            and isinstance(info.default, str)
            and not info.default
        ):
            return _BARE_VIOLATION
        if params.forbid_inline_union:
            arms = ube.count_union_members(info.annotation)
            if arms > params.max_union_arms:
                return {"arms": str(arms)}
        return _NO_VIOLATION

    @staticmethod
    def v_model_config(
        params: me.ModelConfigParams,
        target: type,
    ) -> t.StrMapping | None:
        """MODEL_CONFIG — Pydantic model_config governance via flags."""
        from flext_core._models.pydantic import FlextModelsPydantic as mp
        from flext_core._utilities.beartype_engine import ube

        if params.forbid_v1_config and isinstance(target.__dict__.get("Config"), type):
            return _BARE_VIOLATION
        if not issubclass(target, mp.BaseModel):
            return _NO_VIOLATION
        if ube.has_relaxed_extra_base(target):
            return _NO_VIOLATION
        extra = target.model_config.get("extra")
        if params.require_extra_forbid and extra is None:
            return _BARE_VIOLATION
        local = target.__dict__.get("model_config", {})
        if (
            params.allowed_extra_values
            and extra not in {None, "forbid", *params.allowed_extra_values}
            and "extra" in local
        ):
            return {"extra": str(extra)}
        if (
            params.require_frozen_for_value_objects
            and any(
                b.__name__ in c.ENFORCEMENT_VALUE_OBJECT_BASES for b in target.__mro__
            )
            and not target.model_config.get("frozen", False)
        ):
            return _BARE_VIOLATION
        return _NO_VIOLATION
