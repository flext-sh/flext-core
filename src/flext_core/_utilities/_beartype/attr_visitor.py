"""Class-attribute governance — constants, aliases, TypeAdapters."""

from __future__ import annotations

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}


class FlextUtilitiesBeartypeAttrVisitor:
    """ATTR_SHAPE — class-attribute shape governance."""

    @staticmethod
    def _v_attr_shape(
        params: me.AttrShapeParams,
        name: str,
        value: tp.JsonValue,
    ) -> t.StrMapping | None:
        """ATTR_SHAPE — class-attribute governance (constants / aliases / TypeAdapters)."""
        from flext_core._utilities.beartype_engine import ube

        if params.forbid_mutable_value:
            mk = ube.mutable_kind(value)
            if mk is not None:
                return {"kind": mk}
        if params.require_uppercase_name and name != name.upper():
            return _BARE_VIOLATION
        if params.forbid_any_in_alias and ube.alias_contains_any(
            getattr(value, "__value__", None)
        ):
            return _BARE_VIOLATION
        if (
            params.require_typeadapter_naming
            and type(value).__name__ == "TypeAdapter"
            and not (name.startswith("ADAPTER_") or name.upper() == name)
        ):
            return {"name": name, "upper_name": name.upper()}
        return _NO_VIOLATION
