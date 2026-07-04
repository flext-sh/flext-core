"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._parts.class_visitor_part_02_01 import alias_first_violation
from ._parts.class_visitor_part_02_02 import (
    redundant_inner_violation,
    self_ref_violation,
)
from .class_visitor_part_01 import (
    NO_VIOLATION,
    FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitorPart01,
)

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeClassVisitor(FlextUtilitiesBeartypeClassVisitorPart01):
    @staticmethod
    def v_mro_shape(
        params: me.MroShapeParams,
        target: type,
    ) -> t.StrMapping | None:
        """MRO_SHAPE — facade base ordering and inner-namespace redundancy."""
        if not target.__bases__:
            return NO_VIOLATION

        alias_violation = alias_first_violation(target, params)
        redundant_inner = redundant_inner_violation(
            target,
            alias_violation,
            forbid_redundant_inner=params.forbid_redundant_inner,
        )
        violation = alias_violation or redundant_inner
        self_ref = self_ref_violation(target, violation, params)
        return violation or self_ref


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
