"""Class placement, MRO, and protocol tree governance.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models import FlextModelsEnforcement
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype._class_visitor import (
    class_visitor_part_01,
    class_visitor_part_02_01,
    class_visitor_part_02_02,
)
from flext_core._utilities._beartype._class_visitor.class_visitor_part_01 import (
    NO_VIOLATION,
)


class FlextUtilitiesBeartypeClassVisitorPart02(
    class_visitor_part_01.FlextUtilitiesBeartypeClassVisitorPart01
):
    @staticmethod
    def v_mro_shape(
        params: FlextModelsEnforcement.MroShapeParams, target: type
    ) -> FlextTypingBase.StrMapping | None:
        """MRO_SHAPE — facade base ordering and inner-namespace redundancy.

        Returns:
            The resulting ``FlextTypingBase.StrMapping | None``.

        """
        if not target.__bases__:
            return NO_VIOLATION

        alias_violation = class_visitor_part_02_01.alias_first_violation(target, params)
        redundant_inner = class_visitor_part_02_02.redundant_inner_violation(
            target,
            alias_violation,
            forbid_redundant_inner=params.forbid_redundant_inner,
        )
        violation = alias_violation or redundant_inner
        self_ref = class_visitor_part_02_02.self_ref_violation(
            target, violation, params
        )
        return violation or self_ref


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitorPart02"]
