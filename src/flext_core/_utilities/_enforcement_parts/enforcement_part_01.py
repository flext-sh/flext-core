"""Runtime enforcement predicate bindings, typed from the predicate package data.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsEnforcement, FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase

PREDICATE_BINDINGS: FlextTypingBase.MappingKV[
    str,
    tuple[
        FlextConstantsEnforcement.EnforcementPredicateKind,
        FlextModelsPydantic.BaseModel,
    ],
] = MappingProxyType({
    tag: (spec.predicate, spec.params)
    for tag, spec in (
        (tag, FlextModelsEnforcement.EnforcementPredicateSpec.model_validate(raw))
        for tag, raw in FlextConstantsEnforcement.ENFORCEMENT_PREDICATE_SPECS.items()
    )
})
"""Runtime tag → (predicate kind, typed parameters); one data row per rule."""


__all__: list[str] = ["PREDICATE_BINDINGS"]
