"""Runtime enforcement predicate bindings, typed from the predicate package data.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t

PREDICATE_BINDINGS: t.MappingKV[
    str,
    tuple[c.EnforcementPredicateKind, mp.BaseModel],
] = MappingProxyType({
    tag: (spec.predicate, spec.params)
    for tag, spec in (
        (tag, me.EnforcementPredicateSpec.model_validate(raw))
        for tag, raw in c.ENFORCEMENT_PREDICATE_SPECS.items()
    )
})
"""Runtime tag → (predicate kind, typed parameters); one data row per rule."""


__all__: list[str] = ["PREDICATE_BINDINGS"]
