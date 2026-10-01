"""Runtime enforcement predicate bindings, typed from the predicate package data."""

from __future__ import annotations

from types import MappingProxyType

from ..._constants.enforcement import FlextConstantsEnforcement as c
from ..._models.enforcement import FlextModelsEnforcement as me
from ..._models.pydantic import FlextModelsPydantic as mp
from ..._typings.base import FlextTypingBase as t

PREDICATE_BINDINGS: t.MappingKV[
    str, tuple[c.EnforcementPredicateKind, mp.BaseModel]
] = MappingProxyType({
    tag: (spec.predicate, spec.params)
    for tag, spec in (
        (tag, me.EnforcementPredicateSpec.model_validate(raw))
        for tag, raw in c.ENFORCEMENT_PREDICATE_SPECS.items()
    )
})
"""Runtime tag → (predicate kind, typed parameters); one data row per rule."""


__all__: list[str] = ["PREDICATE_BINDINGS"]
