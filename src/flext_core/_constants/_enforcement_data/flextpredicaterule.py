"""Runtime enforcement predicate model keyed by tag.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel

from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_04 import (
    FlextConstantsEnforcementRules,
)
from flext_core._typings.base import FlextTypingBase


class FlextPredicateRule(BaseModel):
    """One runtime enforcement predicate keyed by tag."""

    category: FlextConstantsEnforcementRules.EnforcementCategory
    collect: str = ""
    recursive: bool = False
    layer: str = ""
    problem: str
    fix: str
    predicate: str
    params: FlextTypingBase.JsonDict


__all__: list[str] = ["FlextPredicateRule"]
