"""Catalog source models for enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from ..._typings.base import FlextTypingBase as t
from ._base import EnforcementModelBase, FlextModelsEnforcementBase


class FlextModelsEnforcementSources(FlextModelsEnforcementBase):
    """Source-discriminator models used by enforcement catalog rules."""

    class EnforcementInfraRuleSource(EnforcementModelBase):
        """Rule applied by the flext-infra rule engine from its rule catalog.

        ``rule_ids`` name rules declared in flext-infra ``config/rules``; the
        engine reports typed findings keyed by those ids, and a named id the
        engine does not declare is a defect, never an empty result.
        """

        kind: Literal["flext_infra_rule"] = "flext_infra_rule"
        rule_ids: Annotated[t.StrSequence, Field(min_length=1)]

    class EnforcementRuntimeWarningSource(EnforcementModelBase):
        """Rule backed by a ``warnings`` category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: str

    class EnforcementBeartypeSource(EnforcementModelBase):
        """Rule dispatched through the runtime predicate bound to ``tag``."""

        kind: Literal["beartype"] = "beartype"
        tag: str

    class EnforcementCodeSmellSource(EnforcementModelBase):
        """Rule backed by a code-smell predicate (qlty/ metrics)."""

        kind: Literal["code_smell"] = "code_smell"
        smell_tag: str


__all__: list[str] = ["FlextModelsEnforcementSources"]
