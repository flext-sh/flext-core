"""Catalog source models for enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from flext_core._models._enforcement._base import (
    FlextModelsEnforcementBase,
    FlextModelsEnforcementModelBase,
)
from flext_core._typings.base import FlextTypingBase


class FlextModelsEnforcementSources(FlextModelsEnforcementBase):
    """Source-discriminator models used by enforcement catalog rules."""

    class EnforcementInfraRuleSource(FlextModelsEnforcementModelBase):
        """Rule applied by the flext-infra rule engine from its rule catalog.

        ``rule_ids`` name rules declared in flext-infra ``config/rules``; the
        engine reports typed findings keyed by those ids, and a named id the
        engine does not declare is a defect, never an empty result.
        """

        kind: Literal["flext_infra_rule"] = "flext_infra_rule"
        rule_ids: Annotated[FlextTypingBase.StrSequence, Field(min_length=1)]

    class EnforcementRuntimeWarningSource(FlextModelsEnforcementModelBase):
        """Rule backed by a ``warnings`` category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: str

    class EnforcementBeartypeSource(FlextModelsEnforcementModelBase):
        """Rule dispatched through the runtime predicate bound to ``tag``.

        ``tag`` is the rule's identity in the runtime engine; its predicate
        kind is derived from the tag's binding, never stored beside it.
        """

        kind: Literal["beartype"] = "beartype"
        tag: Annotated[str, Field(min_length=1)]

    class EnforcementCodeSmellSource(FlextModelsEnforcementModelBase):
        """Rule backed by a code-smell predicate (qlty/ metrics)."""

        kind: Literal["code_smell"] = "code_smell"
        smell_tag: str

    class EnforcementFixAction(FlextModelsEnforcementModelBase):
        """Actionable fix contract for an enforcement rule.

        Stored on ``EnforcementRuleSpec.fix_action`` and consumed by the
        flext-infra fix orchestrator to route violations to the right fixer.
        """

        kind: Literal["transformer", "rope", "manual"]
        target: str
        params: FlextTypingBase.JsonMapping = Field(default_factory=dict)
        safe: bool = True


__all__: list[str] = ["FlextModelsEnforcementSources"]
