"""Catalog source models for enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..._constants.enforcement import FlextConstantsEnforcement as ce
from ..._typings.base import FlextTypingBase as t
from flext_core._models._enforcement._base import (
    FlextModelsEnforcementBase,
    FlextModelsEnforcementModelBase,
)


class FlextModelsEnforcementSources(FlextModelsEnforcementBase):
    """Source-discriminator models used by enforcement catalog rules."""

    class EnforcementInfraDetectorSource(FlextModelsEnforcementModelBase):
        """Rule backed by a ``FlextInfraNamespaceEnforcer`` detector field."""

        kind: Literal["flext_infra_detector"] = "flext_infra_detector"
        violation_field: str
        match_missing: bool = False

    class EnforcementTestsValidatorSource(FlextModelsEnforcementModelBase):
        """Rule backed by a ``FlextTestsValidator`` classmethod."""

        kind: Literal["flext_tests_validator"] = "flext_tests_validator"
        method: str
        rule_ids: t.StrSequence = ()

    class EnforcementRuntimeWarningSource(FlextModelsEnforcementModelBase):
        """Rule backed by a ``warnings`` category raised at runtime."""

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: str

    class EnforcementBeartypeSource(FlextModelsEnforcementModelBase):
        """Rule dispatched through a beartype predicate binding."""

        kind: Literal["beartype"] = "beartype"
        predicate_kind: ce.EnforcementPredicateKind

    class EnforcementRuffSource(FlextModelsEnforcementModelBase):
        """Rule delegated to ruff."""

        kind: Literal["ruff"] = "ruff"
        rule_code: str

    class EnforcementSkillPointerSource(FlextModelsEnforcementModelBase):
        """Rule as narrative skill content only."""

        kind: Literal["skill_pointer"] = "skill_pointer"
        skill: str
        anchor: str = ""

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
        params: t.JsonMapping = Field(default_factory=dict)
        safe: bool = True


__all__: list[str] = ["FlextModelsEnforcementSources"]
