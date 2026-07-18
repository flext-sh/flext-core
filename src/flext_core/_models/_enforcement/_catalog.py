"""Enforcement rule catalog models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Discriminator, Field, model_validator

from flext_core import FlextConstants as c

from ._base import EnforcementModelBase, FlextModelsEnforcementBase
from ._sources import FlextModelsEnforcementSources
from flext_core._typings.base import FlextTypingBase as t


type EnforcementRuleSource = (
    FlextModelsEnforcementSources.EnforcementInfraDetectorSource
    | FlextModelsEnforcementSources.EnforcementTestsValidatorSource
    | FlextModelsEnforcementSources.EnforcementRuntimeWarningSource
    | FlextModelsEnforcementSources.EnforcementBeartypeSource
    | FlextModelsEnforcementSources.EnforcementCodeSmellSource
    | FlextModelsEnforcementSources.EnforcementRuffSource
    | FlextModelsEnforcementSources.EnforcementSkillPointerSource
)


class FlextModelsEnforcementCatalog(FlextModelsEnforcementSources):
    """Rule-spec and catalog containers for enforcement."""

    class EnforcementRuleSpec(EnforcementModelBase):
        """Single rule entry in the enforcement catalog."""

        id: Annotated[str, Field(pattern=c.PATTERN_ENFORCE_RULE_ID)]
        description: str
        severity: FlextModelsEnforcementBase.EnforcementRuleSeverity
        source: Annotated[EnforcementRuleSource, Discriminator("kind")]
        agents_md_anchor: str = ""
        skills: t.StrSequence = ()
        enabled: bool = True
        promote_to_error_when_strict: bool = True
        notes: str = ""
        fix_action: FlextModelsEnforcementSources.EnforcementFixAction | None = None

    class EnforcementCatalog(EnforcementModelBase):
        """Frozen catalog of all enforcement rules."""

        version: int = 1
        rules: tuple[FlextModelsEnforcementCatalog.EnforcementRuleSpec, ...] = ()

        @model_validator(mode="after")
        def _check_unique_ids(self) -> FlextModelsEnforcementCatalog.EnforcementCatalog:
            seen: set[str] = set()
            for rule in self.rules:
                if rule.id in seen:
                    msg = f"duplicate rule id in catalog: {rule.id!r}"
                    raise ValueError(msg)
                seen.add(rule.id)
            return self

        def by_id(
            self, rule_id: str
        ) -> FlextModelsEnforcementCatalog.EnforcementRuleSpec | None:
            """Return the rule with ``rule_id`` or ``None`` if absent."""
            for rule in self.rules:
                if rule.id == rule_id:
                    return rule
            return None

        def enabled_rules(
            self,
        ) -> tuple[FlextModelsEnforcementCatalog.EnforcementRuleSpec, ...]:
            """Return only the rules with ``enabled=True``."""
            return tuple(rule for rule in self.rules if rule.enabled)

        def by_kind(
            self, kind: FlextModelsEnforcementBase.EnforcementSourceKind
        ) -> tuple[FlextModelsEnforcementCatalog.EnforcementRuleSpec, ...]:
            """Filter rules by source kind."""
            return tuple(rule for rule in self.rules if rule.source.kind == kind.value)


__all__: list[str] = ["FlextModelsEnforcementCatalog"]
