"""Enforcement rule catalog models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Discriminator, Field, model_validator

from flext_core import c
from flext_core._models._enforcement._base import FlextModelsEnforcementModelBase
from flext_core._models._enforcement._sources import FlextModelsEnforcementSources
from flext_core._typings.base import FlextTypingBase
from flext_core.typings import EnforcementRuleSource


class FlextModelsEnforcementCatalog(FlextModelsEnforcementSources):
    """Rule-spec and catalog containers for enforcement."""

    class EnforcementRuleSpec(FlextModelsEnforcementModelBase):
        """Single rule entry in the enforcement catalog."""

        id: Annotated[str, Field(pattern=c.PATTERN_ENFORCE_RULE_ID)]
        description: str
        severity: c.EnforcementRuleSeverity
        source: Annotated[EnforcementRuleSource, Discriminator("kind")]
        agents_md_anchor: str = ""
        skills: FlextTypingBase.StrSequence = ()
        enabled: bool = True
        promote_to_error_when_strict: bool = True
        notes: str = ""
        fix_action: FlextModelsEnforcementSources.EnforcementFixAction | None = None

    class EnforcementCatalog(FlextModelsEnforcementModelBase):
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
            self,
            rule_id: str,
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
            self,
            kind: c.EnforcementSourceKind,
        ) -> tuple[FlextModelsEnforcementCatalog.EnforcementRuleSpec, ...]:
            """Filter rules by source kind.

            Returns:
                The resulting ``tuple[FlextModelsEnforcementCatalog.EnforcementRuleSpec,
                    ...]``.
            """
            return tuple(rule for rule in self.rules if rule.source.kind == kind.value)


__all__: list[str] = ["FlextModelsEnforcementCatalog"]
