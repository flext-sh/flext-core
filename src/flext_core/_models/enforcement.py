"""Pydantic v2 data models for enforcement violation reporting.

Typed containers (``m.Enforcement.*``) that enforcement checks return
instead of raw string lists. This eliminates ``t.StrSequence`` conversions
at call sites and lets callers narrow by severity / layer / qualname
without reparsing formatted messages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from enum import StrEnum, unique
from typing import Annotated, ClassVar, Literal

from pydantic import Discriminator, Field, model_validator

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.annotateds import FlextTypesAnnotateds as ta
from flext_core._typings.base import FlextTypingBase as t


class FlextModelsEnforcement:
    """Namespace for enforcement violation data models.

    Exposed via MRO on the ``m`` facade. Members include violation containers
    (``Violation`` / ``Report``) plus the cross-layer rule catalog types
    (``EnforcementRuleSeverity`` / ``EnforcementSourceKind`` /
    ``EnforcementRuleSpec`` / ``EnforcementCatalog``) that index detectors,
    validators, runtime warnings, ruff codes, ast-grep rules, and skill
    pointers under stable ``ENFORCE-NNN`` identifiers.
    """

    class Violation(mp.BaseModel):
        """Single enforcement violation located at ``qualname``."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        qualname: Annotated[
            str,
            Field(description="Qualified name of the violating class/attribute."),
        ]
        layer: Annotated[
            str,
            Field(description="Layer where the violation was detected."),
        ]
        severity: Annotated[
            str,
            Field(description='Severity label (e.g. "HARD rules", "best practices").'),
        ]
        message: Annotated[
            str,
            Field(description="Human-readable violation message."),
        ]

    class Report(mp.BaseModel):
        """Aggregated violation report returned by a check or runner."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        violations: Annotated[
            Sequence[FlextModelsEnforcement.Violation],
            Field(
                default_factory=list,
                description="Violations detected by the check.",
            ),
        ] = Field(default_factory=tuple)

        @property
        def messages(self) -> t.StrSequence:
            """Return plain messages for backwards-compatible emission."""
            return [v.message for v in self.violations]

        @property
        def empty(self) -> bool:
            """True when no violations were recorded."""
            return not self.violations

        def __len__(self) -> int:
            """Expose violation count for ``len(report)``."""
            return len(self.violations)

        def __bool__(self) -> bool:
            """Truthy when violations exist."""
            return bool(self.violations)

        def __getitem__(self, index: int) -> str:
            """Return the nth message for ``report[i]`` access."""
            return self.messages[index]

        def __contains__(self, fragment: t.Scalar | None) -> bool:
            """``"Any" in report`` — search message text."""
            if not isinstance(fragment, str):
                return False
            return any(fragment in m for m in self.messages)

    @unique
    class EnforcementRuleSeverity(StrEnum):
        """Severity scale for catalog rules.

        Matches ``c.Tests.ValidatorSeverity`` semantics (CRITICAL/HIGH/MEDIUM/LOW)
        so adapters can map either way without reshaping. Distinct from
        ``c.EnforcementSeverity`` (HARD rules / best practices / namespace
        rules), which describes Pydantic-governance violations.
        """

        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    @unique
    class EnforcementSourceKind(StrEnum):
        """Addressable origin layer for a catalog rule."""

        FLEXT_INFRA_DETECTOR = "flext_infra_detector"
        FLEXT_TESTS_VALIDATOR = "flext_tests_validator"
        RUNTIME_WARNING = "runtime_warning"
        BEARTYPE = "beartype"
        RUFF = "ruff"
        AST_GREP = "ast_grep"
        SKILL_POINTER = "skill_pointer"

    class EnforcementInfraDetectorSource(mp.BaseModel):
        """Rule backed by a ``FlextInfraNamespaceEnforcer`` detector field."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["flext_infra_detector"] = "flext_infra_detector"
        violation_field: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Attribute name on m.Infra.ProjectEnforcementReport — e.g. "
                    "'loose_objects', 'cyclic_imports', 'facade_statuses'."
                ),
            ),
        ]
        match_missing: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True the adapter flags entries where 'exists' is "
                    "False (used with 'facade_statuses' to detect missing "
                    "facades); when False the field is treated as a violation "
                    "list already."
                ),
            ),
        ]

    class EnforcementTestsValidatorSource(mp.BaseModel):
        """Rule backed by a ``FlextTestsValidator`` (``tv.*``) classmethod."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["flext_tests_validator"] = "flext_tests_validator"
        method: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Public classmethod on FlextTestsValidator — one of "
                    "'imports', 'types', 'bypass', 'layer', 'tests', "
                    "'validate_config', 'markdown'."
                ),
            ),
        ]
        rule_ids: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description=(
                    "Filter within the validator's output — e.g. ('IMPORT-001',"
                    " 'IMPORT-002'). Empty tuple means accept every rule the "
                    "validator emits."
                ),
            ),
        ]

    class EnforcementRuntimeWarningSource(mp.BaseModel):
        """Rule backed by a ``warnings``-category raised at runtime."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["runtime_warning"] = "runtime_warning"
        category: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Dotted path to the warning class — e.g. "
                    "'flext_core._constants.enforcement.FlextMroViolation'."
                ),
            ),
        ]

    class EnforcementBeartypeSource(mp.BaseModel):
        """Rule delegated to a ``FlextUtilitiesBeartypeEngine`` hook."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["beartype"] = "beartype"
        hook: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Hook name on FlextUtilitiesBeartypeEngine — e.g. "
                    "'check_model_rebuild_call'."
                ),
            ),
        ]

    class EnforcementRuffSource(mp.BaseModel):
        """Rule delegated to ``ruff`` (documentation-only in dispatcher)."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["ruff"] = "ruff"
        rule_code: Annotated[
            ta.NonEmptyStr,
            Field(description="Ruff rule code — e.g. 'ANN401', 'PGH003', 'TID252'."),
        ]

    class EnforcementAstGrepSource(mp.BaseModel):
        """Rule delegated to ``ast-grep`` via ``sgconfig.yml``."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["ast_grep"] = "ast_grep"
        skill: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Skill directory under .agents/skills — e.g. 'flext-patterns'."
                ),
            ),
        ]
        rule_id: Annotated[
            ta.NonEmptyStr,
            Field(
                description=(
                    "Rule id within .agents/skills/<skill>/rules.yml::rules[].id."
                ),
            ),
        ]

    class EnforcementSkillPointerSource(mp.BaseModel):
        """Rule that is narrative skill content only (no automation)."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        kind: Literal["skill_pointer"] = "skill_pointer"
        skill: Annotated[
            ta.NonEmptyStr,
            Field(description="Skill directory under .agents/skills."),
        ]
        anchor: Annotated[
            str,
            Field(
                default="",
                description="Optional in-file anchor (heading slug) inside SKILL.md.",
            ),
        ]

    class EnforcementRuleSpec(mp.BaseModel):
        """Single rule entry in the enforcement catalog."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        id: Annotated[
            str,
            Field(
                pattern=r"^ENFORCE-\d{3}$",
                description="Stable catalog identifier (ENFORCE-NNN).",
            ),
        ]
        description: Annotated[
            ta.NonEmptyStr,
            Field(description="Human-readable rule summary."),
        ]
        severity: Annotated[
            FlextModelsEnforcement.EnforcementRuleSeverity,
            Field(description="Catalog severity — CRITICAL/HIGH/MEDIUM/LOW."),
        ]
        source: Annotated[
            (
                FlextModelsEnforcement.EnforcementInfraDetectorSource
                | FlextModelsEnforcement.EnforcementTestsValidatorSource
                | FlextModelsEnforcement.EnforcementRuntimeWarningSource
                | FlextModelsEnforcement.EnforcementBeartypeSource
                | FlextModelsEnforcement.EnforcementRuffSource
                | FlextModelsEnforcement.EnforcementAstGrepSource
                | FlextModelsEnforcement.EnforcementSkillPointerSource
            ),
            Discriminator("kind"),
            Field(description="Addressable origin of the rule."),
        ]
        agents_md_anchor: Annotated[
            str,
            Field(
                default="",
                description="Optional AGENTS.md anchor (heading slug).",
            ),
        ]
        skills: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description="Skill directories under .agents/skills that describe this rule.",
            ),
        ]
        enabled: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "Whether the dispatcher runs this rule. Skill-pointer/ruff/"
                    "ast-grep rules stay enabled=True for indexing but the "
                    "dispatcher skips them (documentation-only)."
                ),
            ),
        ]
        promote_to_error_when_strict: Annotated[
            bool,
            Field(
                default=True,
                description=(
                    "When --flext-enforce-strict is set, violations of this "
                    "rule become pytest failures instead of warnings."
                ),
            ),
        ]
        notes: Annotated[
            str,
            Field(default="", description="Free-form implementation notes."),
        ]

    class EnforcementCatalog(mp.BaseModel):
        """Frozen catalog of all enforcement rules (SSOT)."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        version: Annotated[
            int,
            Field(
                default=1,
                description="Schema version; bumped only on breaking shape changes.",
            ),
        ]
        rules: Annotated[
            tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...],
            Field(description="All catalog rules in registration order."),
        ]

        @model_validator(mode="after")
        def _check_unique_ids(self) -> FlextModelsEnforcement.EnforcementCatalog:
            seen: set[str] = set()
            for rule in self.rules:
                if rule.id in seen:
                    msg = f"duplicate rule id in catalog: {rule.id!r}"
                    raise ValueError(msg)
                seen.add(rule.id)
            return self

        def by_id(
            self, rule_id: str
        ) -> FlextModelsEnforcement.EnforcementRuleSpec | None:
            """Return the rule with ``rule_id`` or ``None`` if absent."""
            for rule in self.rules:
                if rule.id == rule_id:
                    return rule
            return None

        def enabled_rules(
            self,
        ) -> tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...]:
            """Return only the rules with ``enabled=True``."""
            return tuple(r for r in self.rules if r.enabled)

        def by_kind(
            self,
            kind: FlextModelsEnforcement.EnforcementSourceKind,
        ) -> tuple[FlextModelsEnforcement.EnforcementRuleSpec, ...]:
            """Filter rules by source kind."""
            return tuple(r for r in self.rules if r.source.kind == kind.value)


__all__: list[str] = ["FlextModelsEnforcement"]
