"""Domain models for the enforcement subpackage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from flext_core import m
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_infra import t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraModelsEnforcement:
    """Models for enforcement collection — exposed through the ``m.Infra`` facade."""

    class FixedViolation(m.Value):
        """One violation that was fixed."""

        rule_id: Annotated[str, m.Field(description="Enforcement rule ID")]
        file_path: Annotated[str, m.Field(description="File that was modified")]
        message: Annotated[str, m.Field(description="Human-readable fix summary")]

    class PreviewedViolation(m.Value):
        """One violation with a non-mutating dry-run fix preview."""

        rule_id: Annotated[str, m.Field(description="Enforcement rule ID")]
        file_path: Annotated[str, m.Field(description="File that would change")]
        message: Annotated[str, m.Field(description="Human-readable preview summary")]

    class SkippedViolation(m.Value):
        """One violation that was skipped."""

        rule_id: Annotated[str, m.Field(description="Enforcement rule ID")]
        file_path: Annotated[str, m.Field(description="File containing the violation")]
        reason: Annotated[str, m.Field(description="Why the fix was skipped")]

    class FailedFix(m.Value):
        """One fix attempt that failed."""

        rule_id: Annotated[str, m.Field(description="Enforcement rule ID")]
        file_path: Annotated[str, m.Field(description="Target file when known")]
        error: Annotated[str, m.Field(description="Failure message")]

    class ProjectFixResult(m.Value):
        """Aggregated fix result for a single project."""

        project: Annotated[str, m.Field(description="Project name")]
        fixed: Annotated[
            t.SequenceOf[FlextInfraModelsEnforcement.FixedViolation],
            m.Field(description="Fixed violations"),
        ] = ()
        previewed: Annotated[
            t.SequenceOf[FlextInfraModelsEnforcement.PreviewedViolation],
            m.Field(description="Dry-run previews"),
        ] = ()
        skipped: Annotated[
            t.SequenceOf[FlextInfraModelsEnforcement.SkippedViolation],
            m.Field(description="Skipped violations"),
        ] = ()
        failed: Annotated[
            t.SequenceOf[FlextInfraModelsEnforcement.FailedFix],
            m.Field(description="Failed fix attempts"),
        ] = ()
        files_modified: Annotated[
            t.StrSequence, m.Field(description="Modified file paths")
        ] = ()

    class EnforcementEvaluation(m.ArbitraryTypesModel):
        """Collected rule probes and collection failures for one project."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        violations: Annotated[
            tuple[tuple[me.EnforcementRuleSpec, p.AttributeProbe], ...],
            m.Field(description="Rule/probe pairs collected for the project"),
        ]
        failures: Annotated[
            tuple[FlextInfraModelsEnforcement.FailedFix, ...],
            m.Field(description="Structured collection/routing failures"),
        ]

    class EnforcementProbe(m.ArbitraryTypesModel):
        """Structural probe consumed by fixer adapters.

        Replaces the former ``SimpleNamespace`` probe: validated core fields
        plus ``extra="allow"`` for rule-specific attributes that adapters
        inspect via ``getattr``.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="allow")

        file_path: Annotated[str, m.Field(description="Target file path")]
        line: Annotated[int, m.Field(description="Line number of the violation")] = 0
        rule_id: Annotated[str, m.Field(description="Originating rule identifier")] = ""


__all__: list[str] = ["FlextInfraModelsEnforcement"]
