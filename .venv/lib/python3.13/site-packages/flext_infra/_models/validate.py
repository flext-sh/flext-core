"""Domain models for the core subpackage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from flext_core import m
from flext_infra import c, t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsCore:
    """Models for core infrastructure services (subprocess, validation).

    Canonical base policy:
    - ``ArbitraryTypesModel`` for mutable report/result payloads.
    - ``ContractModel`` reserved for immutable settings/settings contracts.
    """

    class ValidationReport(m.ArbitraryTypesModel):
        """Validation report model with violations and summary."""

        passed: Annotated[bool, m.Field(description="Validation status")]
        violations: Annotated[
            t.StrSequence, m.Field(description="Collected validation violations")
        ] = m.Field(default_factory=tuple)
        summary: Annotated[
            str, m.Field(description="Human-readable validation summary")
        ] = ""

    class SkillRuleEvaluationContext(m.ArbitraryTypesModel):
        """Resolved inputs for one skill rule evaluation pass."""

        rules_list: Annotated[t.JsonList, m.Field(description="Rules to evaluate")]
        skill_dir: Annotated[Path, m.Field(description="Skill directory path")]
        root: Annotated[Path, m.Field(description="Repository root path")]
        mode: Annotated[
            c.Infra.OperationMode, m.Field(description="Skill validation mode")
        ]
        include_globs: Annotated[t.StrSequence, m.Field(description="Include globs")]
        exclude_globs: Annotated[t.StrSequence, m.Field(description="Exclude globs")]

    class SkillReportContext(m.ArbitraryTypesModel):
        """Resolved inputs for one skill validation report."""

        rules: Annotated[
            t.MappingKV[str, t.Infra.InfraValue], m.Field(description="Rules payload")
        ]
        root: Annotated[Path, m.Field(description="Repository root path")]
        skill_name: Annotated[str, m.Field(description="Skill folder name")]
        mode: Annotated[
            c.Infra.OperationMode, m.Field(description="Skill validation mode")
        ]
        counts: Annotated[t.IntMapping, m.Field(description="Violation counts")]
        violations: Annotated[t.StrSequence, m.Field(description="Violations")]

    class StubAnalysisReport(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Structured typed-dependency analysis result for a project."""

        mypy_hints: Annotated[
            t.MutableSequenceOf[str],
            m.Field(description="Install-package hints extracted from mypy output"),
        ] = m.Field(default_factory=list)
        internal_missing: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Missing internal imports")
        ] = m.Field(default_factory=list)
        unresolved_missing: Annotated[
            t.MutableSequenceOf[str],
            m.Field(
                description="Missing external imports without an installed typed dependency"
            ),
        ] = m.Field(default_factory=list)
        total_missing: Annotated[
            t.NonNegativeInt, m.Field(description="Total missing imports")
        ]

    class PytestDiagnostics(m.ArbitraryTypesModel):
        """Extracted diagnostics summary from junit XML and pytest logs."""

        failed_count: Annotated[
            t.NonNegativeInt, m.Field(description="Failed test case count")
        ]
        error_count: Annotated[
            t.NonNegativeInt, m.Field(description="Errored test case count")
        ]
        warning_count: Annotated[
            t.NonNegativeInt, m.Field(description="Warning line count")
        ]
        skipped_count: Annotated[
            t.NonNegativeInt, m.Field(description="Skipped test case count")
        ]
        failed_cases: Annotated[
            t.StrSequence, m.Field(description="Failed test labels")
        ] = m.Field(default_factory=tuple)
        error_traces: Annotated[
            t.StrSequence, m.Field(description="Collected error traces")
        ] = m.Field(default_factory=tuple)
        warning_lines: Annotated[
            t.StrSequence, m.Field(description="Captured warning lines")
        ] = m.Field(default_factory=tuple)
        skip_cases: Annotated[
            t.StrSequence, m.Field(description="Skipped test labels")
        ] = m.Field(default_factory=tuple)
        slow_entries: Annotated[
            t.StrSequence, m.Field(description="Slow test entries")
        ] = m.Field(default_factory=tuple)

    class DiagResult(m.ArbitraryTypesModel):
        """Internal container for extracted diagnostics.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        failed_cases: Annotated[
            t.MutableSequenceOf[str],
            m.Field(description="Collected failed test-case labels"),
        ] = m.Field(default_factory=list)
        error_cases: Annotated[
            t.MutableSequenceOf[str],
            m.Field(description="Collected error test-case labels"),
        ] = m.Field(default_factory=list)
        error_traces: Annotated[
            t.MutableSequenceOf[str],
            m.Field(description="Collected error trace chunks"),
        ] = m.Field(default_factory=list)
        skip_cases: Annotated[
            t.MutableSequenceOf[str],
            m.Field(description="Collected skipped test-case labels"),
        ] = m.Field(default_factory=list)
        warning_lines: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Collected warning lines")
        ] = m.Field(default_factory=list)
        slow_entries: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Collected slow-test entries")
        ] = m.Field(default_factory=list)

    class InventoryReport(m.ArbitraryTypesModel):
        """Summary of written inventory report artifacts."""

        total_scripts: Annotated[
            t.NonNegativeInt, m.Field(description="Total discovered scripts")
        ]
        reports_written: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Written report file paths")
        ] = m.Field(default_factory=list)

    class GateContractViolation(m.Value):
        """One gate-script contract violation."""

        # Why: leaf validate contract owned by m.Infra (collision-safe vs census Violation).
        script: Annotated[str, m.Field(description="Script path")]
        check: Annotated[str, m.Field(description="Failed check")]
        message: Annotated[str, m.Field(description="Violation message")]
        severity: Annotated[str, m.Field(description="Severity")] = (
            c.Infra.GateSeverity.ERROR.value
        )

    class GateContractScriptInfo(m.Value):
        """Validation result for one gate script."""

        path: Annotated[str, m.Field(description="Script path")]
        extension: Annotated[str, m.Field(description="File extension")]
        role: Annotated[str, m.Field(description="Script role")]
        violations: Annotated[
            tuple[FlextInfraModelsCore.GateContractViolation, ...],
            m.Field(description="Violations"),
        ] = ()

    class GateContractSummary(m.Value):
        """Aggregate gate-contract counts."""

        errors: Annotated[int, m.Field(description="Error count")] = 0
        gate_scripts: Annotated[int, m.Field(description="Gate script count")] = 0
        ok: Annotated[int, m.Field(description="Passing gate script count")] = 0
        warnings: Annotated[int, m.Field(description="Warning count")] = 0

    class GateContractRunResult(m.Value):
        """CLI outcome for one gate-contract validation run."""

        exit_code: Annotated[int, m.Field(description="Process exit code")]
        violation_count: Annotated[int, m.Field(description="Error count")]

    class NamespaceValidateCommand(mm.ReadMixin, m.ContractModel):
        """CLI payload for ``flext-infra validate namespace``.

        Read-only namespace rule scan (NS-000..003) across selected projects.
        """

        scan_tests: Annotated[
            bool,
            m.Field(
                alias="scan-tests",
                description="Include test packages in the namespace scan",
            ),
        ] = False


__all__: list[str] = ["FlextInfraModelsCore"]
