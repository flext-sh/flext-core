"""Domain models for the core subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels


class FlextInfraCoreModels:
    """Models for core infrastructure services (subprocess, validation).

    Canonical base policy:
    - ``ArbitraryTypesModel`` for mutable report/result payloads.
    - ``FrozenStrictModel`` reserved for immutable settings/config contracts.
    """

    class CommandOutput(FlextModels.ArbitraryTypesModel):
        """Standardized subprocess output payload."""

        stdout: str = Field(default="", description="Captured standard output")
        stderr: str = Field(default="", description="Captured standard error")
        exit_code: int = Field(description="Command exit code")
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class ValidationReport(FlextModels.ArbitraryTypesModel):
        """Validation report model with violations and summary."""

        passed: bool = Field(description="Validation status")
        violations: list[str] = Field(
            default_factory=list,
            description="Collected validation violations",
        )
        summary: str = Field(
            default="",
            description="Human-readable validation summary",
        )

    class StubAnalysisReport(FlextModels.ArbitraryTypesModel):
        """Structured stub-chain analysis result for a project."""

        project: str = Field(min_length=1, description="Project name")
        mypy_hints: list[str] = Field(
            default_factory=list,
            description="types- package hints from mypy output",
        )
        internal_missing: list[str] = Field(
            default_factory=list,
            description="Missing internal imports",
        )
        unresolved_missing: list[str] = Field(
            default_factory=list,
            description="Missing external imports without stubs",
        )
        total_missing: int = Field(ge=0, description="Total missing imports")

    class PytestDiagnostics(FlextModels.ArbitraryTypesModel):
        """Extracted diagnostics summary from junit XML and pytest logs."""

        failed_count: int = Field(ge=0, description="Failed test case count")
        error_count: int = Field(ge=0, description="Error trace count")
        warning_count: int = Field(ge=0, description="Warning line count")
        skipped_count: int = Field(ge=0, description="Skipped test case count")
        failed_cases: list[str] = Field(
            default_factory=list,
            description="Failed test labels",
        )
        error_traces: list[str] = Field(
            default_factory=list,
            description="Collected error traces",
        )
        warning_lines: list[str] = Field(
            default_factory=list,
            description="Captured warning lines",
        )
        skip_cases: list[str] = Field(
            default_factory=list,
            description="Skipped test labels",
        )
        slow_entries: list[str] = Field(
            default_factory=list,
            description="Slow test entries",
        )

    class InventoryReport(FlextModels.ArbitraryTypesModel):
        """Summary of written inventory report artifacts."""

        total_scripts: int = Field(ge=0, description="Total discovered scripts")
        reports_written: list[str] = Field(
            default_factory=list,
            description="Written report file paths",
        )

    class ScriptsInventorySnapshot(FlextModels.ArbitraryTypesModel):
        """Inventory snapshot payload persisted to reports."""

        generated_at: str = Field(min_length=1, description="ISO timestamp")
        repo_root: str = Field(min_length=1, description="Workspace root path")
        total_scripts: int = Field(ge=0, description="Total discovered scripts")
        scripts: list[str] = Field(
            default_factory=list,
            description="Discovered script paths",
        )

    class ScriptsWiringSnapshot(FlextModels.ArbitraryTypesModel):
        """Script wiring report payload persisted to reports."""

        generated_at: str = Field(min_length=1, description="ISO timestamp")
        root_makefile: list[str] = Field(
            default_factory=list,
            description="Top-level makefile entries",
        )
        unwired_scripts: list[str] = Field(
            default_factory=list,
            description="Scripts not wired to automation",
        )

    class ExternalScriptsSnapshot(FlextModels.ArbitraryTypesModel):
        """External script candidate payload persisted to reports."""

        generated_at: str = Field(min_length=1, description="ISO timestamp")
        candidates: list[str] = Field(
            default_factory=list,
            description="External script candidates",
        )


__all__ = ["FlextInfraCoreModels"]
