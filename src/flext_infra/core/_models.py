"""Domain models for the core subpackage."""

from __future__ import annotations

from typing import Annotated

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

        stdout: Annotated[
            str, Field(default="", description="Captured standard output")
        ]
        stderr: Annotated[str, Field(default="", description="Captured standard error")]
        exit_code: Annotated[int, Field(description="Command exit code")]
        duration: Annotated[
            float, Field(default=0.0, ge=0.0, description="Duration in seconds")
        ] = 0.0

    class ValidationReport(FlextModels.ArbitraryTypesModel):
        """Validation report model with violations and summary."""

        passed: Annotated[bool, Field(description="Validation status")]
        violations: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Collected validation violations",
            ),
        ]
        summary: Annotated[
            str,
            Field(
                default="",
                description="Human-readable validation summary",
            ),
        ]

    class StubAnalysisReport(FlextModels.ArbitraryTypesModel):
        """Structured stub-chain analysis result for a project."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        mypy_hints: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="types- package hints from mypy output",
            ),
        ]
        internal_missing: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Missing internal imports",
            ),
        ]
        unresolved_missing: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Missing external imports without stubs",
            ),
        ]
        total_missing: Annotated[int, Field(ge=0, description="Total missing imports")]

    class PytestDiagnostics(FlextModels.ArbitraryTypesModel):
        """Extracted diagnostics summary from junit XML and pytest logs."""

        failed_count: Annotated[int, Field(ge=0, description="Failed test case count")]
        error_count: Annotated[int, Field(ge=0, description="Error trace count")]
        warning_count: Annotated[int, Field(ge=0, description="Warning line count")]
        skipped_count: Annotated[
            int, Field(ge=0, description="Skipped test case count")
        ]
        failed_cases: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Failed test labels",
            ),
        ]
        error_traces: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Collected error traces",
            ),
        ]
        warning_lines: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Captured warning lines",
            ),
        ]
        skip_cases: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Skipped test labels",
            ),
        ]
        slow_entries: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Slow test entries",
            ),
        ]

    class InventoryReport(FlextModels.ArbitraryTypesModel):
        """Summary of written inventory report artifacts."""

        total_scripts: Annotated[
            int, Field(ge=0, description="Total discovered scripts")
        ]
        reports_written: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Written report file paths",
            ),
        ]


__all__ = ["FlextInfraCoreModels"]
