"""Domain models for the check subpackage."""

from __future__ import annotations

from pydantic import ConfigDict, Field, computed_field

from flext_core import FlextModels
from flext_infra.constants import FlextInfraConstants as c


class FlextInfraCheckModels:
    """Quality-gate check domain models."""

    class Issue(FlextModels.FrozenStrictModel):
        """Single issue reported by a quality gate tool."""

        file: str = Field(description="Source file path")
        line: int = Field(description="Line number")
        column: int = Field(description="Column number")
        code: str = Field(description="Rule or error code")
        message: str = Field(description="Human-readable issue description")
        severity: str = Field(
            default=c.Infra.Toml.ERROR,
            description="Issue severity level",
        )

        @computed_field
        @property
        def formatted(self) -> str:
            """Format issue as ``file:line:col [code] message``."""
            code_part = f"[{self.code}] " if self.code else ""
            return (
                f"{self.file}:{self.line}:{self.column} {code_part}{self.message}"
            ).strip()

    class GateExecution(FlextModels.ArbitraryTypesModel):
        """Execution result for a single quality gate."""

        result: FlextInfraCheckModels.GateResult = Field(
            description="Gate result model"
        )
        issues: list[FlextInfraCheckModels.Issue] = Field(
            default_factory=list,
            description="Detected issues",
        )
        raw_output: str = Field(default="", description="Raw tool output")

    class GateResult(FlextModels.ArbitraryTypesModel):
        """Result summary for a single quality gate execution."""

        gate: str = Field(min_length=1, description="Gate name")
        project: str = Field(min_length=1, description="Project name")
        passed: bool = Field(description="Gate execution status")
        errors: list[str] = Field(
            default_factory=list,
            description="Gate error messages",
        )
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class ProjectResult(FlextModels.ArbitraryTypesModel):
        """Aggregated gate results for a single project."""

        project: str = Field(description="Project name")
        gates: dict[str, FlextInfraCheckModels.GateExecution] = Field(
            default_factory=dict,
            description="Gate name to execution mapping",
        )

        @computed_field
        @property
        def passed(self) -> bool:
            """Whether every gate passed."""
            return all(v.result.passed for v in self.gates.values())

        @computed_field
        @property
        def total_errors(self) -> int:
            """Total issue count across all gates."""
            return sum(len(v.issues) for v in self.gates.values())

    class RunCommandResult(FlextModels.FrozenStrictModel):
        """Subprocess execution result."""

        stdout: str = Field(description="Captured standard output")
        stderr: str = Field(description="Captured standard error")
        returncode: int = Field(description="Process exit code")

    # -- Tool-specific JSON parsing models ---------------------------------

    class Parsers:
        """Models for parsing tool-specific JSON output."""

        class RuffLintLocation(FlextModels.ArbitraryTypesModel):
            """Location block inside a Ruff lint JSON entry."""

            model_config = ConfigDict(extra="ignore")

            row: int = Field(default=0, description="Line number")
            column: int = Field(default=0, description="Column number")

        class RuffLintError(FlextModels.ArbitraryTypesModel):
            """Single Ruff lint error from JSON output."""

            model_config = ConfigDict(extra="ignore")

            filename: str = Field(default="?", description="Source file path")
            location: FlextInfraCheckModels.Parsers.RuffLintLocation = Field(
                default_factory=lambda: (
                    FlextInfraCheckModels.Parsers.RuffLintLocation()
                ),
                description="Error location",
            )
            code: str = Field(default="", description="Ruff rule code")
            message: str = Field(default="", description="Error description")

        class PyreflyError(FlextModels.ArbitraryTypesModel):
            """Single Pyrefly error entry."""

            model_config = ConfigDict(extra="ignore")

            path: str = Field(default="?", description="Source file path")
            line: int = Field(default=0, description="Line number")
            column: int = Field(default=0, description="Column number")
            name: str = Field(default="", description="Error name/code")
            description: str = Field(default="", description="Error description")
            severity: str = Field(
                default=c.Infra.Toml.ERROR,
                description="Severity level",
            )

        class PyreflyOutput(FlextModels.ArbitraryTypesModel):
            """Pyrefly JSON output wrapper."""

            model_config = ConfigDict(extra="ignore")

            errors: list[FlextInfraCheckModels.Parsers.PyreflyError] = Field(
                default_factory=list,
                description="Pyrefly errors",
            )

        class MypyJsonError(FlextModels.ArbitraryTypesModel):
            """Single mypy JSON error entry."""

            model_config = ConfigDict(extra="ignore")

            file: str = Field(default="?", description="Source file path")
            line: int = Field(default=0, description="Line number")
            column: int = Field(default=0, description="Column number")
            code: str = Field(default="", description="Mypy error code")
            message: str = Field(default="", description="Error description")
            severity: str = Field(
                default=c.Infra.Toml.ERROR,
                description="Severity level",
            )

        class PyrightPosition(FlextModels.ArbitraryTypesModel):
            """Pyright position with zero-based line/character."""

            model_config = ConfigDict(extra="ignore")

            line: int = Field(default=0, description="Zero-based line number")
            character: int = Field(
                default=0,
                description="Zero-based character offset",
            )

        class PyrightRange(FlextModels.ArbitraryTypesModel):
            """Pyright range with start position."""

            model_config = ConfigDict(extra="ignore")

            start: FlextInfraCheckModels.Parsers.PyrightPosition = Field(
                default_factory=lambda: FlextInfraCheckModels.Parsers.PyrightPosition(),
                description="Range start",
            )

        class PyrightDiagnostic(FlextModels.ArbitraryTypesModel):
            """Single Pyright diagnostic entry."""

            model_config = ConfigDict(extra="ignore")

            file: str = Field(default="?", description="Source file path")
            range: FlextInfraCheckModels.Parsers.PyrightRange = Field(
                default_factory=lambda: FlextInfraCheckModels.Parsers.PyrightRange(),
                description="Diagnostic range",
            )
            rule: str = Field(default="", description="Pyright rule name")
            message: str = Field(
                default="",
                description="Diagnostic message",
            )
            severity: str = Field(
                default=c.Infra.Toml.ERROR,
                description="Severity level",
            )

        class PyrightOutput(FlextModels.ArbitraryTypesModel):
            """Pyright JSON output wrapper."""

            model_config = ConfigDict(extra="ignore")

            general_diagnostics: list[
                FlextInfraCheckModels.Parsers.PyrightDiagnostic
            ] = Field(
                alias="generalDiagnostics",
                default_factory=list,
                description="General diagnostics list",
            )

        class BanditIssue(FlextModels.ArbitraryTypesModel):
            """Single Bandit security finding."""

            model_config = ConfigDict(extra="ignore")

            filename: str = Field(
                default="?",
                description="Source file path",
            )
            line_number: int = Field(default=0, description="Line number")
            test_id: str = Field(default="", description="Bandit test ID")
            issue_text: str = Field(
                default="",
                description="Issue description",
            )
            issue_severity: str = Field(
                default="MEDIUM",
                description="Severity level",
            )

        class BanditOutput(FlextModels.ArbitraryTypesModel):
            """Bandit JSON output wrapper."""

            model_config = ConfigDict(extra="ignore")

            results: list[FlextInfraCheckModels.Parsers.BanditIssue] = Field(
                default_factory=list,
                description="Bandit findings",
            )

    # -- SARIF 2.1.0 report models -----------------------------------------

    class Sarif:
        """SARIF 2.1.0 report models."""

        class MessageText(FlextModels.FrozenStrictModel):
            """SARIF message with text content."""

            text: str = Field(description="Message text")

        class RuleDescriptor(FlextModels.FrozenStrictModel):
            """SARIF rule descriptor with short description."""

            id: str = Field(description="Rule identifier")
            short_description: FlextInfraCheckModels.Sarif.MessageText = Field(
                alias="shortDescription",
                description="Rule short description",
            )

        class ArtifactLocation(FlextModels.FrozenStrictModel):
            """SARIF artifact location with URI."""

            uri: str = Field(description="Artifact URI")
            uri_base_id: str = Field(
                alias="uriBaseId",
                default="%SRCROOT%",
                description="URI base identifier",
            )

        class Region(FlextModels.FrozenStrictModel):
            """SARIF region with start line/column."""

            start_line: int = Field(description="Start line (1-based)")
            start_column: int = Field(description="Start column (1-based)")

        class PhysicalLocation(FlextModels.FrozenStrictModel):
            """SARIF physical location combining artifact and region."""

            artifact_location: FlextInfraCheckModels.Sarif.ArtifactLocation = Field(
                alias="artifactLocation", description="Artifact location"
            )
            region: FlextInfraCheckModels.Sarif.Region = Field(
                description="Source region"
            )

        class Location(FlextModels.FrozenStrictModel):
            """SARIF location wrapper."""

            physical_location: FlextInfraCheckModels.Sarif.PhysicalLocation = Field(
                alias="physicalLocation", description="Physical location"
            )

        class Result(FlextModels.FrozenStrictModel):
            """SARIF result entry."""

            rule_id: str = Field(description="Rule identifier")
            level: str = Field(description="Result level (error/warning)")
            message: FlextInfraCheckModels.Sarif.MessageText = Field(
                description="Result message"
            )
            locations: list[FlextInfraCheckModels.Sarif.Location] = Field(
                description="Result locations"
            )

        class ToolDriver(FlextModels.FrozenStrictModel):
            """SARIF tool driver metadata."""

            name: str = Field(description="Tool name")
            information_uri: str = Field(
                default="",
                description="Tool documentation URL",
            )
            rules: list[FlextInfraCheckModels.Sarif.RuleDescriptor] = Field(
                default_factory=list,
                description="Rule descriptors",
            )

        class Tool(FlextModels.FrozenStrictModel):
            """SARIF tool wrapper."""

            driver: FlextInfraCheckModels.Sarif.ToolDriver = Field(
                description="Tool driver"
            )

        class Run(FlextModels.FrozenStrictModel):
            """SARIF run entry."""

            tool: FlextInfraCheckModels.Sarif.Tool = Field(
                description="Tool information"
            )
            results: list[FlextInfraCheckModels.Sarif.Result] = Field(
                default_factory=list,
                description="Run results",
            )

        class Report(FlextModels.ArbitraryTypesModel):
            """Complete SARIF 2.1.0 report."""

            model_config = ConfigDict(extra="forbid", populate_by_name=True)

            schema_uri: str = Field(
                default="https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
                alias="$schema",
                description="SARIF schema URI",
            )
            version: str = Field(default="2.1.0", description="SARIF version")
            runs: list[FlextInfraCheckModels.Sarif.Run] = Field(
                default_factory=list,
                description="SARIF runs",
            )


__all__ = ["FlextInfraCheckModels"]
