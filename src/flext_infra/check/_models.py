"""Domain models for the check subpackage."""

from __future__ import annotations

from pydantic import ConfigDict, Field, computed_field, model_serializer

from flext_core import FlextModels
from flext_infra import c, t


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
            description="Gate result model",
        )
        issues: list[FlextInfraCheckModels.Issue] = Field(
            default_factory=lambda: list[FlextInfraCheckModels.Issue](),
            description="Detected issues",
        )
        raw_output: str = Field(default="", description="Raw tool output")

    class GateResult(FlextModels.ArbitraryTypesModel):
        """Result summary for a single quality gate execution."""

        gate: str = Field(min_length=1, description="Gate name")
        project: str = Field(min_length=1, description="Project name")
        passed: bool = Field(description="Gate execution status")
        errors: list[str] = Field(
            default_factory=lambda: list[str](),
            description="Gate error messages",
        )
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class CheckResult(GateResult):
        pass

    class ProjectResult(FlextModels.ArbitraryTypesModel):
        """Aggregated gate results for a single project."""

        project: str = Field(description="Project name")
        gates: dict[str, FlextInfraCheckModels.GateExecution] = Field(
            default_factory=lambda: dict[str, FlextInfraCheckModels.GateExecution](),
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

    class WorkspaceCheckReport(FlextModels.ArbitraryTypesModel):
        generated_at: str = Field(description="UTC timestamp for report generation")
        gates: list[str] = Field(
            default_factory=lambda: list[str](),
            description="Gates executed in this run",
        )
        projects: list[FlextInfraCheckModels.ProjectResult] = Field(
            default_factory=lambda: list[FlextInfraCheckModels.ProjectResult](),
            description="Per-project check results",
        )

    # -- SARIF 2.1.0 report models -----------------------------------------

    class Sarif:
        """SARIF 2.1.0 report models."""

        class Rule(FlextModels.FrozenStrictModel):
            """Compact SARIF rule descriptor."""

            id: str = Field(description="Rule identifier")
            short_description: str = Field(description="Rule short description")

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.ContainerValue]:
                return {
                    "id": self.id,
                    "shortDescription": {"text": self.short_description},
                }

        class Location(FlextModels.FrozenStrictModel):
            """Compact SARIF location source span."""

            uri: str = Field(description="Artifact URI")
            start_line: int = Field(description="Start line (1-based)")
            start_column: int = Field(description="Start column (1-based)")
            uri_base_id: str = Field(
                default="%SRCROOT%",
                description="URI base identifier",
            )

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.ContainerValue]:
                return {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self.uri,
                            "uriBaseId": self.uri_base_id,
                        },
                        "region": {
                            "startLine": self.start_line,
                            "startColumn": self.start_column,
                        },
                    },
                }

        class Result(FlextModels.FrozenStrictModel):
            """SARIF result entry."""

            rule_id: str = Field(description="Rule identifier")
            level: str = Field(description="Result level (error/warning)")
            message: str = Field(description="Result message")
            locations: list[FlextInfraCheckModels.Sarif.Location] = Field(
                description="Result locations",
            )

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.ContainerValue]:
                return {
                    "ruleId": self.rule_id,
                    "level": self.level,
                    "message": {"text": self.message},
                    "locations": [
                        location.model_dump(by_alias=True)
                        for location in self.locations
                    ],
                }

        class Run(FlextModels.FrozenStrictModel):
            """SARIF run entry."""

            tool_name: str = Field(description="Tool name")
            information_uri: str = Field(
                default="",
                description="Tool documentation URL",
            )
            rules: list[FlextInfraCheckModels.Sarif.Rule] = Field(
                default_factory=lambda: list[FlextInfraCheckModels.Sarif.Rule](),
                description="Rule descriptors",
            )
            results: list[FlextInfraCheckModels.Sarif.Result] = Field(
                default_factory=lambda: list[FlextInfraCheckModels.Sarif.Result](),
                description="Run results",
            )

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.ContainerValue]:
                return {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "informationUri": self.information_uri,
                            "rules": [
                                rule.model_dump(by_alias=True) for rule in self.rules
                            ],
                        },
                    },
                    "results": [
                        result.model_dump(by_alias=True) for result in self.results
                    ],
                }

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
                default_factory=lambda: list[FlextInfraCheckModels.Sarif.Run](),
                description="SARIF runs",
            )


__all__ = ["FlextInfraCheckModels"]
