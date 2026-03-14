"""Domain models for the check subpackage."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, computed_field, model_serializer

from flext_core import FlextModels
from flext_infra import c, t


class FlextInfraCheckModels:
    """Quality-gate check domain models."""

    class Issue(FlextModels.FrozenStrictModel):
        """Single issue reported by a quality gate tool."""

        file: Annotated[str, Field(description="Source file path")]
        line: Annotated[int, Field(description="Line number")]
        column: Annotated[int, Field(description="Column number")]
        code: Annotated[str, Field(description="Rule or error code")]
        message: Annotated[str, Field(description="Human-readable issue description")]
        severity: Annotated[
            str,
            Field(
                default=c.Infra.Toml.ERROR,
                description="Issue severity level",
            ),
        ] = c.Infra.Toml.ERROR

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

        result: Annotated[
            FlextInfraCheckModels.GateResult,
            Field(
                description="Gate result model",
            ),
        ]
        issues: Annotated[
            list[FlextInfraCheckModels.Issue],
            Field(
                default_factory=lambda: list[FlextInfraCheckModels.Issue](),
                description="Detected issues",
            ),
        ] = Field(default_factory=lambda: list[FlextInfraCheckModels.Issue]())
        raw_output: Annotated[str, Field(default="", description="Raw tool output")]

    class GateResult(FlextModels.ArbitraryTypesModel):
        """Result summary for a single quality gate execution."""

        gate: Annotated[str, Field(min_length=1, description="Gate name")]
        project: Annotated[str, Field(min_length=1, description="Project name")]
        passed: Annotated[bool, Field(description="Gate execution status")]
        errors: Annotated[
            list[str],
            Field(
                default_factory=lambda: list[str](),
                description="Gate error messages",
            ),
        ] = Field(default_factory=lambda: list[str]())
        duration: Annotated[
            float, Field(default=0.0, ge=0.0, description="Duration in seconds")
        ]

    class CheckResult(GateResult):
        pass

    class ProjectResult(FlextModels.ArbitraryTypesModel):
        """Aggregated gate results for a single project."""

        project: Annotated[str, Field(description="Project name")]
        gates: Annotated[
            dict[str, FlextInfraCheckModels.GateExecution],
            Field(
                default_factory=lambda: dict[
                    str, FlextInfraCheckModels.GateExecution
                ](),
                description="Gate name to execution mapping",
            ),
        ] = Field(
            default_factory=lambda: dict[str, FlextInfraCheckModels.GateExecution]()
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
        generated_at: Annotated[
            str, Field(description="UTC timestamp for report generation")
        ]
        gates: Annotated[
            list[str],
            Field(
                default_factory=lambda: list[str](),
                description="Gates executed in this run",
            ),
        ] = Field(default_factory=lambda: list[str]())
        projects: Annotated[
            list[FlextInfraCheckModels.ProjectResult],
            Field(
                default_factory=lambda: list[FlextInfraCheckModels.ProjectResult](),
                description="Per-project check results",
            ),
        ] = Field(default_factory=lambda: list[FlextInfraCheckModels.ProjectResult]())

    # -- SARIF 2.1.0 report models -----------------------------------------

    class Sarif:
        """SARIF 2.1.0 report models."""

        class Rule(FlextModels.FrozenStrictModel):
            """Compact SARIF rule descriptor."""

            id: Annotated[str, Field(description="Rule identifier")]
            short_description: Annotated[
                str, Field(description="Rule short description")
            ]

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.Infra.InfraValue]:
                return {
                    "id": self.id,
                    "shortDescription": {"text": self.short_description},
                }

        class Location(FlextModels.FrozenStrictModel):
            """Compact SARIF location source span."""

            uri: Annotated[str, Field(description="Artifact URI")]
            start_line: Annotated[int, Field(description="Start line (1-based)")]
            start_column: Annotated[int, Field(description="Start column (1-based)")]
            uri_base_id: Annotated[
                str,
                Field(
                    default="%SRCROOT%",
                    description="URI base identifier",
                ),
            ] = "%SRCROOT%"

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.Infra.InfraValue]:
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

            rule_id: Annotated[str, Field(description="Rule identifier")]
            level: Annotated[str, Field(description="Result level (error/warning)")]
            message: Annotated[str, Field(description="Result message")]
            locations: Annotated[
                list[FlextInfraCheckModels.Sarif.Location],
                Field(
                    description="Result locations",
                ),
            ]

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.Infra.InfraValue]:
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

            tool_name: Annotated[str, Field(description="Tool name")]
            information_uri: Annotated[
                str,
                Field(
                    default="",
                    description="Tool documentation URL",
                ),
            ] = ""
            rules: Annotated[
                list[FlextInfraCheckModels.Sarif.Rule],
                Field(
                    default_factory=lambda: list[FlextInfraCheckModels.Sarif.Rule](),
                    description="Rule descriptors",
                ),
            ] = Field(default_factory=lambda: list[FlextInfraCheckModels.Sarif.Rule]())
            results: Annotated[
                list[FlextInfraCheckModels.Sarif.Result],
                Field(
                    default_factory=lambda: list[FlextInfraCheckModels.Sarif.Result](),
                    description="Run results",
                ),
            ] = Field(
                default_factory=lambda: list[FlextInfraCheckModels.Sarif.Result]()
            )

            @model_serializer(mode="plain")
            def _serialize(self) -> dict[str, t.Infra.InfraValue]:
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

            schema_uri: Annotated[
                str,
                Field(
                    default="https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
                    alias="$schema",
                    description="SARIF schema URI",
                ),
            ] = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json"
            version: Annotated[
                str, Field(default="2.1.0", description="SARIF version")
            ] = "2.1.0"
            runs: Annotated[
                list[FlextInfraCheckModels.Sarif.Run],
                Field(
                    default_factory=lambda: list[FlextInfraCheckModels.Sarif.Run](),
                    description="SARIF runs",
                ),
            ] = Field(default_factory=lambda: list[FlextInfraCheckModels.Sarif.Run]())


__all__ = ["FlextInfraCheckModels"]
