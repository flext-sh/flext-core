"""Domain models for the check subpackage."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated, ClassVar

from flext_core import m
from flext_core import u
from flext_infra import c, t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsCheck:
    """Quality-gate check domain models."""

    class RunCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra check run``.

        Inherits canonical ``gates`` (parsed to ``t.StrSequence``),
        ``apply``/``dry_run``, ``workspace``, ``projects``, ``fail_fast``,
        ``verbose`` from ``WriteMixin``.
        """

        reports_dir: Annotated[
            str,
            m.Field(
                alias="reports-dir", description="Directory used to write check reports"
            ),
        ] = f"{c.Infra.REPORTS_DIR_NAME}/check"
        fix: Annotated[
            bool, m.Field(False, description="Apply supported gate fixes before run")
        ] = False
        check_only: Annotated[
            bool,
            m.Field(
                alias="check-only",
                description="Enable check-only mode for supported tools",
            ),
        ] = False
        ruff_args: Annotated[
            str | None,
            m.Field(alias="ruff-args", description="Extra arguments forwarded to Ruff"),
        ] = None
        pyright_args: Annotated[
            str | None,
            m.Field(
                alias="pyright-args", description="Extra arguments forwarded to Pyright"
            ),
        ] = None

        @property
        def reports_dir_path(self) -> Path:
            """Resolved reports directory path."""
            reports_dir = Path(self.reports_dir).expanduser()
            if reports_dir.is_absolute():
                return reports_dir.resolve()
            return (Path.cwd() / reports_dir).resolve()

    class CheckProjectTarget(m.ArbitraryTypesModel):
        """Resolved project target for workspace gate execution."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True, validate_default=False
        )

        name: Annotated[str, m.Field(description="Display/project name")]
        path: Annotated[Path, m.Field(description="Resolved project root path")]

        @classmethod
        def from_workspace_name(
            cls, repository_root: Path, project_name: str
        ) -> FlextInfraModelsCheck.CheckProjectTarget:
            """Build a target from the public run_projects name contract."""
            return cls(name=project_name, path=repository_root / project_name)

    class MypyResourceLimit(m.ContractModel):
        """Validated memory and wall-time limits for every Mypy process."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        memory_limit_mb: Annotated[
            int,
            m.Field(
                gt=0,
                le=c.Infra.MYPY_MEMORY_LIMIT_MB_DEFAULT,
                description="Positive Mypy address-space limit in MiB",
            ),
        ] = c.Infra.MYPY_MEMORY_LIMIT_MB_DEFAULT
        timeout_seconds: Annotated[
            int,
            m.Field(
                gt=0,
                le=c.Infra.MYPY_TIMEOUT_SECONDS_DEFAULT,
                description="Positive Mypy wall-time limit in seconds",
            ),
        ] = c.Infra.MYPY_TIMEOUT_SECONDS_DEFAULT

        @m.computed_field
        @property
        def memory_limit_bytes(self) -> int:
            """Validated limit converted to bytes for prlimit."""
            return self.memory_limit_mb * 1024 * 1024

    class FixPyreflyConfigCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra check fix-pyrefly-settings``."""

    class FixEnforcementCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra check fix-enforcement``."""

        rules: Annotated[
            t.StrSequence,
            m.Field(description="Comma-separated enforcement rule IDs to fix"),
        ] = ()
        safe_only: Annotated[
            bool,
            m.Field(
                alias="safe-only",
                description="Only apply fixes marked safe in the catalog",
            ),
        ] = True
        check_after: Annotated[
            bool,
            m.Field(
                alias="check-after",
                description="Re-run the corresponding check after fixing",
            ),
        ] = True

        @m.field_validator("rules", mode="before")
        @classmethod
        def _parse_rules(cls, value: str | t.SequenceOf[str] | None) -> t.StrSequence:
            """Accept CSV string, sequence, or None; normalize to StrSequence."""
            if value is None:
                return ()
            if isinstance(value, str):
                return tuple(part.strip() for part in value.split(",") if part.strip())
            normalized: list[str] = []
            for part in value:
                if not part:
                    continue
                normalized.extend(
                    token.strip() for token in part.split(",") if token.strip()
                )
            return tuple(normalized)

        @m.field_validator("projects", mode="before")
        @classmethod
        def _parse_projects(
            cls, value: str | t.SequenceOf[str] | None
        ) -> t.StrSequence | None:
            """Accept CSV string, sequence, or None; normalize to StrSequence."""
            if value is None:
                return None
            if isinstance(value, str):
                return tuple(part.strip() for part in value.split(",") if part.strip())
            normalized: list[str] = []
            for part in value:
                if not part:
                    continue
                normalized.extend(
                    token.strip() for token in part.split(",") if token.strip()
                )
            return tuple(normalized) or None

    class Issue(m.ContractModel):
        """Single issue reported by a quality gate tool."""

        file: Annotated[str, m.Field(description="Source file path")]
        line: Annotated[int, m.Field(description="Line number")]
        column: Annotated[int, m.Field(description="Column number")]
        code: Annotated[str, m.Field(description="Rule or error code")]
        message: Annotated[str, m.Field(description="Human-readable issue description")]
        severity: Annotated[str, m.Field(description="Issue severity level")] = (
            c.Infra.ERROR
        )

        @m.computed_field
        @property
        def formatted(self) -> str:
            """Format issue as ``file:line:col [code] message``."""
            code_part = f"[{self.code}] " if self.code else ""
            return (
                f"{self.file}:{self.line}:{self.column} {code_part}{self.message}"
            ).strip()

    class GateResult(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Result summary for a single quality gate execution."""

        gate: Annotated[str, m.Field(description="Gate name")]
        passed: Annotated[bool, m.Field(description="Gate execution status")]
        errors: t.StrSequence = m.Field(
            default_factory=tuple, description="Gate error messages"
        )
        duration: float = m.Field(
            0.0, description="Duration in seconds", validate_default=True
        )

    class GateExecution(m.ArbitraryTypesModel):
        """Execution result for a single quality gate."""

        result: FlextInfraModelsCheck.GateResult = m.Field(
            description="Gate result model"
        )
        issues: tuple[FlextInfraModelsCheck.Issue, ...] = m.Field(
            default_factory=tuple, description="Detected issues"
        )
        raw_output: str = m.Field(
            "", description="Raw tool output", validate_default=True
        )

        @m.computed_field
        @property
        def error_count(self) -> int:
            """Number of diagnostics with error severity."""
            return sum(
                1 for issue in self.issues if issue.severity.lower() == c.Infra.ERROR
            )

    class ProjectResult(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Aggregated gate results for a single project.

        Enforcement exemption: ``gates`` is a ``MutableMapping`` populated
        incrementally as each gate completes; no shared state — one fresh
        dict per instance.
        """

        gates: MutableMapping[str, FlextInfraModelsCheck.GateExecution] = m.Field(
            default_factory=dict, description="Gate name to execution mapping"
        )

        @m.computed_field
        @property
        def passed(self) -> bool:
            """Whether every gate passed."""
            return all(v.result.passed for v in self.gates.values())

        @m.computed_field
        @property
        def total_errors(self) -> int:
            """Total error-severity diagnostic count across all gates."""
            return sum(v.error_count for v in self.gates.values())

    class LoopOutcome(m.ArbitraryTypesModel):
        """Bundled results from the project-checking loop."""

        # Why: owned by m.Infra; ArbitraryTypesModel keeps protocol field writability.

        results: Annotated[
            tuple[FlextInfraModelsCheck.ProjectResult, ...],
            m.Field(description="Individual project execution results."),
        ]
        failed: Annotated[
            int,
            m.Field(description="Number of projects that failed one or more gates."),
        ]
        skipped: Annotated[
            int,
            m.Field(
                description="Number of projects that were skipped during execution."
            ),
        ]
        total_elapsed: Annotated[
            float,
            m.Field(description="Total time elapsed in seconds for the entire loop."),
        ]

    # -- SARIF 2.1.0 report models -----------------------------------------

    class SarifRule(m.ContractModel):
        """Compact SARIF rule descriptor."""

        id: Annotated[str, m.Field(description="Rule identifier")]
        short_description: Annotated[str, m.Field(description="Rule short description")]

        @u.model_serializer
        def _serialize(self) -> t.JsonMapping:
            """Serialize."""
            return {"id": self.id, "shortDescription": {"text": self.short_description}}

    class SarifLocation(m.ContractModel):
        """Compact SARIF location source span."""

        uri: Annotated[str, m.Field(description="Artifact URI")]
        start_line: Annotated[int, m.Field(description="Start line (1-based)")]
        start_column: Annotated[int, m.Field(description="Start column (1-based)")]
        uri_base_id: str = m.Field(
            "%SRCROOT%", description="URI base identifier", validate_default=True
        )

        @u.model_serializer
        def _serialize(self) -> t.JsonMapping:
            """Serialize."""
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
                }
            }

    class SarifResult(m.ContractModel):
        """SARIF result entry."""

        rule_id: Annotated[str, m.Field(description="Rule identifier")]
        level: Annotated[str, m.Field(description="Result level (error/warning)")]
        message: Annotated[str, m.Field(description="Result message")]
        locations: list[FlextInfraModelsCheck.SarifLocation] = m.Field(
            description="Result locations"
        )

        @u.model_serializer
        def _serialize(self) -> t.JsonMapping:
            """Serialize."""
            return {
                "ruleId": self.rule_id,
                "level": self.level,
                "message": {"text": self.message},
                "locations": [
                    location.model_dump(by_alias=True) for location in self.locations
                ],
            }

    class SarifRun(m.ContractModel):
        """SARIF run entry."""

        tool_name: Annotated[str, m.Field(description="Tool name")]
        information_uri: str = m.Field(
            "", description="Tool documentation URL", validate_default=True
        )
        rules: tuple[FlextInfraModelsCheck.SarifRule, ...] = m.Field(
            default_factory=tuple, description="Rule descriptors"
        )
        results: tuple[FlextInfraModelsCheck.SarifResult, ...] = m.Field(
            default_factory=tuple, description="Run results"
        )

        @u.model_serializer
        def _serialize(self) -> t.JsonMapping:
            """Serialize."""
            return {
                "tool": {
                    "driver": {
                        "name": self.tool_name,
                        "informationUri": self.information_uri,
                        "rules": [
                            rule.model_dump(by_alias=True) for rule in self.rules
                        ],
                    }
                },
                "results": [
                    result.model_dump(by_alias=True) for result in self.results
                ],
            }

    class SarifReport(m.ArbitraryTypesModel):
        """Complete SARIF 2.1.0 report."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        schema_uri: str = m.Field(
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
            alias="$schema",
            description="SARIF schema URI",
            validate_default=True,
        )
        version: str = m.Field(
            "2.1.0", description="SARIF version", validate_default=True
        )
        runs: tuple[FlextInfraModelsCheck.SarifRun, ...] = m.Field(
            default_factory=tuple, description="SARIF runs"
        )


__all__: list[str] = ["FlextInfraModelsCheck"]
