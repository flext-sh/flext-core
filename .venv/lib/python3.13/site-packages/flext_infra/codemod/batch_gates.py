"""Gate measurement and ast-grep batch execution for the mod safety circuit."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, Final

from flext_infra import c, m, p, r, t, u

_TOOL_TIMEOUT_SECONDS: Final[int] = 900


class FlextInfraModGateSnapshot(m.ArbitraryTypesModel):
    """Error-count snapshot of the two mod circuit gates."""

    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

    ruff_errors: Annotated[
        t.NonNegativeInt, m.Field(description="ruff check error count")
    ]
    pyrefly_errors: Annotated[
        t.NonNegativeInt, m.Field(description="pyrefly error count")
    ]


class FlextInfraModScanReport(m.ArbitraryTypesModel):
    """Verified actionable rewrite report."""

    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

    nodes: Annotated[t.NonNegativeInt, m.Field(description="actionable node count")]
    files: Annotated[
        frozenset[Path], m.Field(description="files containing actionable nodes")
    ]


class FlextInfraModGateEngine:
    """Measure ruff/pyrefly counts and execute the ast-grep rule batch."""

    @staticmethod
    def circuit_broken(
        baseline: FlextInfraModGateSnapshot, final: FlextInfraModGateSnapshot
    ) -> bool:
        """Return True when either gate error count increased after the apply."""
        return (
            final.ruff_errors > baseline.ruff_errors
            or final.pyrefly_errors > baseline.pyrefly_errors
        )

    @staticmethod
    def _run_tool(root: Path, command: t.StrSequence) -> p.Result[p.Cli.CommandOutput]:
        """Run one circuit tool and tolerate its findings exit codes."""
        run = u.Cli.run_raw(command, cwd=root, timeout=_TOOL_TIMEOUT_SECONDS)
        if run.failure:
            return r[p.Cli.CommandOutput].fail(
                run.error or f"tool execution failed: {command[0]}"
            )
        output = run.value
        if output.exit_code not in {0, 1}:
            detail = (output.stderr or output.stdout).strip()
            return r[p.Cli.CommandOutput].fail(
                detail or f"{command[0]} exited with code {output.exit_code}"
            )
        return r[p.Cli.CommandOutput].ok(output)

    @staticmethod
    def _count_json_lines(stdout: str) -> int:
        """Count JSON-per-line findings (ast-grep ``--json=stream`` output)."""
        count = 0
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if line and u.Cli.json_parse(line).success:
                count += 1
        return count

    @staticmethod
    def _fixable_rule_ids(rule: Path) -> p.Result[frozenset[str]]:
        documents = rule.read_text(encoding="utf-8").split("\n---")
        fixable_ids: set[str] = set()
        for raw_document in documents:
            if not any(
                line.strip() and not line.lstrip().startswith("#")
                for line in raw_document.splitlines()
            ):
                continue
            parsed = u.Cli.yaml_parse(raw_document)
            if parsed.failure:
                return r[frozenset[str]].fail(
                    parsed.error or f"invalid ast-grep rule document in {rule}"
                )
            rule_id = parsed.value.get("id")
            if not isinstance(rule_id, str) or not rule_id:
                return r[frozenset[str]].fail(
                    f"ast-grep rule document missing required id: {rule}"
                )
            if "fix" in parsed.value:
                fixable_ids.add(rule_id)
        return r[frozenset[str]].ok(frozenset(fixable_ids))

    @staticmethod
    def _actionable_findings(
        stdout: str, fixable_ids: frozenset[str]
    ) -> FlextInfraModScanReport:
        nodes = 0
        files: set[Path] = set()
        for raw_line in stdout.splitlines():
            parsed = u.Cli.json_parse(raw_line.strip())
            if parsed.failure or not isinstance(parsed.value, Mapping):
                continue
            finding = parsed.value
            if finding.get("ruleId") not in fixable_ids:
                continue
            text = finding.get("text")
            replacement = finding.get("replacement")
            file = finding.get("file")
            if not isinstance(text, str) or not isinstance(replacement, str):
                continue
            if text == replacement or not isinstance(file, str):
                continue
            nodes += 1
            files.add(Path(file))
        return FlextInfraModScanReport(nodes=nodes, files=frozenset(files))

    @classmethod
    def _count_tool_errors(cls, stdout: str) -> int:
        """Count error items from array, object-with-errors, or JSONL output."""
        parsed = u.Cli.json_parse(stdout.strip()) if stdout.strip() else None
        if parsed is not None and parsed.success:
            value = parsed.value
            if isinstance(value, list):
                return len(value)
            if isinstance(value, Mapping):
                errors = value.get("errors")
                return len(errors) if isinstance(errors, list) else 0
        return cls._count_json_lines(stdout)

    @classmethod
    def measure(cls, root: Path) -> p.Result[FlextInfraModGateSnapshot]:
        """Capture the ruff + pyrefly error counts for one project root."""
        ruff_run = cls._run_tool(
            root,
            (
                c.Infra.RUFF,
                c.Infra.VERB_CHECK,
                ".",
                "--no-fix",
                "--output-format",
                c.Infra.OUTPUT_JSON,
                "--quiet",
            ),
        )
        if ruff_run.failure:
            return r[FlextInfraModGateSnapshot].from_failure(ruff_run)
        pyrefly_run = cls._run_tool(
            root,
            (
                c.Infra.PYREFLY,
                c.Infra.CHECK,
                ".",
                "--config",
                c.Infra.PYPROJECT_FILENAME,
                "--python-interpreter-path",
                sys.executable,
                "--output-format",
                c.Infra.OUTPUT_JSON,
                "--summary=none",
            ),
        )
        if pyrefly_run.failure:
            return r[FlextInfraModGateSnapshot].from_failure(pyrefly_run)
        return r[FlextInfraModGateSnapshot].ok(
            FlextInfraModGateSnapshot(
                ruff_errors=cls._count_tool_errors(ruff_run.value.stdout or ""),
                pyrefly_errors=cls._count_tool_errors(pyrefly_run.value.stdout or ""),
            )
        )

    @staticmethod
    def _count_fixable_findings(stdout: str, fixable_ids: frozenset[str]) -> int:
        findings = 0
        for raw_line in stdout.splitlines():
            parsed = u.Cli.json_parse(raw_line.strip())
            if parsed.failure or not isinstance(parsed.value, Mapping):
                continue
            if parsed.value.get("ruleId") in fixable_ids:
                findings += 1
        return findings

    @classmethod
    def scan(
        cls, root: Path, rules: t.SequenceOf[Path], *, fix: bool
    ) -> p.Result[FlextInfraModScanReport]:
        """Scan or apply actionable rewrite documents."""
        nodes = 0
        files: set[Path] = set()
        for rule in rules:
            fixable = cls._fixable_rule_ids(rule)
            if fixable.failure:
                return r[FlextInfraModScanReport].from_failure(fixable)
            if not fixable.value:
                continue
            command: list[str] = [c.Infra.SG, c.Infra.SCAN, "--rule", str(rule)]
            command.extend(("--json=stream", "."))
            run = cls._run_tool(root, tuple(command))
            if run.failure:
                return r[FlextInfraModScanReport].from_failure(run)
            report = cls._actionable_findings(run.value.stdout or "", fixable.value)
            nodes += report.nodes
            files.update(report.files)
            if fix and report.nodes:
                apply_run = cls._run_tool(
                    root,
                    (
                        c.Infra.SG,
                        c.Infra.SCAN,
                        "--rule",
                        str(rule),
                        "--update-all",
                        ".",
                    ),
                )
                if apply_run.failure:
                    return r[FlextInfraModScanReport].from_failure(apply_run)
        return r[FlextInfraModScanReport].ok(
            FlextInfraModScanReport(nodes=nodes, files=frozenset(files))
        )


__all__: list[str] = [
    "FlextInfraModGateEngine",
    "FlextInfraModGateSnapshot",
    "FlextInfraModScanReport",
]
