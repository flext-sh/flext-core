"""qlty code-smell quality gate — smell findings as FLEXT architecture violations.

Every qlty smell type (identical/similar-code, function/file-complexity,
function-parameters, return-statements, nested-control-flow, boolean-logic)
is reported per project and ALSO emitted as a ``FlextSmellViolation`` warning on
every run — warnings fire for all findings, always, regardless of gate mode.
``c.Infra.SMELLS_GATE_MODE`` only decides pass/fail: WARN is report-only.
"""

from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import e as core_e
from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

# flext-0ftd.3.5: the empty package initializer is not a compatibility export;
# consume the declaration at its canonical owner after the lazy-init cutover.
from flext_infra.transformers.smells.base import smell_fixer_for
from flext_infra.transformers.smells.boolean_logic import FlextInfraBooleanLogicFixer

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraSmellsGate(FlextInfraGate):
    """Report qlty smells per project from one process-cached workspace scan.

    A single ``qlty smells --all`` scan covers the whole workspace so
    cross-project duplication clusters stay visible; per-project results are
    filtered by SARIF URI prefix. The scan output is cached per workspace
    root for the lifetime of the process (one scan per ``check run``).
    """

    gate_id: ClassVar[str] = "smells"
    gate_name: ClassVar[str] = "Code Smells"
    can_fix: ClassVar[bool] = True
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["smells"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["smells"][1]

    # flext-pulj: process results stay structural outside the Pydantic boundary.
    _scan_cache: ClassVar[dict[str, p.Cli.CommandOutput]] = {}

    @override
    def fix(self, project_dir: Path, ctx: m.Infra.GateContext) -> m.Infra.GateExecution:
        """Apply AST-based fixers for auto-fixable smell findings.

        Runs the same scan as ``check()``, then attempts a registered fixer
        for every issue whose code has ``auto=true`` in flext-core metadata.
        Only rewrites files when a fixer actually changes the source.
        """
        if ctx.check_only or not ctx.apply_fixes:
            return self._check_only_fix_result(project_dir)
        started = time.monotonic()
        scan = self._workspace_scan()
        issues = self._issues_from_sarif(scan.stdout or "{}", project_dir.name)
        if not issues and scan.exit_code != 0:
            issues = (self._tool_failure_issue(scan),)
        auto_issues = [issue for issue in issues if self._is_auto_fixable(issue)]
        changes: list[str] = []
        for issue in auto_issues:
            tag = c.Infra.SMELLS_RULE_TAGS.get(issue.code, "")
            fixer = (
                FlextInfraBooleanLogicFixer()
                if tag == FlextInfraBooleanLogicFixer.tag
                else smell_fixer_for(tag)
            )
            if fixer is None:
                continue
            fixed, fix_changes = fixer.fix(project_dir, issue)
            if fixed:
                changes.extend(fix_changes)
        for issue in issues:
            warnings.warn(issue.formatted, core_e.SmellViolation, stacklevel=2)
        return self._build_check_gate_execution(
            project_dir,
            passed=True,
            issues=issues,
            raw_output="\n".join(changes) if changes else scan.stderr,
            started=started,
            errors=changes,
        )

    @staticmethod
    def _is_auto_fixable(issue: m.Infra.Issue) -> bool:
        """Return True when flext-core marks this smell tag as auto-fixable."""
        tag = c.Infra.SMELLS_RULE_TAGS.get(issue.code, "")
        strategy = c.ENFORCEMENT_SMELL_FIX_STRATEGIES.get(tag)
        return bool(strategy and strategy.get("auto"))

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """One cached full-workspace qlty scan, filtered to ``project_dir``."""
        _ = ctx
        started = time.monotonic()
        scan = self._workspace_scan()
        issues = self._issues_from_sarif(scan.stdout or "{}", project_dir.name)
        if not issues and scan.exit_code != 0:
            issues = (self._tool_failure_issue(scan),)
        for issue in issues:
            warnings.warn(issue.formatted, core_e.SmellViolation, stacklevel=2)
        passed = c.Infra.SMELLS_GATE_MODE is c.Infra.GateMode.WARN or not issues
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=issues,
            raw_output=scan.stderr,
            started=started,
        )

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Full-workspace scan command (check() bypasses per-project dirs)."""
        _ = project_dir, ctx, check_dirs
        binary = self._resolve_binary()
        return [binary or c.Infra.QLTY_BINARY, *c.Infra.SMELLS_QLTY_ARGS]

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse SARIF stdout into per-project issues (check_files path)."""
        _ = ctx
        issues = self._issues_from_sarif(result.stdout or "{}", project_dir.name)
        passed = c.Infra.SMELLS_GATE_MODE is c.Infra.GateMode.WARN or not issues
        return passed, issues

    def _workspace_scan(self) -> p.Cli.CommandOutput:
        """Run the workspace scan once per process; a missing binary is VISIBLE."""
        key = str(self._repository_root)
        cached = self._scan_cache.get(key)
        if cached is not None:
            return cached
        binary = self._resolve_binary()
        if binary is None:
            fallback = Path.home() / c.Infra.QLTY_BINARY_FALLBACK_SUFFIX
            output = m.Cli.CommandOutput(
                stdout="",
                stderr=(
                    f"{c.Infra.QLTY_BINARY} binary not found on PATH nor at {fallback}"
                ),
                exit_code=1,
            )
        else:
            output = self._run(
                [binary, *c.Infra.SMELLS_QLTY_ARGS],
                self._repository_root,
                timeout=c.Infra.TIMEOUT_LONG,
            )
            if "No qlty config file found" in output.stderr:
                output = m.Cli.CommandOutput(stdout="{}", stderr="", exit_code=0)
        self._scan_cache[key] = output
        return output

    @staticmethod
    def _resolve_binary() -> str | None:
        """Locate qlty on PATH, else the user-local install; None when absent."""
        found = shutil.which(c.Infra.QLTY_BINARY)
        if isinstance(found, str):
            return found
        fallback = Path.home() / c.Infra.QLTY_BINARY_FALLBACK_SUFFIX
        return str(fallback) if fallback.is_file() else None

    def _tool_failure_issue(self, scan: p.Cli.CommandOutput) -> m.Infra.Issue:
        """Scanner absence/crash must never read as a clean pass."""
        return m.Infra.Issue(
            file=c.Infra.PYPROJECT_FILENAME,
            line=1,
            column=0,
            code=self.gate_id,
            message=scan.stderr or "qlty execution failed",
            severity=self._severity(),
        )

    @staticmethod
    def _severity() -> str:
        """WARNING while report-only; ERROR once SMELLS_GATE_MODE is STRICT."""
        if c.Infra.SMELLS_GATE_MODE is c.Infra.GateMode.STRICT:
            return str(c.Infra.GateSeverity.ERROR.value)
        return str(c.Infra.GateSeverity.WARNING.value)

    @classmethod
    def _issues_from_sarif(
        cls, sarif_json: str, project_name: str
    ) -> tuple[m.Infra.Issue, ...]:
        """Extract one Issue per smell finding inside ``project_name``.

        Pure function over a literal qlty SARIF payload (unit-testable, no
        subprocess) — same strategy as ``loc_cap._files_over_cap``.
        """
        parsed = u.Cli.json_parse(sarif_json or "{}")
        empty_json: t.JsonValue = {}
        data = u.Cli.json_as_mapping(parsed.unwrap() if parsed.success else empty_json)
        prefix = f"{project_name}/"
        return tuple(
            cls._issue_from_result(result, prefix)
            for run in u.Cli.json_deep_mapping_list(data, "runs")
            for result in u.Cli.json_deep_mapping_list(run, "results")
            if cls._result_uri(result).startswith(prefix)
        )

    @classmethod
    def _issue_from_result(cls, result: t.JsonMapping, prefix: str) -> m.Infra.Issue:
        """Map one SARIF result to an Issue enriched with the FLEXT fix text."""
        rule_id = u.Cli.json_pick_str(result, "ruleId")
        code = rule_id.removeprefix(c.Infra.SMELLS_RULE_PREFIX)
        physical = u.Cli.json_deep_mapping(
            cls._first_location(result), "physicalLocation"
        )
        sarif_text = u.Cli.json_pick_str(
            u.Cli.json_deep_mapping(result, "message"), "text"
        )
        return m.Infra.Issue(
            file=cls._result_uri(result).removeprefix(prefix),
            line=u.Cli.json_nested_int(physical, "region", "startLine", default=1),
            column=u.Cli.json_nested_int(physical, "region", "startColumn"),
            code=code,
            message=cls._enriched_message(code, sarif_text),
            severity=cls._severity(),
        )

    @classmethod
    def _result_uri(cls, result: t.JsonMapping) -> str:
        """Workspace-relative URI of the finding's first location."""
        uri: str = u.Cli.json_pick_str(
            u.Cli.json_deep_mapping(
                cls._first_location(result), "physicalLocation", "artifactLocation"
            ),
            "uri",
        )
        return uri

    @staticmethod
    def _first_location(result: t.JsonMapping) -> t.JsonMapping:
        """First SARIF location mapping (empty mapping when absent)."""
        locations = u.Cli.json_deep_mapping_list(result, "locations")
        return locations[0] if locations else {}

    @staticmethod
    def _enriched_message(code: str, sarif_text: str) -> str:
        """Append the flext-core (problem, fix) law text when the tag exists."""
        tag = c.Infra.SMELLS_RULE_TAGS.get(code, "")
        text = c.ENFORCEMENT_RULES_TEXT.get(tag) if tag else None
        if text is None:
            return sarif_text
        problem, fix = text
        return f"{sarif_text} — {problem}. Fix: {fix}"


__all__: list[str] = ["FlextInfraSmellsGate"]
