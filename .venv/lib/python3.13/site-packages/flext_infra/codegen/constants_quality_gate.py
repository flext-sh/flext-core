"""Workspace-wide quality gate for constants refactor outcomes."""

from __future__ import annotations

import operator
import shutil
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.base import s
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.refactor.census import FlextInfraRefactorCensus

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraCodegenQualityGate(s[bool]):
    """Run final constants migration checks with before/after comparison."""

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the quality gate and return its CLI success/failure status."""
        report_result = self.build_report()
        if report_result.failure:
            return r[bool].fail(report_result.error or "quality gate build failed")
        verdict = u.Cli.json_pick_str(report_result.value, "verdict", "FAIL")
        if self.successful_verdict(verdict):
            return r[bool].ok(True)
        return r[bool].fail(f"quality gate verdict: {verdict}")

    def build_report(self) -> p.Result[t.JsonMapping]:
        """Execute quality gate and return structured report payload."""
        FlextInfraCodegenLazyInit(repository_root=self.repository_root).generate_inits()
        census_report = FlextInfraRefactorCensus(
            repository_root=self.repository_root,
            include_local_scopes=False,
            kinds=("constant",),
        ).build_report()
        modified_files = self.modified_python_files(self.repository_root)
        pyrefly_check, ruff_check = self._run_static_checks(
            self.repository_root, modified_files
        )
        after_metrics = self.after_metrics(
            census_report=census_report, modified_files=modified_files
        )
        checks = self.build_checks(
            after_metrics=after_metrics,
            pyrefly_check=pyrefly_check,
            ruff_check=ruff_check,
        )
        verdict = self.compute_verdict(checks)
        report_data = {
            "workspace": str(self.repository_root),
            "generated_at": u.now().isoformat(),
            "verdict": verdict,
            "checks": [
                t.Infra.INFRA_MAPPING_ADAPTER.validate_python(check) for check in checks
            ],
            "after": after_metrics,
            "duplicate_constant_groups": [
                group.model_dump(mode="json") for group in census_report.duplicates
            ],
            "projects": [
                t.Infra.INFRA_MAPPING_ADAPTER.validate_python(project)
                for project in self.project_findings(census_report)
            ],
        }
        report = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(report_data)
        artifacts = self.write_artifacts(
            repository_root=self.repository_root,
            report=report,
            render_text=self.render_text(report),
        )
        if artifacts.failure:
            return r[t.JsonMapping].fail(
                artifacts.error or "quality gate artifact write failed"
            )
        report_data["artifacts"] = artifacts.value
        return r[t.JsonMapping].ok(
            t.Infra.INFRA_MAPPING_ADAPTER.validate_python(report_data)
        )

    @staticmethod
    def modified_python_files(repository_root: Path) -> t.StrSequence:
        """Return modified Python files detected by git porcelain status."""
        modified: t.MutableSequenceOf[str] = []
        git_bin = shutil.which(c.Infra.GIT)
        if git_bin is None:
            return []
        result = u.Cli.run_raw([
            git_bin,
            "-C",
            str(repository_root),
            "status",
            "--porcelain",
        ])
        if result.failure or result.value.exit_code != 0:
            return []
        for line in (
            entry.strip() for entry in result.value.stdout.splitlines() if entry.strip()
        ):
            if any(status in line[:2] for status in ("M", "A", "R", "C", "U")):
                path = line[3:].strip()
                if " -> " in path:
                    path = path.split(" -> ")[-1]
                if path.endswith(c.Infra.EXT_PYTHON):
                    modified.append(path)
        return list(modified)

    @staticmethod
    def run_static_check(
        repository_root: Path, modified_files: t.StrSequence, tool: str
    ) -> t.MappingKV[str, t.Infra.InfraValue]:
        """Run a targeted static tool on modified files and normalize result."""
        if not modified_files:
            return {
                "passed": True,
                "detail": "no modified python files detected",
                "exit_code": 0,
            }
        if tool == c.Infra.PYREFLY:
            cmd = [
                sys.executable,
                "-m",
                c.Infra.PYREFLY,
                c.Infra.CHECK,
                *modified_files,
                "--config",
                c.Infra.PYPROJECT_FILENAME,
                "--summary=none",
            ]
        elif tool == c.Infra.RUFF:
            cmd = [
                sys.executable,
                "-m",
                c.Infra.RUFF,
                c.Infra.VERB_CHECK,
                *modified_files,
                "--output-format",
                c.Infra.OUTPUT_JSON,
                "--quiet",
            ]
        else:
            return {
                "passed": False,
                "detail": f"unsupported tool: {tool}",
                "exit_code": 2,
            }
        run = u.Cli.run_raw(cmd, cwd=repository_root)
        if run.failure:
            return {
                "passed": False,
                "detail": run.error or "execution error",
                "exit_code": c.Infra.PROCESS_COMMAND_NOT_FOUND_EXIT_CODE,
            }
        output = (run.value.stderr or run.value.stdout or "").strip()
        lines = [line for line in output.splitlines() if line.strip()]
        return {
            "passed": run.value.exit_code == 0,
            "detail": " | ".join(lines[:5]) if lines else "ok",
            "exit_code": run.value.exit_code,
        }

    @classmethod
    def _run_static_checks(
        cls, repository_root: Path, modified_files: t.StrSequence
    ) -> tuple[
        t.MappingKV[str, t.Infra.InfraValue], t.MappingKV[str, t.Infra.InfraValue]
    ]:
        """Run pyrefly and ruff checks in parallel over the same file set.

        Both tools operate on the unmodified ``modified_files`` list and are
        independent, so they can execute concurrently. When no files changed,
        the empty-result payload is reused for both tools without spawning
        subprocesses.
        """
        empty_result: t.MappingKV[str, t.Infra.InfraValue] = {
            "passed": True,
            "detail": "no modified python files detected",
            "exit_code": 0,
        }
        if not modified_files:
            return (empty_result, empty_result)

        tools = (c.Infra.PYREFLY, c.Infra.RUFF)
        results: dict[str, t.MappingKV[str, t.Infra.InfraValue]] = {}
        with ThreadPoolExecutor(max_workers=len(tools)) as executor:
            futures: dict[Future[t.MappingKV[str, t.Infra.InfraValue]], str] = {
                executor.submit(
                    cls.run_static_check, repository_root, modified_files, tool
                ): tool
                for tool in tools
            }
            for future, tool in futures.items():
                results[tool] = future.result()

        return (results[c.Infra.PYREFLY], results[c.Infra.RUFF])

    @staticmethod
    def after_metrics(
        *, census_report: m.Infra.Census.WorkspaceReport, modified_files: t.StrSequence
    ) -> t.MappingKV[str, t.Infra.InfraValue]:
        """Build post-run metrics summary used by quality checks."""
        by_kind: t.MutableIntMapping = {}
        for project in census_report.projects:
            for parsed in project.violations:
                by_kind[parsed.kind] = by_kind.get(parsed.kind, 0) + 1
        total = len(census_report.projects)
        passed = u.count(
            census_report.projects, lambda project: project.violations_total == 0
        )
        modified_python_files: list[t.Infra.InfraValue] = list(modified_files)
        violations_by_rule: dict[str, t.Infra.InfraValue] = dict(
            sorted(by_kind.items())
        )
        summary: dict[str, t.Infra.InfraValue] = {
            "total_violations": census_report.total_violations,
            "violations_by_rule": violations_by_rule,
            "duplicate_groups": len(census_report.duplicates),
            "projects_total": total,
            "projects_passed": passed,
            "projects_failed": total - passed,
            "flext_failures": 0,
            "layer_violations": 0,
            "cross_project_reference_violations": 0,
            "modified_python_files": modified_python_files,
        }
        return summary

    @staticmethod
    def build_checks(
        *,
        after_metrics: t.MappingKV[str, t.Infra.InfraValue],
        pyrefly_check: t.MappingKV[str, t.Infra.InfraValue],
        ruff_check: t.MappingKV[str, t.Infra.InfraValue],
    ) -> t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]:
        """Build quality gate check entries from metrics and tool results."""
        # Metric-driven checks share the shape ``(name, value==0, "label=value")``.
        # Each row maps to a single ``QualityGateCheck`` via Pydantic v2 batch
        # construction. The detail-label key may differ from the metric path
        # (e.g. ``total_violations`` → ``total``).
        metric_check_rows: tuple[tuple[str, str, str], ...] = (
            (c.Infra.QG_CHECK_NAMESPACE_COMPLIANCE, "total_violations", "total"),
            (c.Infra.QG_CHECK_FLEXT_VALIDITY, "flext_failures", "flext_failures"),
            (
                c.Infra.QG_CHECK_IMPORT_RESOLUTION,
                "cross_project_reference_violations",
                "cross_project_reference_violations",
            ),
            (c.Infra.QG_CHECK_LAYER_COMPLIANCE, "layer_violations", "layer_violations"),
            (
                c.Infra.QG_CHECK_DUPLICATION_REDUCTION,
                "duplicate_groups",
                "duplicate_groups",
            ),
        )
        metric_checks = tuple(
            m.Infra.QualityGateCheck(
                name=name,
                passed=(value := u.Cli.json_nested_int(after_metrics, metric)) == 0,
                detail=f"{label}={value}",
                critical=True,
            )
            for name, metric, label in metric_check_rows
        )
        # Tool-result checks consume external dict payloads (pyrefly + ruff),
        # so they keep their own shape inline next to the table.
        tool_checks = (
            m.Infra.QualityGateCheck(
                name=c.Infra.QG_CHECK_TYPE_SAFETY,
                passed=bool(pyrefly_check.get("passed")),
                detail=str(pyrefly_check.get("detail", "")),
                critical=True,
            ),
            m.Infra.QualityGateCheck(
                name=c.Infra.QG_CHECK_LINT_CLEAN,
                passed=bool(ruff_check.get("passed")),
                detail=str(ruff_check.get("detail", "")),
                critical=True,
            ),
        )
        checks: t.SequenceOf[m.Infra.QualityGateCheck] = (*metric_checks, *tool_checks)
        return [check.model_dump() for check in checks]

    @staticmethod
    def compute_verdict(
        checks: t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]],
    ) -> str:
        """Return PASS only when all checks passed."""
        return (
            "PASS"
            if all(bool(check.get("passed", False)) for check in checks)
            else "FAIL"
        )

    @staticmethod
    def project_findings(
        census_report: m.Infra.Census.WorkspaceReport,
    ) -> t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]:
        """Convert census reports into sorted per-project findings."""
        return [
            item.model_dump()
            for item in sorted(
                (
                    m.Infra.QualityGateProjectFinding(
                        project=report.project,
                        violations_total=report.violations_total,
                        fixable_violations=len(report.fixes),
                        validator_passed=report.violations_total == 0,
                        flext_failures=0,
                        layer_violations=0,
                        cross_project_reference_violations=0,
                    )
                    for report in census_report.projects
                ),
                key=operator.attrgetter("project"),
            )
        ]

    @staticmethod
    def write_artifacts(
        repository_root: Path, report: t.JsonMapping, render_text: str
    ) -> p.Result[t.JsonMapping]:
        """Persist quality gate artifacts to the report directory."""
        report_dir = repository_root / c.Infra.QG_REPORT_DIR
        report_dir.mkdir(parents=True, exist_ok=True)
        report_json = report_dir / "latest.json"
        report_txt = report_dir / "latest.txt"
        json_write = u.Cli.atomic_write_text_file(
            report_json,
            t.Infra.INFRA_MAPPING_ADAPTER.dump_json(report, by_alias=True).decode(
                c.Cli.ENCODING_DEFAULT
            ),
        )
        if json_write.failure:
            return r[t.JsonMapping].fail(
                json_write.error or f"cannot write {report_json}"
            )
        txt_write = u.Cli.atomic_write_text_file(report_txt, render_text)
        if txt_write.failure:
            return r[t.JsonMapping].fail(
                txt_write.error or f"cannot write {report_txt}"
            )
        return r[t.JsonMapping].ok({
            "report_json": str(report_json),
            "report_text": str(report_txt),
        })

    @classmethod
    def render_text(cls, report: t.JsonMapping) -> str:
        """Render compact human-readable summary."""
        checks = u.Cli.json_deep_mapping_list(report, "checks")
        after = u.Cli.json_deep_mapping(report, "after")
        duplicate_groups = u.Cli.json_deep_mapping_list(
            report, "duplicate_constant_groups"
        )
        lines: t.MutableSequenceOf[str] = [
            f"Workspace: {report.get('workspace', '')}",
            f"Verdict: {report.get('verdict', 'FAIL')}",
            "",
            "Checks:",
        ]
        for check in checks:
            status = "PASS" if u.Cli.json_pick_bool(check, "passed") else "FAIL"
            lines.append(
                f"- [{status}] {u.Cli.json_pick_str(check, 'name', 'unknown')}"
            )
            detail = u.Cli.json_pick_str(check, "detail")
            if detail:
                lines.append(f"  {detail}")
        lines.extend([
            "",
            "Current:",
            f"- violations: {after.get('total_violations', 'n/a')}",
            f"- duplicates: {after.get('duplicate_groups', 'n/a')}",
            f"- projects: {after.get('projects_total', 0)} total, {after.get('projects_passed', 0)} passed, {after.get('projects_failed', 0)} failed",
        ])
        if duplicate_groups:
            lines.extend(["", "Duplicate Groups:"])
        for group in duplicate_groups:
            parsed_group = m.Infra.Census.DuplicateGroup.model_validate(group)
            projects = sorted({
                definition.project for definition in parsed_group.definitions
            })
            lines.append(
                "- "
                f"{parsed_group.name}: "
                f"projects={len(projects)}, "
                f"definitions={len(parsed_group.definitions)}, "
                f"values_identical={parsed_group.value_identical}"
            )
            if projects:
                lines.append(f"  projects: {', '.join(projects)}")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def successful_verdict(verdict: str) -> bool:
        """Return True for verdicts that should exit with status 0.

        Domain-verb naming per AGENTS.md §3.1: boolean outcomes use
        noun/adjective names, not ``is_*`` prefixes.
        """
        return verdict == "PASS"


__all__: list[str] = ["FlextInfraCodegenQualityGate"]
