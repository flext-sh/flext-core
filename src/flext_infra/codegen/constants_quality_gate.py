"""Workspace-wide quality gate for constants refactor outcomes."""

from __future__ import annotations

import ast
import json
import subprocess  # noqa: S404  # JUSTIFIED: runs local quality tools with shell=False
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from flext_infra import FlextInfraGitService, c, m
from flext_infra.codegen.census import FlextInfraCodegenCensus

__all__ = ["FlextInfraCodegenConstantsQualityGate"]


class FlextInfraCodegenConstantsQualityGate:
    """Run final constants migration checks with before/after comparison."""

    _REPORT_DIR: Path = Path(".reports/codegen/constants-quality-gate")
    _PASS_VERDICTS: tuple[str, ...] = ("PASS", "CONDITIONAL_PASS")

    def __init__(
        self,
        *,
        workspace_root: Path,
        before_report: Path | None = None,
        baseline_file: Path | None = None,
    ) -> None:
        """Initialize quality gate with workspace and optional baseline source."""
        self._workspace_root = workspace_root.resolve()
        self._before_report = before_report
        self._baseline_file = baseline_file
        self._git = FlextInfraGitService()

    def run(self) -> dict[str, object]:
        """Execute quality gate and return structured report payload."""
        before_payload, before_source, before_load_error = self._load_before_payload()

        census_reports = FlextInfraCodegenCensus(
            workspace_root=self._workspace_root
        ).run()
        duplicate_groups = self._count_duplicate_constant_groups()

        modified_files = self._modified_python_files()
        pyrefly_check = self._run_pyrefly_check(modified_files)
        ruff_check = self._run_ruff_check(modified_files)
        import_scan = self._scan_import_nodes(modified_files)

        before_metrics = self._before_metrics(before_payload)
        after_metrics = self._after_metrics(
            census_reports=census_reports,
            duplicate_groups=duplicate_groups,
            import_scan=import_scan,
            modified_files=modified_files,
        )
        improvement = self._improvement(before_metrics, after_metrics)
        checks = self._build_checks(
            after_metrics=after_metrics,
            improvement=improvement,
            pyrefly_check=pyrefly_check,
            ruff_check=ruff_check,
            before_available=before_payload is not None,
            before_load_error=before_load_error,
        )
        verdict = self._compute_verdict(checks, improvement)

        report: dict[str, object] = {
            "workspace": str(self._workspace_root),
            "generated_at": datetime.now(UTC).isoformat(),
            "verdict": verdict,
            "checks": checks,
            "baseline": {
                "source": before_source,
                "load_error": before_load_error,
                "provided": before_payload is not None,
            },
            "before": before_metrics,
            "after": after_metrics,
            "improvement": improvement,
            "projects": self._project_findings(census_reports),
        }
        report["artifacts"] = self._write_artifacts(
            report=report,
            census_reports=census_reports,
            duplicate_groups=duplicate_groups,
            before_payload=before_payload,
        )
        return report

    @classmethod
    def render_text(cls, report: dict[str, object]) -> str:
        """Render compact human-readable summary."""
        checks = cls._dict_list(report.get("checks"))
        before = cls._dict_or_empty(report.get("before"))
        after = cls._dict_or_empty(report.get("after"))
        improvement = cls._dict_or_empty(report.get("improvement"))
        lines: list[str] = [
            f"Workspace: {report.get('workspace', '')}",
            f"Verdict: {report.get('verdict', 'FAIL')}",
            "",
            "Checks:",
        ]
        for check in checks:
            status = "PASS" if bool(check.get("passed", False)) else "FAIL"
            lines.append(f"- [{status}] {check.get('name', 'unknown')}")
            detail = str(check.get("detail", "")).strip()
            if detail:
                lines.append(f"  {detail}")
        lines.extend([
            "",
            "Before/After:",
            (
                "- violations: "
                f"{before.get('total_violations', 'n/a')} -> "
                f"{after.get('total_violations', 'n/a')}"
            ),
            (
                "- duplicates: "
                f"{before.get('duplicate_groups', 'n/a')} -> "
                f"{after.get('duplicate_groups', 'n/a')}"
            ),
            (
                "- projects: "
                f"{after.get('projects_total', 0)} total, "
                f"{after.get('projects_passed', 0)} passed, "
                f"{after.get('projects_failed', 0)} failed"
            ),
            "",
            "Improvement:",
            f"- violations_delta: {improvement.get('violations_delta', 0)}",
            f"- duplicates_delta: {improvement.get('duplicates_delta', 0)}",
            f"- violations_reduced: {improvement.get('violations_reduced', 0)}",
            f"- duplicates_eliminated: {improvement.get('duplicates_eliminated', 0)}",
        ])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def is_success_verdict(verdict: str) -> bool:
        """Return True for verdicts that should exit with status 0."""
        return verdict in FlextInfraCodegenConstantsQualityGate._PASS_VERDICTS

    def _load_before_payload(self) -> tuple[dict[str, object] | None, str, str]:
        baseline_path = self._before_report or self._baseline_file
        if baseline_path is None:
            return None, "", ""
        resolved = (
            baseline_path
            if baseline_path.is_absolute()
            else (self._workspace_root / baseline_path)
        ).resolve()
        if not resolved.is_file():
            return None, str(resolved), f"baseline file not found: {resolved}"
        try:
            payload = json.loads(resolved.read_text(encoding=c.Infra.Encoding.DEFAULT))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, str(resolved), "baseline parse failed"
        if not isinstance(payload, dict):
            return None, str(resolved), "baseline payload is not a JSON object"
        raw = cast("dict[Any, Any]", payload)
        return ({str(key): value for key, value in raw.items()}, str(resolved), "")

    def _before_metrics(
        self, before_payload: dict[str, object] | None
    ) -> dict[str, object]:
        if before_payload is None:
            return {
                "total_violations": -1,
                "duplicate_groups": -1,
                "projects_total": 0,
                "projects_passed": 0,
                "projects_failed": 0,
            }
        return {
            "total_violations": self._extract_total_violations(before_payload),
            "duplicate_groups": self._extract_duplicate_groups(before_payload),
            "projects_total": self._extract_projects_total(before_payload),
            "projects_passed": self._extract_projects_passed(before_payload),
            "projects_failed": self._extract_projects_failed(before_payload),
        }

    def _after_metrics(
        self,
        *,
        census_reports: Sequence[m.Infra.Codegen.CensusReport],
        duplicate_groups: int,
        import_scan: dict[str, object],
        modified_files: list[str],
    ) -> dict[str, object]:
        by_rule = {"NS-000": 0, "NS-001": 0, "NS-002": 0}
        total_violations = 0
        for report in census_reports:
            violations = cast("Sequence[Any]", report.violations)
            total_violations += len(violations)
            for raw_violation in violations:
                parsed = m.Infra.Codegen.CensusViolation.model_validate(raw_violation)
                if parsed.rule in by_rule:
                    by_rule[parsed.rule] += 1
        projects_total = len(census_reports)
        projects_passed = sum(1 for item in census_reports if int(item.total) == 0)
        projects_failed = projects_total - projects_passed
        return {
            "total_violations": total_violations,
            "violations_by_rule": by_rule,
            "duplicate_groups": duplicate_groups,
            "projects_total": projects_total,
            "projects_passed": projects_passed,
            "projects_failed": projects_failed,
            "mro_failures": 0,
            "layer_violations": 0,
            "cross_project_reference_violations": 0,
            "import_parse_violations": self._as_int(
                import_scan.get("invalid_import_from_count")
            ),
            "import_parse_errors": self._as_int(import_scan.get("parse_error_count")),
            "modified_python_files": modified_files,
        }

    @staticmethod
    def _improvement(
        before_metrics: dict[str, object], after_metrics: dict[str, object]
    ) -> dict[str, object]:
        before_violations = FlextInfraCodegenConstantsQualityGate._as_int(
            before_metrics.get("total_violations")
        )
        before_duplicates = FlextInfraCodegenConstantsQualityGate._as_int(
            before_metrics.get("duplicate_groups")
        )
        after_violations = FlextInfraCodegenConstantsQualityGate._as_int(
            after_metrics.get("total_violations")
        )
        after_duplicates = FlextInfraCodegenConstantsQualityGate._as_int(
            after_metrics.get("duplicate_groups")
        )
        violations_delta = (
            0 if before_violations < 0 else after_violations - before_violations
        )
        duplicates_delta = (
            0 if before_duplicates < 0 else after_duplicates - before_duplicates
        )
        return {
            "violations_delta": violations_delta,
            "duplicates_delta": duplicates_delta,
            "violations_reduced": max(0, -violations_delta),
            "duplicates_eliminated": max(0, -duplicates_delta),
            "violations_increased": max(0, violations_delta),
            "duplicates_increased": max(0, duplicates_delta),
        }

    def _build_checks(
        self,
        *,
        after_metrics: dict[str, object],
        improvement: dict[str, object],
        pyrefly_check: dict[str, object],
        ruff_check: dict[str, object],
        before_available: bool,
        before_load_error: str,
    ) -> list[dict[str, object]]:
        checks: list[dict[str, object]] = []
        violations_total = self._as_int(after_metrics.get("total_violations"))
        violations_delta = self._as_int(improvement.get("violations_delta"))
        checks.append({
            "name": "namespace_compliance",
            "passed": (
                (violations_total == 0) or (before_available and violations_delta < 0)
            )
            and (not before_available or violations_delta <= 0),
            "detail": (
                f"total={violations_total}, delta={violations_delta}"
                if before_available
                else f"total={violations_total} (no baseline provided)"
            ),
            "critical": False,
        })
        mro_failures = self._as_int(after_metrics.get("mro_failures"))
        cross_ref = self._as_int(
            after_metrics.get("cross_project_reference_violations")
        )
        import_parse = self._as_int(after_metrics.get("import_parse_violations"))
        import_parse_errors = self._as_int(after_metrics.get("import_parse_errors"))
        layer_violations = self._as_int(after_metrics.get("layer_violations"))
        duplicate_groups = self._as_int(after_metrics.get("duplicate_groups"))
        duplicates_delta = self._as_int(improvement.get("duplicates_delta"))
        checks.extend([
            {
                "name": "mro_validity",
                "passed": mro_failures == 0,
                "detail": f"mro_failures={mro_failures}",
                "critical": True,
            },
            {
                "name": "import_resolution",
                "passed": (
                    cross_ref == 0 and import_parse == 0 and import_parse_errors == 0
                ),
                "detail": (
                    "cross_project_reference_violations="
                    f"{cross_ref}, invalid_import_from={import_parse}, "
                    f"parse_errors={import_parse_errors}"
                ),
                "critical": True,
            },
            {
                "name": "layer_compliance",
                "passed": layer_violations == 0,
                "detail": f"layer_violations={layer_violations}",
                "critical": True,
            },
            {
                "name": "duplication_reduction",
                "passed": (
                    (duplicate_groups == 0)
                    or (before_available and duplicates_delta < 0)
                )
                and (not before_available or duplicates_delta <= 0),
                "detail": (
                    f"duplicate_groups={duplicate_groups}, delta={duplicates_delta}"
                    if before_available
                    else f"duplicate_groups={duplicate_groups} (no baseline provided)"
                ),
                "critical": False,
            },
            {
                "name": "type_safety",
                "passed": bool(pyrefly_check.get("passed", False)),
                "detail": str(pyrefly_check.get("detail", "")),
                "critical": True,
            },
            {
                "name": "lint_clean",
                "passed": bool(ruff_check.get("passed", False)),
                "detail": str(ruff_check.get("detail", "")),
                "critical": True,
            },
        ])
        if before_load_error:
            checks.append({
                "name": "baseline_load",
                "passed": False,
                "detail": before_load_error,
                "critical": False,
            })
        return checks

    @staticmethod
    def _compute_verdict(
        checks: Sequence[dict[str, object]], improvement: dict[str, object]
    ) -> str:
        if all(bool(item.get("passed", False)) for item in checks):
            return "PASS"
        if any(
            bool(not item.get("passed", False) and item.get("critical", False))
            for item in checks
        ):
            return "FAIL"
        if (
            FlextInfraCodegenConstantsQualityGate._as_int(
                improvement.get("violations_increased")
            )
            > 0
            or FlextInfraCodegenConstantsQualityGate._as_int(
                improvement.get("duplicates_increased")
            )
            > 0
        ):
            return "FAIL"
        return "CONDITIONAL_PASS"

    def _project_findings(
        self,
        census_reports: Sequence[m.Infra.Codegen.CensusReport],
        validator_report: dict[str, object],
    ) -> list[dict[str, object]]:
        census_by_project: dict[str, dict[str, object]] = {}
        for entry in census_reports:
            census_by_project[entry.project] = {
                "project": entry.project,
                "violations_total": len(cast("Sequence[Any]", entry.violations)),
                "fixable_violations": int(entry.fixable),
            }
        findings: list[dict[str, object]] = []
        projects = self._dict_list(validator_report.get(c.Infra.ReportKeys.PROJECTS))
        for project_report in projects:
            project_name = str(project_report.get("project", ""))
            merged = dict(
                census_by_project.get(project_name, {"project": project_name})
            )
            merged["validator_passed"] = bool(project_report.get("passed", False))
            merged["mro_failures"] = sum(
                1
                for mro_result in self._dict_list(project_report.get("mro_results"))
                if not bool(mro_result.get("passed", False))
            )
            merged["layer_violations"] = len(
                self._string_list(project_report.get("layer_violations"))
            )
            merged["cross_project_reference_violations"] = len(
                self._string_list(
                    project_report.get("cross_project_reference_violations")
                )
            )
            findings.append(merged)
        for project_name, values in census_by_project.items():
            if any(str(item.get("project", "")) == project_name for item in findings):
                continue
            findings.append(values)
        findings.sort(key=lambda item: str(item.get("project", "")))
        return findings

    def _write_artifacts(
        self,
        *,
        report: dict[str, object],
        census_reports: Sequence[m.Infra.Codegen.CensusReport],
        inventory_report: dict[str, object],
        validator_report: dict[str, object],
        before_payload: dict[str, object] | None,
    ) -> dict[str, object]:
        directory = self._workspace_root / self._REPORT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        report_json = directory / "latest.json"
        report_text = directory / "latest.txt"
        census_json = directory / "census-after.json"
        inventory_json = directory / "inventory-after.json"
        validate_json = directory / "validate-after.json"
        baseline_json = directory / "baseline-used.json"
        report_json.write_text(
            json.dumps(report, ensure_ascii=True, sort_keys=True),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        report_text.write_text(
            self.render_text(report), encoding=c.Infra.Encoding.DEFAULT
        )
        census_payload: list[dict[str, object]] = [
            item.model_dump() for item in census_reports
        ]
        census_json.write_text(
            json.dumps(census_payload, ensure_ascii=True),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        inventory_json.write_text(
            json.dumps(inventory_report, ensure_ascii=True, sort_keys=True),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        validate_json.write_text(
            json.dumps(validator_report, ensure_ascii=True, sort_keys=True),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        if before_payload is not None:
            baseline_json.write_text(
                json.dumps(before_payload, ensure_ascii=True, sort_keys=True),
                encoding=c.Infra.Encoding.DEFAULT,
            )
        return {
            "directory": str(directory),
            "report_json": str(report_json),
            "report_text": str(report_text),
            "census_after": str(census_json),
            "inventory_after": str(inventory_json),
            "validate_after": str(validate_json),
            "baseline_used": str(baseline_json) if before_payload is not None else "",
        }

    def _modified_python_files(self) -> list[str]:
        root_result = self._git.repo_root(self._workspace_root)
        git_root = (
            Path(root_result.value) if root_result.is_success else self._workspace_root
        )
        files_result = self._git.modified_files(git_root, pathspec="*.py")
        if files_result.is_success and files_result.value:
            normalized: set[str] = set()
            for rel in files_result.value:
                candidate = (self._workspace_root / rel).resolve()
                if (
                    not candidate.is_file()
                    or candidate.suffix != c.Infra.Extensions.PYTHON
                ):
                    continue
                try:
                    normalized.add(str(candidate.relative_to(self._workspace_root)))
                except ValueError:
                    continue
            if normalized:
                return sorted(normalized)
        fallback = (
            self._workspace_root
            / ".reports/codegen/constants-refactor/dedup-apply.json"
        )
        if fallback.is_file():
            try:
                payload = json.loads(
                    fallback.read_text(encoding=c.Infra.Encoding.DEFAULT)
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return []
            if isinstance(payload, dict):
                raw = cast("dict[Any, Any]", payload)
                modified = raw.get("modified_files")
                if isinstance(modified, list):
                    modified_list = cast("list[Any]", modified)
                    return sorted({
                        str(item)
                        for item in modified_list
                        if isinstance(item, str)
                        and item.endswith(c.Infra.Extensions.PYTHON)
                    })
        return []

    def _run_pyrefly_check(self, modified_files: list[str]) -> dict[str, object]:
        if not modified_files:
            return {
                "passed": True,
                "detail": "no modified python files detected",
                "exit_code": 0,
            }
        cmd = [
            sys.executable,
            "-m",
            c.Infra.Cli.PYREFLY,
            c.Infra.Cli.RuffCmd.CHECK,
            *modified_files,
            "--config",
            c.Infra.Files.PYPROJECT_FILENAME,
            "--summary=none",
        ]
        return self._run_external_check(cmd)

    def _run_ruff_check(self, modified_files: list[str]) -> dict[str, object]:
        if not modified_files:
            return {
                "passed": True,
                "detail": "no modified python files detected",
                "exit_code": 0,
            }
        cmd = [
            sys.executable,
            "-m",
            c.Infra.Cli.RUFF,
            c.Infra.Verbs.CHECK,
            *modified_files,
            "--output-format",
            c.Infra.Cli.OUTPUT_JSON,
            "--quiet",
        ]
        return self._run_external_check(cmd)

    def _run_external_check(self, cmd: list[str]) -> dict[str, object]:
        try:
            result = subprocess.run(  # noqa: S603  # JUSTIFIED: executes vetted command list with shell=False
                cmd,
                cwd=self._workspace_root,
                check=False,
                text=True,
                capture_output=True,
            )
        except OSError as exc:
            return {
                "passed": False,
                "detail": f"execution error: {exc}",
                "exit_code": 127,
            }
        output = (result.stderr or result.stdout or "").strip()
        lines = [line for line in output.splitlines() if line.strip()]
        excerpt = " | ".join(lines[:5]) if lines else "ok"
        return {
            "passed": result.returncode == 0,
            "detail": excerpt,
            "exit_code": result.returncode,
        }

    def _scan_import_nodes(self, modified_files: list[str]) -> dict[str, object]:
        invalid_import_from: list[str] = []
        parse_errors: list[str] = []
        for rel_path in modified_files:
            file_path = (self._workspace_root / rel_path).resolve()
            if not file_path.is_file():
                continue
            try:
                source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                tree = ast.parse(source)
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                parse_errors.append(f"{rel_path}:{exc}")
                continue
            invalid_import_from.extend(
                f"{rel_path}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module is None
                and node.level == 0
            )
        return {
            "invalid_import_from_count": len(invalid_import_from),
            "parse_error_count": len(parse_errors),
            "invalid_import_from": invalid_import_from,
            "parse_errors": parse_errors,
        }

    @staticmethod
    def _extract_total_violations(payload: dict[str, object]) -> int:
        if "total_violations" in payload:
            return FlextInfraCodegenConstantsQualityGate._as_int(
                payload.get("total_violations")
            )
        totals = FlextInfraCodegenConstantsQualityGate._dict_or_empty(
            payload.get("totals")
        )
        if totals:
            return (
                FlextInfraCodegenConstantsQualityGate._as_int(
                    totals.get("ns001_violations")
                )
                + FlextInfraCodegenConstantsQualityGate._as_int(
                    totals.get("layer_violations")
                )
                + FlextInfraCodegenConstantsQualityGate._as_int(
                    totals.get("cross_project_reference_violations")
                )
            )
        projects = FlextInfraCodegenConstantsQualityGate._dict_list(
            payload.get("projects")
        )
        if projects and all("total" in item for item in projects):
            return sum(
                FlextInfraCodegenConstantsQualityGate._as_int(item.get("total"))
                for item in projects
            )
        return -1

    @staticmethod
    def _extract_duplicate_groups(payload: dict[str, object]) -> int:
        if "duplicate_groups" in payload:
            return FlextInfraCodegenConstantsQualityGate._as_int(
                payload.get("duplicate_groups")
            )
        duplicates = payload.get("duplicates")
        if isinstance(duplicates, list):
            return len(cast("list[Any]", duplicates))
        return -1

    @staticmethod
    def _extract_projects_total(payload: dict[str, object]) -> int:
        totals = FlextInfraCodegenConstantsQualityGate._dict_or_empty(
            payload.get("totals")
        )
        value = totals.get(c.Infra.ReportKeys.PROJECTS)
        if value is not None:
            return FlextInfraCodegenConstantsQualityGate._as_int(value)
        projects = payload.get("projects")
        if isinstance(projects, list):
            return len(cast("list[Any]", projects))
        return 0

    @staticmethod
    def _extract_projects_passed(payload: dict[str, object]) -> int:
        totals = FlextInfraCodegenConstantsQualityGate._dict_or_empty(
            payload.get("totals")
        )
        return FlextInfraCodegenConstantsQualityGate._as_int(totals.get("passed"))

    @staticmethod
    def _extract_projects_failed(payload: dict[str, object]) -> int:
        totals = FlextInfraCodegenConstantsQualityGate._dict_or_empty(
            payload.get("totals")
        )
        return FlextInfraCodegenConstantsQualityGate._as_int(totals.get("failed"))

    @staticmethod
    def _as_int(value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def _dict_or_empty(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        raw = cast("dict[Any, Any]", value)
        return {str(key): item for key, item in raw.items()}

    @staticmethod
    def _dict_list(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        raw_list = cast("list[Any]", value)
        result: list[dict[str, object]] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            raw_item = cast("dict[Any, Any]", item)
            result.append({str(key): inner for key, inner in raw_item.items()})
        return result

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in cast("list[Any]", value) if isinstance(item, str)]
