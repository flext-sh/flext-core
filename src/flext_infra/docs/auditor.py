"""Documentation auditor service.

Audits documentation for broken links and forbidden terms,
returning structured FlextResult reports.

Usage:
    python -m flext_infra docs audit --root flext-core

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from flext_core import FlextLogger, FlextResult, r, t
from pydantic import BaseModel, ConfigDict, Field

from flext_infra import FlextInfraPatterns, c, output
from flext_infra.docs.shared import (
    DEFAULT_DOCS_OUTPUT_DIR,
    FlextInfraDocScope,
    FlextInfraDocsShared,
)

logger = FlextLogger.create_module_logger(__name__)


class AuditIssue(BaseModel):
    """Single documentation audit finding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str = Field(..., description="File path relative to scope.")
    issue_type: str = Field(..., description="Type of issue found.")
    severity: str = Field(..., description="Severity level (high, medium, low).")
    message: str = Field(..., description="Human-readable issue message.")


class AuditReport(BaseModel):
    """Structured audit report for a scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str = Field(..., description="Scope name.")
    issues: list[AuditIssue] = Field(
        default_factory=list,
        description="List of audit issues.",
    )
    checks: list[str] = Field(default_factory=list, description="Checks that were run.")
    strict: bool = Field(default=False, description="Whether strict mode was enabled.")
    passed: bool = Field(default=False, description="Whether audit passed.")


class DocAuditor:
    """Infrastructure service for documentation auditing.

    Scans markdown documentation for broken internal links and
    forbidden terms, returning structured FlextResult reports.
    """

    def audit(
        self,
        root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = DEFAULT_DOCS_OUTPUT_DIR,
        check: str = "all",
        strict: bool = True,
    ) -> FlextResult[list[AuditReport]]:
        """Run documentation audit across project scopes.

        Args:
            root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            check: Comma-separated checks (links, forbidden-terms, all).
            strict: Fail on any issues found.

        Returns:
            FlextResult with list of AuditReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[AuditReport]].fail(scopes_result.error or "scope error")

        default_budget, by_scope_budget = self._load_audit_budgets(root)
        reports: list[AuditReport] = []
        for scope in scopes_result.value:
            report = self._audit_scope(
                scope,
                check=check,
                strict=strict,
                max_issues_default=default_budget,
                max_issues_by_scope=by_scope_budget,
            )
            reports.append(report)

        return r[list[AuditReport]].ok(reports)

    def _audit_scope(
        self,
        scope: FlextInfraDocScope,
        *,
        check: str,
        strict: bool,
        max_issues_default: int | None,
        max_issues_by_scope: Mapping[str, int],
    ) -> AuditReport:
        """Run configured audit checks on a single scope."""
        checks = {part.strip() for part in check.split(",") if part.strip()}
        if not checks or "all" in checks:
            checks = {"links", "forbidden-terms"}

        issues: list[AuditIssue] = []
        if "links" in checks:
            issues.extend(self._broken_link_issues(scope))
        if "forbidden-terms" in checks:
            issues.extend(self._forbidden_term_issues(scope))

        # Write reports
        summary: Mapping[str, t.ConfigMapValue] = {
            "scope": scope.name,
            "issues": len(issues),
            "checks": sorted(checks),
            "strict": strict,
            "report_dir": scope.report_dir.as_posix(),
        }
        issues_payload: list[Mapping[str, t.ConfigMapValue]] = [
            {
                "file": issue.file,
                "issue_type": issue.issue_type,
                "severity": issue.severity,
                "message": issue.message,
            }
            for issue in issues
        ]
        summary_payload: Mapping[str, t.ConfigMapValue] = {
            "summary": summary,
            "issues": issues_payload,
        }
        _ = FlextInfraDocsShared.write_json(
            scope.report_dir / "audit-summary.json",
            summary_payload,
        )
        _ = FlextInfraDocsShared.write_markdown(
            scope.report_dir / "audit-report.md",
            self._to_markdown(scope, issues),
        )

        # Determine pass/fail
        max_issues = max_issues_by_scope.get(scope.name, max_issues_default)
        if strict:
            limit = 0 if max_issues is None else max_issues
            passed = len(issues) <= limit
        else:
            passed = True

        status = c.Status.OK if passed else c.Status.FAIL
        reason = f"issues:{len(issues)}"
        logger.info(
            "docs_audit_scope_completed",
            project=scope.name,
            phase="audit",
            result=status,
            reason=reason,
        )

        return AuditReport(
            scope=scope.name,
            issues=issues,
            checks=sorted(checks),
            strict=strict,
            passed=passed,
        )

    def _broken_link_issues(self, scope: FlextInfraDocScope) -> list[AuditIssue]:
        """Collect broken internal-link issues for markdown files in scope."""
        issues: list[AuditIssue] = []
        for md_file in FlextInfraDocsShared.iter_markdown_files(scope.path):
            content = md_file.read_text(encoding=c.Encoding.DEFAULT, errors="ignore")
            rel = md_file.relative_to(scope.path).as_posix()
            in_fenced_code = False
            for number, line in enumerate(content.splitlines(), start=1):
                stripped = line.lstrip()
                if stripped.startswith("```"):
                    in_fenced_code = not in_fenced_code
                    continue
                if in_fenced_code:
                    continue
                clean_line = FlextInfraPatterns.INLINE_CODE_RE.sub("", line)
                for raw in FlextInfraPatterns.MARKDOWN_LINK_URL_RE.findall(clean_line):
                    target = self._normalize_link(raw)
                    if (
                        not target
                        or target.startswith("#")
                        or self._is_external(target)
                    ):
                        continue
                    if self._should_skip_target(raw, target):
                        continue
                    path = (md_file.parent / target).resolve()
                    if not path.exists():
                        issues.append(
                            AuditIssue(
                                file=rel,
                                issue_type="broken_link",
                                severity="high",
                                message=f"line {number}: target not found -> {raw}",
                            ),
                        )
        return issues

    def _forbidden_term_issues(self, scope: FlextInfraDocScope) -> list[AuditIssue]:
        """Collect forbidden-term issues for markdown files in scope."""
        issues: list[AuditIssue] = []
        terms: tuple[str, ...] = ()
        for md_file in FlextInfraDocsShared.iter_markdown_files(scope.path):
            rel = md_file.relative_to(scope.path).as_posix()
            rel_lower = rel.lower()
            if scope.name == "root":
                if not rel_lower.startswith("docs/"):
                    continue
            elif not scope.name.startswith("flext-"):
                continue
            content = md_file.read_text(
                encoding=c.Encoding.DEFAULT,
                errors="ignore",
            ).lower()
            issues.extend(
                AuditIssue(
                    file=rel,
                    issue_type="forbidden_term",
                    severity="medium",
                    message=f"contains forbidden term '{term}'",
                )
                for term in terms
                if term in content
            )
        return issues

    @staticmethod
    def _normalize_link(target: str) -> str:
        """Strip fragment and query-string from a markdown link target."""
        value = target.strip()
        if value.startswith("<") and value.endswith(">"):
            value = value[1:-1].strip()
        return value.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]

    @staticmethod
    def _should_skip_target(raw: str, target: str) -> bool:
        """Return whether link text should be ignored as a non-path target."""
        if target.startswith("http"):
            return False
        if "," in raw and ".md" not in raw and "/" not in raw:
            return True
        return bool(" " in raw and ".md" not in raw and "/" not in raw)

    @staticmethod
    def _is_external(target: str) -> bool:
        """Return True when target points outside the repository."""
        lower = target.strip().lower().lstrip("<")
        return lower.startswith(("http://", "https://", "mailto:", "tel:", "data:"))

    @staticmethod
    def _to_markdown(scope: FlextInfraDocScope, issues: list[AuditIssue]) -> list[str]:
        """Format audit issues as a markdown report."""
        return [
            "# Docs Audit Report",
            "",
            f"Scope: {scope.name}",
            f"Files scanned: {len(FlextInfraDocsShared.iter_markdown_files(scope.path))}",
            f"Issues: {len(issues)}",
            "",
            "| file | type | severity | message |",
            "|---|---|---|---|",
            *[
                f"| {issue.file} | {issue.issue_type} | {issue.severity} | {issue.message} |"
                for issue in issues
            ],
        ]

    @staticmethod
    def _load_audit_budgets(root: Path) -> tuple[int | None, Mapping[str, int]]:
        """Load audit issue budgets from architecture config."""
        config_path: Path | None = None
        for candidate in [root, *root.parents]:
            path = candidate / "docs/architecture/architecture_config.json"
            if path.exists():
                config_path = path
                break
        if config_path is None:
            return None, {}

        payload = json.loads(
            config_path.read_text(encoding=c.Encoding.DEFAULT, errors="ignore"),
        )
        docs_validation = payload.get("docs_validation", {})
        audit_gate = docs_validation.get("audit_gate", {})
        default_budget = audit_gate.get("max_issues_default")
        by_scope_raw = audit_gate.get("max_issues_by_scope", {})
        by_scope = {
            str(name): int(value)
            for name, value in by_scope_raw.items()
            if isinstance(value, (int, float))
        }
        if isinstance(default_budget, (int, float)):
            return int(default_budget), by_scope
        return None, by_scope


def main() -> int:
    """CLI entry point for the documentation auditor."""
    parser = argparse.ArgumentParser(description="Audit documentation for issues")
    _ = parser.add_argument("--root", default=".")
    _ = parser.add_argument("--project")
    _ = parser.add_argument("--projects")
    _ = parser.add_argument("--output-dir", default=DEFAULT_DOCS_OUTPUT_DIR)
    _ = parser.add_argument("--check", default="all")
    _ = parser.add_argument("--strict", type=int, default=1)
    args = parser.parse_args()

    auditor = DocAuditor()
    result = auditor.audit(
        root=Path(args.root).resolve(),
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
        check=args.check,
        strict=bool(args.strict),
    )

    if result.is_failure:
        output.error(result.error or "audit failed")
        return 1

    failures = sum(1 for report in result.value if not report.passed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AuditIssue", "AuditReport", "DocAuditor"]
