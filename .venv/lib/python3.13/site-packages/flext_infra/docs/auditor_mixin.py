"""Auditor helper mixin for the documentation auditor service.

Provides static helper methods used by the documentation auditor service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from flext_infra import c, r, t, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, p


class FlextInfraDocAuditorMixin:
    """Mixin providing helper methods for the documentation auditor."""

    @staticmethod
    def find_architecture_config(repository_root: Path) -> Path | None:
        """Return repository-local architecture settings without crossing ownership."""
        path = repository_root / "docs/architecture/architecture_config.json"
        return path if path.is_file() else None

    @staticmethod
    def parse_audit_gate(
        audit_gate: t.MappingKV[str, t.Infra.InfraValue],
    ) -> p.Result[t.Pair[int | None, t.IntMapping]]:
        """Extract default budget and per-scope budgets from an audit_gate mapping."""
        default_budget = audit_gate.get("max_issues_default")
        by_scope_raw_value = audit_gate.get("max_issues_by_scope")
        by_scope_raw: t.MappingKV[str, t.Infra.InfraValue] = {}
        if by_scope_raw_value is not None:
            if not isinstance(by_scope_raw_value, Mapping):
                return r[t.Pair[int | None, t.IntMapping]].fail(
                    "docs_validation.audit_gate.max_issues_by_scope must be a mapping"
                )
            try:
                by_scope_raw = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(
                    by_scope_raw_value, strict=True
                )
            except c.ValidationError as exc:
                return r[t.Pair[int | None, t.IntMapping]].fail_op(
                    "audit budget mapping validation", exc
                )
        by_scope: t.MutableIntMapping = {}
        for name, value in by_scope_raw.items():
            if not isinstance(value, t.NUMERIC_TYPES):
                return r[t.Pair[int | None, t.IntMapping]].fail(
                    f"docs_validation.audit_gate.max_issues_by_scope.{name} must be numeric"
                )
            by_scope[name] = int(value)
        if default_budget is not None and not isinstance(
            default_budget, t.NUMERIC_TYPES
        ):
            return r[t.Pair[int | None, t.IntMapping]].fail(
                "docs_validation.audit_gate.max_issues_default must be numeric"
            )
        resolved_default = int(default_budget) if default_budget is not None else None
        return r[t.Pair[int | None, t.IntMapping]].ok((resolved_default, by_scope))

    @classmethod
    def load_audit_budgets(
        cls, repository_root: Path
    ) -> p.Result[t.Pair[int | None, t.IntMapping]]:
        """Load audit budgets from repository-local architecture settings."""
        empty: t.Pair[int | None, t.IntMapping] = (None, {})
        settings = cls.find_architecture_config(repository_root)
        if settings is None:
            return r[t.Pair[int | None, t.IntMapping]].ok(empty)
        payload_result = u.Cli.json_read(settings)
        if payload_result.failure:
            return r[t.Pair[int | None, t.IntMapping]].fail(
                payload_result.error or f"reading audit config {settings}",
                error_code=payload_result.error_code,
            )
        docs_validation = payload_result.value.get("docs_validation")
        if docs_validation is None:
            return r[t.Pair[int | None, t.IntMapping]].ok(empty)
        if not isinstance(docs_validation, Mapping):
            return r[t.Pair[int | None, t.IntMapping]].fail(
                "docs_validation must be a mapping"
            )
        audit_gate = docs_validation.get("audit_gate")
        if audit_gate is None:
            return r[t.Pair[int | None, t.IntMapping]].ok(empty)
        if not isinstance(audit_gate, Mapping):
            return r[t.Pair[int | None, t.IntMapping]].fail(
                "docs_validation.audit_gate must be a mapping"
            )
        try:
            validated_gate = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(
                audit_gate, strict=False
            )
        except c.ValidationError as exc:
            return r[t.Pair[int | None, t.IntMapping]].fail_op(
                "audit gate validation", exc
            )
        return cls.parse_audit_gate(validated_gate)

    @staticmethod
    def resolve_checks(check: str) -> t.Infra.StrSet:
        """Parse check string into a resolved set of check names."""
        checks = {part.strip() for part in check.split(",") if part.strip()}
        if not checks or "all" in checks:
            return {
                "links",
                "forbidden-terms",
                "placeholders",
                "stale-symbols",
                "scope-boundary",
                "generated-ownership",
                "docstrings",
                "python-codeblocks",
            }
        return checks

    @staticmethod
    def write_audit_reports(
        scope: m.Infra.DocScope,
        issues: t.SequenceOf[m.Infra.AuditIssue],
        checks: t.Infra.StrSet,
        *,
        strict: bool,
        docstring_coverage: m.Infra.DocstringCoverage | None = None,
        to_markdown_fn: Callable[
            [
                m.Infra.DocScope,
                t.SequenceOf[m.Infra.AuditIssue],
                m.Infra.DocstringCoverage | None,
            ],
            t.StrSequence,
        ],
    ) -> None:
        """Persist JSON summary and markdown report to the scope report directory."""
        validated_checks = t.json_list_adapter().validate_python(sorted(checks))
        sorted_checks: t.JsonValueList = list(validated_checks)
        summary: t.JsonDict = {
            c.Infra.RK_SCOPE: scope.name,
            "issues": len(issues),
            c.Infra.VERB_CHECKS: sorted_checks,
            c.Infra.OperationMode.STRICT: strict,
            "report_dir": scope.report_dir.as_posix(),
        }
        if docstring_coverage is not None:
            summary["docstring_coverage"] = docstring_coverage.model_dump()
        issues_payload: t.JsonValue = [
            {
                c.Infra.RK_FILE: issue.file,
                "issue_type": issue.issue_type,
                "severity": issue.severity,
                c.Infra.RK_MESSAGE: issue.message,
            }
            for issue in issues
        ]
        summary_payload: t.JsonDict = {
            c.Infra.RK_SUMMARY: summary,
            "issues": issues_payload,
        }
        _ = u.Cli.json_write(scope.report_dir / "audit-summary.json", summary_payload)
        _ = u.Infra.write_markdown(
            scope.report_dir / "audit-report.md",
            to_markdown_fn(scope, issues, docstring_coverage),
        )


__all__: list[str] = ["FlextInfraDocAuditorMixin"]
