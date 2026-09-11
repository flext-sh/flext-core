"""Documentation auditor service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_infra import m, r, u
from flext_infra.docs._auditor_checks import FlextInfraDocAuditorChecksMixin
from flext_infra.docs._auditor_report import FlextInfraDocAuditorReportMixin
from flext_infra.docs.auditor_mixin import FlextInfraDocAuditorMixin
from flext_infra.docs.base import FlextInfraDocServiceBase

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraDocAuditor(
    FlextInfraDocServiceBase,
    FlextInfraDocAuditorMixin,
    FlextInfraDocAuditorChecksMixin,
    FlextInfraDocAuditorReportMixin,
):
    """Audit governed docs scopes using code-backed and policy-backed checks."""

    strict_mode: Annotated[
        bool, m.Field(alias="strict", description="Strict audit mode")
    ] = False

    # kimi-docs flext-3o9s: seletor de checks via CLI como --checks (flag --check é o
    # alias bool de check_only na base); default "all" = comportamento anterior.
    checks: Annotated[
        str, m.Field(description="Comma-separated audit checks (default: all)")
    ] = "all"

    # kimi-docs flext-3o9s: threshold de cobertura de docstrings via CLI
    # (--docstring-min); None = desativado. Substitui o interrogate paralelo.
    docstring_min: Annotated[
        float | None,
        m.Field(
            description="Minimum docstring coverage percent; breach fails the audit"
        ),
    ] = None

    def audit(
        self,
        repository_root: Path,
        *,
        projects: t.StrSequence | None = None,
        output_dir: Path | str | None = None,
        params: m.Infra.AuditScopeParams | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.DocsPhaseReport]]:
        """Audit root and governed project docs scopes."""
        resolved_params = self._audit_params(repository_root, params)
        if resolved_params.failure:
            return r.fail(
                resolved_params.error or "audit parameter resolution failed",
                error_code=resolved_params.error_code,
            )
        return self.run_scoped_docs(
            repository_root,
            projects=projects,
            output_dir=output_dir,
            handler=lambda scope: self.audit_scope(scope, params=resolved_params.value),
        )

    def audit_scope(
        self, scope: m.Infra.DocScope, *, params: m.Infra.AuditScopeParams
    ) -> m.Infra.DocsPhaseReport:
        """Audit one scope and persist the standard reports."""
        checks = sorted(self.resolve_checks(params.check))
        issues = self._collect_issues(scope, checks)
        docstring_coverage = (
            u.Infra.docs_public_docstring_coverage(scope)
            if "docstrings" in checks
            else None
        )
        report = self._audit_report(
            scope,
            issues=issues,
            checks=checks,
            params=params,
            docstring_coverage=docstring_coverage,
        )
        self.write_audit_reports(
            scope,
            issues,
            set(checks),
            strict=params.strict,
            docstring_coverage=docstring_coverage,
            to_markdown_fn=u.Infra.docs_audit_markdown,
        )
        self.logger.info(
            "docs_audit_scope_completed",
            project=scope.name,
            phase="audit",
            result=report.result,
            reason=report.reason,
        )
        return report

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the configured docs audit flow."""
        return self._propagate_phase_outcome(
            "audit",
            self.audit(
                repository_root=self.repository_root,
                projects=self.selected_projects,
                output_dir=self.output_dir,
                params=m.Infra.AuditScopeParams(
                    check=self.checks,
                    strict=self.strict_mode,
                    docstring_min=self.docstring_min,
                ),
            ),
            failure_predicate=lambda report: not report.passed,
        )

    def _audit_params(
        self, repository_root: Path, params: m.Infra.AuditScopeParams | None
    ) -> p.Result[m.Infra.AuditScopeParams]:
        """Resolve runtime audit parameters and load default budgets when absent."""
        if params is not None and params.budgets is not None:
            return r[m.Infra.AuditScopeParams].ok(params)
        budgets_result = self.load_audit_budgets(repository_root)
        if budgets_result.failure:
            return r[m.Infra.AuditScopeParams].fail(
                budgets_result.error or "audit budget resolution failed",
                error_code=budgets_result.error_code,
            )
        budgets = budgets_result.value
        if params is None:
            resolved = m.Infra.AuditScopeParams(
                check="all",
                strict=self.strict_mode,
                docstring_min=self.docstring_min,
                budgets=budgets,
            )
        else:
            resolved = m.Infra.AuditScopeParams(
                check=params.check,
                strict=params.strict,
                docstring_min=params.docstring_min,
                budgets=budgets,
            )
        return r[m.Infra.AuditScopeParams].ok(resolved)


if __name__ == "__main__":
    raise SystemExit(0)


__all__: list[str] = ["FlextInfraDocAuditor"]
