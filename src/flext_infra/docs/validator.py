"""Documentation validator service.

Validates documentation for ADR skill references and generates
validation reports, returning structured r reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_core import FlextLogger, r
from flext_infra import FlextInfraDocsShared, c, m, u

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraDocValidator:
    """Infrastructure service for documentation validation.

    Checks ADR skill references and generates validation reports,
    returning structured r reports.
    """

    @staticmethod
    def _has_adr_reference(skill_path: Path) -> bool:
        """Check whether a skill file contains an ADR reference."""
        text = skill_path.read_text(
            encoding=c.Infra.Encoding.DEFAULT,
            errors=c.Infra.Toml.IGNORE,
        ).lower()
        return "adr" in text

    @staticmethod
    def _maybe_write_todo(
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        apply_mode: bool,
    ) -> bool:
        """Write a TODOS.md file for the scope if apply mode is enabled."""
        if scope.name == c.Infra.ReportKeys.ROOT or not apply_mode:
            return False
        path = scope.path / "TODOS.md"
        content = "# TODOS\n\n- [ ] Resolve documentation validation findings from `.reports/docs/validate-report.md`.\n"
        _ = path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
        return True

    def validate(
        self,
        workspace_root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR,
        check: str = "all",
        apply: bool = False,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        """Run documentation validation across project scopes.

        Args:
            workspace_root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            check: Validation checks to run.
            apply: Write TODOS.md if needed.

        Returns:
            r with list of ValidateReport objects.

        """
        scopes_result: r[list[m.Infra.Docs.FlextInfraDocScope]] = (
            FlextInfraDocsShared.build_scopes(
                workspace_root=workspace_root,
                project=project,
                projects=projects,
                output_dir=output_dir,
            )
        )
        if scopes_result.is_failure:
            return r[list[m.Infra.Docs.DocsPhaseReport]].fail(
                scopes_result.error or "scope error",
            )
        reports: list[m.Infra.Docs.DocsPhaseReport] = []
        for scope in scopes_result.value:
            report = self._validate_scope(scope, check=check, apply_mode=apply)
            reports.append(report)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(reports)

    def _run_adr_skill_check(self, workspace_root: Path) -> tuple[int, list[str]]:
        """Run ADR skill check and return exit code with missing skill names."""
        skills_root = workspace_root / ".claude/skills"
        required: list[str] = []
        config = workspace_root / "docs/architecture/architecture_config.json"
        if config.exists():
            payload_result = u.Infra.read_json(config)
            if payload_result.is_failure:
                return (1, [])
            payload = payload_result.value
            docs_validation = payload.get("docs_validation")
            if isinstance(docs_validation, dict):
                configured = docs_validation.get("required_skills")
                if isinstance(configured, list):
                    try:
                        required_items = TypeAdapter(list[str]).validate_python(
                            configured,
                            strict=True,
                        )
                        required = [item for item in required_items if item]
                    except ValidationError:
                        required = []
        if not required:
            required = ["rules-docs", "scripts-maintenance", "readme-standardization"]
        missing: list[str] = []
        for name in required:
            skill = skills_root / name / "SKILL.md"
            if not skill.exists() or not self._has_adr_reference(skill):
                missing.append(name)
        return (0 if not missing else 1, missing)

    def _validate_scope(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        check: str,
        apply_mode: bool,
    ) -> m.Infra.Docs.DocsPhaseReport:
        """Run validation for a single project scope."""
        status = c.Infra.Status.OK
        message = "validation passed"
        missing_adr_skills: list[str] = []
        config_exists = (
            scope.path / "docs/architecture/architecture_config.json"
        ).exists()
        if (
            scope.name == c.Infra.ReportKeys.ROOT
            and config_exists
            and (check in {"adr-skill", "all"})
        ):
            code, missing = self._run_adr_skill_check(scope.path)
            missing_adr_skills = missing
            if code != 0:
                status = c.Infra.Status.FAIL
                message = f"missing adr references in skills: {', '.join(missing)}"
        wrote_todo = self._maybe_write_todo(scope, apply_mode=apply_mode)
        adr_skills_json: list[JsonValue] = list(missing_adr_skills)
        payload: JsonValue = {
            c.Infra.ReportKeys.SUMMARY: {
                c.Infra.ReportKeys.SCOPE: scope.name,
                "result": status,
                c.Infra.ReportKeys.MESSAGE: message,
                "apply": apply_mode,
            },
            "details": {
                "missing_adr_skills": adr_skills_json,
                "todo_written": wrote_todo,
            },
        }
        _ = u.Infra.write_json(
            scope.report_dir / "validate-summary.json",
            payload,
        )
        _ = FlextInfraDocsShared.write_markdown(
            scope.report_dir / "validate-report.md",
            [
                "# Docs Validate Report",
                "",
                f"Scope: {scope.name}",
                f"Result: {status}",
                f"Message: {message}",
                f"TODO written: {int(wrote_todo)}",
            ],
        )
        logger.info(
            "docs_validate_scope_completed",
            project=scope.name,
            phase=c.Infra.Verbs.VALIDATE,
            result=status,
            reason=message,
        )
        return m.Infra.Docs.DocsPhaseReport(
            phase="validate",
            scope=scope.name,
            result=status,
            message=message,
            missing_adr_skills=missing_adr_skills,
            todo_written=wrote_todo,
            passed=status == c.Infra.Status.OK,
        )


__all__ = ["FlextInfraDocValidator"]
