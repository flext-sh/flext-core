"""Documentation validator service.

Validates documentation for ADR skill references and generates
validation reports, returning structured FlextResult reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from flext_core import FlextLogger, FlextResult, r
from pydantic import BaseModel, ConfigDict, Field

from flext_infra.constants import c
from flext_infra.docs.shared import (
    FlextInfraDocScope,
    FlextInfraDocsShared,
)

logger = FlextLogger.create_module_logger(__name__)


class ValidateReport(BaseModel):
    """Structured validation report for a scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str = Field(..., description="Scope name.")
    result: str = Field(..., description="Validation result status.")
    message: str = Field(..., description="Human-readable result message.")
    missing_adr_skills: list[str] = Field(
        default_factory=list,
        description="List of missing ADR skills.",
    )
    todo_written: bool = Field(
        default=False,
        description="Whether TODOS.md was written.",
    )


class FlextInfraDocValidator:
    """Infrastructure service for documentation validation.

    Checks ADR skill references and generates validation reports,
    returning structured FlextResult reports.
    """

    def validate(
        self,
        root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR,
        check: str = "all",
        apply: bool = False,
    ) -> FlextResult[list[ValidateReport]]:
        """Run documentation validation across project scopes.

        Args:
            root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            check: Validation checks to run.
            apply: Write TODOS.md if needed.

        Returns:
            FlextResult with list of ValidateReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[ValidateReport]].fail(scopes_result.error or "scope error")

        reports: list[ValidateReport] = []
        for scope in scopes_result.value:
            report = self._validate_scope(scope, check=check, apply_mode=apply)
            reports.append(report)

        return r[list[ValidateReport]].ok(reports)

    def _validate_scope(
        self,
        scope: FlextInfraDocScope,
        *,
        check: str,
        apply_mode: bool,
    ) -> ValidateReport:
        """Run validation for a single project scope."""
        status = c.Status.OK
        message = "validation passed"
        missing_adr_skills: list[str] = []

        config_exists = (
            scope.path / "docs/architecture/architecture_config.json"
        ).exists()
        if scope.name == "root" and config_exists and check in {"adr-skill", "all"}:
            code, missing = self._run_adr_skill_check(scope.path)
            missing_adr_skills = missing
            if code != 0:
                status = c.Status.FAIL
                message = f"missing adr references in skills: {', '.join(missing)}"

        wrote_todo = self._maybe_write_todo(scope, apply_mode=apply_mode)

        _ = FlextInfraDocsShared.write_json(
            scope.report_dir / "validate-summary.json",
            {
                "summary": {
                    "scope": scope.name,
                    "result": status,
                    "message": message,
                    "apply": apply_mode,
                },
                "details": {
                    "missing_adr_skills": missing_adr_skills,
                    "todo_written": wrote_todo,
                },
            },
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
            phase="validate",
            result=status,
            reason=message,
        )

        return ValidateReport(
            scope=scope.name,
            result=status,
            message=message,
            missing_adr_skills=missing_adr_skills,
            todo_written=wrote_todo,
        )

    @staticmethod
    def _has_adr_reference(skill_path: Path) -> bool:
        """Check whether a skill file contains an ADR reference."""
        text = skill_path.read_text(
            encoding=c.Encoding.DEFAULT,
            errors="ignore",
        ).lower()
        return "adr" in text

    def _run_adr_skill_check(self, root: Path) -> tuple[int, list[str]]:
        """Run ADR skill check and return exit code with missing skill names."""
        skills_root = root / ".claude/skills"
        required: list[str] = []
        config = root / "docs/architecture/architecture_config.json"
        if config.exists():
            payload = json.loads(
                config.read_text(encoding=c.Encoding.DEFAULT, errors="ignore"),
            )
            docs_validation = payload.get("docs_validation", {})
            configured = docs_validation.get("required_skills", [])
            if isinstance(configured, list):
                required = [
                    item for item in configured if isinstance(item, str) and item
                ]
        if not required:
            required = ["rules-docs", "scripts-maintenance", "readme-standardization"]

        missing: list[str] = []
        for name in required:
            skill = skills_root / name / "SKILL.md"
            if not skill.exists() or not self._has_adr_reference(skill):
                missing.append(name)
        return (0 if not missing else 1), missing

    @staticmethod
    def _maybe_write_todo(scope: FlextInfraDocScope, *, apply_mode: bool) -> bool:
        """Write a TODOS.md file for the scope if apply mode is enabled."""
        if scope.name == "root" or not apply_mode:
            return False
        path = scope.path / "TODOS.md"
        content = (
            "# TODOS\n\n"
            "- [ ] Resolve documentation validation findings "
            "from `.reports/docs/validate-report.md`.\n"
        )
        _ = path.write_text(content, encoding=c.Encoding.DEFAULT)
        return True


__all__ = ["FlextInfraDocValidator", "ValidateReport"]
