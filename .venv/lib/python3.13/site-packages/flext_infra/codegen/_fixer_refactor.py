"""Refactor pass helper for the codegen fixer service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m
from flext_infra.codegen._fixer_results import FlextInfraCodegenFixerResultsMixin
from flext_infra.refactor.service import FlextInfraRefactorService

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraCodegenFixerRefactorMixin(FlextInfraCodegenFixerResultsMixin):
    """Private refactor-service pass for codegen fixer composition."""

    @staticmethod
    def _run_refactor_service(ctx: m.Infra.FixContext, project_path: Path) -> None:
        """Load refactor rules and run the service; record fixed/skipped violations."""
        service = FlextInfraRefactorService()
        config_result = service.load_config()
        rules_result = service.load_rules() if config_result.success else None
        load_error = next(
            (
                message
                for failed, message in (
                    (
                        config_result.failure,
                        config_result.error or "refactor settings load failed",
                    ),
                    (
                        rules_result is not None and rules_result.failure,
                        (
                            rules_result.error
                            if rules_result is not None
                            else "refactor rule load failed"
                        )
                        or "refactor rule load failed",
                    ),
                )
                if failed
            ),
            None,
        )
        if load_error is not None:
            ctx.skip(
                module=project_path.name, rule="REFACTOR", line=0, message=load_error
            )
            return
        refactor_results = tuple(
            service.refactor_project(
                project_path, dry_run=False, apply_safety=False, gates=(c.Infra.LINT,)
            )
        )
        ctx.files_modified |= {
            str(result.file_path) for result in refactor_results if result.success
        }
        ctx.violations_fixed.extend(
            m.Infra.CensusViolation(
                module=str(result.file_path),
                rule="REFACTOR",
                line=1,
                message=change,
                fixable=True,
            )
            for result in refactor_results
            if result.modified
            for change in (tuple(result.changes) or ("refactor applied",))
        )
        ctx.violations_skipped.extend(
            m.Infra.CensusViolation(
                module=str(result.file_path),
                rule="REFACTOR",
                line=1,
                message=result.error or "refactor failed",
                fixable=False,
            )
            for result in refactor_results
            if (not result.modified) and (not result.success)
        )


__all__: list[str] = ["FlextInfraCodegenFixerRefactorMixin"]
