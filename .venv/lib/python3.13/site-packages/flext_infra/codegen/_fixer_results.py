"""Result helpers for the codegen fixer service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import m, u
from flext_infra.validate.namespace_validator import FlextInfraNamespaceValidator

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t

_log = u.fetch_logger(__name__)


class FlextInfraCodegenFixerResultsMixin:
    """Private result and validation helpers for codegen fixer composition."""

    @staticmethod
    def _empty_result(project_name: str) -> m.Infra.AutoFixResult:
        """Empty result."""
        return m.Infra.AutoFixResult(
            project=project_name,
            violations_fixed=[],
            violations_skipped=[],
            files_modified=[],
        )

    @staticmethod
    def _build_result(
        project_name: str, ctx: m.Infra.FixContext
    ) -> m.Infra.AutoFixResult:
        """Build result."""
        return m.Infra.AutoFixResult(
            project=project_name,
            violations_fixed=list(ctx.violations_fixed),
            violations_skipped=list(ctx.violations_skipped),
            files_modified=sorted(ctx.files_modified),
        )

    @staticmethod
    def _load_initial_violations(
        ctx: m.Infra.FixContext, project_path: Path
    ) -> t.SequenceOf[m.Infra.CensusViolation]:
        """Read the initial namespace violations and record skip reason on failure."""
        initial_violations_result = u.Infra.parse_namespace_validation(
            FlextInfraNamespaceValidator().validate_project(project_path)
        )
        if initial_violations_result.failure:
            _log.warning(
                "namespace_validation_failed",
                project=project_path.name,
                error=str(initial_violations_result.error),
            )
            ctx.skip(
                module=project_path.name,
                rule="NAMESPACE",
                line=0,
                message=initial_violations_result.error
                or "namespace validation failed",
            )
            return ()
        return initial_violations_result.unwrap()

    @staticmethod
    def _classify_remaining_violations(
        ctx: m.Infra.FixContext,
        project_path: Path,
        initial_violations: t.SequenceOf[m.Infra.CensusViolation],
    ) -> None:
        """Re-run validation and split outstanding violations into fixed vs skipped."""
        remaining_result = u.Infra.parse_namespace_validation(
            FlextInfraNamespaceValidator().validate_project(project_path)
        )
        if remaining_result.failure:
            ctx.skip(
                module=project_path.name,
                rule="NAMESPACE",
                line=0,
                message=remaining_result.error or "namespace validation failed",
            )
            return
        fixed, skipped = u.Infra.classify_violation_outcomes(
            project_path=project_path,
            initial_violations=initial_violations,
            remaining_violations=remaining_result.unwrap(),
        )
        ctx.violations_fixed.extend(fixed)
        ctx.violations_skipped.extend(skipped)


__all__: list[str] = ["FlextInfraCodegenFixerResultsMixin"]
