"""Validation helpers for docs services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_cli import u
from flext_core import r
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra._utilities.docs_api import FlextInfraUtilitiesDocsApi
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra.constants import c
from flext_infra.typings import t

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, p


class FlextInfraUtilitiesDocsValidate:
    """Reusable validation helpers exposed through ``u.Infra``."""

    @staticmethod
    def docs_has_adr_reference(skill_path: Path) -> bool:
        """Return whether a skill file contains an ADR reference."""
        text = skill_path.read_text(
            encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
        )
        return "adr" in text.lower()

    @staticmethod
    def docs_extract_required_skills(
        payload: t.JsonPayload | t.MappingKV[str, t.Infra.InfraValue],
    ) -> p.Result[t.Infra.InfraSequence]:
        """Extract the configured required skills list from architecture settings.

        ``r.ok(list)`` when the configuration block is present and well
        shaped, ``r.fail(reason)`` otherwise. Callers that want to treat
        absence as "no override" can collapse with ``unwrap_or(())``.
        """
        match payload:
            case Mapping() as outer:
                pass
            case _:
                return r[t.Infra.InfraSequence].fail("payload is not a mapping")
        match outer.get("docs_validation"):
            case Mapping() as inner:
                pass
            case _:
                return r[t.Infra.InfraSequence].fail(
                    "docs_validation block missing or not a mapping"
                )
        match inner.get("required_skills"):
            case list() as configured:
                return r[t.Infra.InfraSequence].ok(configured)
            case _:
                return r[t.Infra.InfraSequence].fail(
                    "required_skills missing or not a list"
                )

    @staticmethod
    def docs_load_required_skills(repository_root: Path) -> p.Result[t.StrSequence]:
        """Load the required skills list from the architecture settings.

        Returns ``r.ok([])`` only when the config file is genuinely absent.
        Any other failure (read error, malformed JSON, broken
        ``docs_validation`` block, invalid ``required_skills`` payload)
        propagates as ``r.fail(...)`` so callers see the config defect
        instead of a silent empty list — fail-fast over fail-quiet.
        """
        settings = repository_root / "docs/architecture/architecture_config.json"
        if not settings.exists():
            return r[t.StrSequence].ok([])
        return (
            u.Cli
            .json_read(settings)
            .map_error(lambda e: f"failed to read architecture config: {e}")
            .flat_map(FlextInfraUtilitiesDocsValidate.docs_extract_required_skills)
            .flat_map(FlextInfraUtilitiesDocsValidate._validate_required_skills)
            .map(lambda values: [item for item in values if item])
        )

    @staticmethod
    def _validate_required_skills(
        raw: t.Infra.InfraSequence,
    ) -> p.Result[t.StrSequence]:
        """Validate ``required_skills`` payload against the canonical adapter."""
        return (
            r[t.StrSequence]
            .create_from_callable(
                lambda: t.Infra.STR_SEQ_ADAPTER.validate_python(raw, strict=True),
                error_code="required_skills_validation",
            )
            .map_error(lambda e: f"invalid required_skills configuration: {e}")
        )

    @staticmethod
    def docs_missing_required_paths(scope: m.Infra.DocScope) -> t.StrSequence:
        """Return required docs paths that are still missing from one scope."""
        if scope.name == c.Infra.RK_ROOT:
            required = [
                "README.md",
                "docs/index.md",
                "docs/api-reference/README.md",
                "docs/api-reference/generated/overview.md",
                "docs/projects/generated/catalog.md",
                "mkdocs.yml",
            ]
        else:
            required = list(FlextInfraUtilitiesDocsScope.required_project_files())
        missing: t.MutableSequenceOf[str] = []
        for rel_path in sorted(set(required)):
            if not (scope.path / rel_path).exists():
                missing.append(rel_path)
        return missing

    @staticmethod
    def docs_contract_messages(scope: m.Infra.DocScope) -> t.StrSequence:
        """Return public API contract problems for one governed project scope."""
        if scope.name == c.Infra.RK_ROOT or not scope.package_name:
            return []
        messages: t.MutableSequenceOf[str] = []
        init_path = (
            scope.path / c.Infra.DEFAULT_SRC_DIR / scope.package_name / c.Infra.INIT_PY
        )
        if not init_path.exists():
            messages.append(
                "missing public package init: "
                f"{init_path.relative_to(scope.path).as_posix()}"
            )
            return messages
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        if not contract.get("modules") and not contract.get("exports"):
            messages.append("empty public API contract from package exports")
        return messages

    @staticmethod
    def docs_write_todo(scope: m.Infra.DocScope, *, apply_mode: bool) -> p.Result[bool]:
        """Write the standard ``TODOS.md`` helper file when requested.

        ``r.ok(True)`` when a TODO file was written, ``r.ok(False)`` when
        the call is a no-op (root scope or non-apply mode), ``r.fail(...)``
        when the underlying ``write_text`` raises.
        """
        if scope.name == c.Infra.RK_ROOT or not apply_mode:
            return r[bool].ok(False)
        path = scope.path / "TODOS.md"
        content = (
            "# TODOS\n\n"
            "- [ ] Resolve documentation validation findings from "
            "`.reports/docs/validate-report.md`.\n"
        )
        try:
            _ = path.write_text(content, encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            return r[bool].fail(f"failed to write {path}: {exc}")
        return r[bool].ok(True)

    @staticmethod
    def docs_write_validate_reports(
        scope: m.Infra.DocScope, report: m.Infra.DocsPhaseReport
    ) -> None:
        """Persist the standard validate summary and markdown report."""
        _ = u.Cli.json_write(
            scope.report_dir / "validate-summary.json",
            {c.Infra.RK_SUMMARY: report.model_dump(mode="json")},
        )
        _ = FlextInfraUtilitiesDocs.write_markdown(
            scope.report_dir / "validate-report.md",
            [
                "# Docs Validate Report",
                "",
                f"Scope: {report.scope}",
                f"Result: {report.result}",
                f"Message: {report.message}",
                f"TODO written: {int(report.todo_written)}",
            ],
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsValidate"]
