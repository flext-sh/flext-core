"""Docs-audit per-issue-type detectors (token/scope/ownership/docstring/codeblock)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra._utilities.docs_api import FlextInfraUtilitiesDocsApi
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.typings import t


class FlextInfraUtilitiesDocsAuditDetectorsMixin:
    """Self-contained docs-audit issue detectors.

    Composed into FlextInfraUtilitiesDocsAudit via inheritance; each detector
    is an independent ``u.Infra.docs_*_issues`` entry over one ``DocScope``.
    """

    @staticmethod
    def docs_text_token_issues(
        scope: m.Infra.DocScope, *, tokens: t.StrSequence, issue_type: str
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect simple token-presence issues from markdown files."""
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        if not tokens:
            return issues
        for md_file in FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope):
            rel = md_file.relative_to(scope.path).as_posix()
            text = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )
            for token in tokens:
                if token in text:
                    issues.append(
                        m.Infra.AuditIssue(
                            file=rel,
                            issue_type=issue_type,
                            severity="medium",
                            message=f"contains `{token}`",
                        )
                    )
        return issues

    @staticmethod
    def docs_scope_boundary_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect mentions of excluded non-FLEXT roots in root docs."""
        if scope.name != c.Infra.RK_ROOT:
            return []
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        excluded = sorted(FlextInfraUtilitiesDocsScope.excluded_roots(scope.path))
        for md_file in [
            scope.path / "README.md",
            *FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope),
        ]:
            if not md_file.exists():
                continue
            text = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )
            for token in excluded:
                if token in text:
                    issues.append(
                        m.Infra.AuditIssue(
                            file=md_file.relative_to(scope.path).as_posix(),
                            issue_type="scope_boundary",
                            severity="high",
                            message=f"root docs mention out-of-scope project `{token}`",
                        )
                    )
        return issues

    @staticmethod
    def docs_generated_ownership_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect manual API pages that duplicate generated ownership."""
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        candidates: t.MutableSequenceOf[Path] = [scope.path / "docs/api-reference.md"]
        for parent in (scope.path / "docs/api-reference", scope.path / "docs/api"):
            if parent.exists():
                candidates.extend(sorted(parent.rglob("*.md")))
        for path in candidates:
            if not path.exists():
                continue
            rel = path.relative_to(scope.path).as_posix()
            if path.is_relative_to(scope.path / c.Infra.DIR_DOCS) and (
                FlextInfraUtilitiesDocsScope.is_excluded_doc_path(
                    scope.path, path.relative_to(scope.path / c.Infra.DIR_DOCS)
                )
            ):
                continue
            if rel == "docs/api-reference/README.md" or rel.startswith(
                "docs/api-reference/generated/"
            ):
                continue
            issues.append(
                m.Infra.AuditIssue(
                    file=rel,
                    issue_type="generated_ownership",
                    severity="medium",
                    message="manual API page duplicates generated API ownership",
                )
            )
        return issues

    @staticmethod
    def docs_public_docstring_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect missing docstring issues for public exports and modules."""
        if scope.name == c.Infra.RK_ROOT or not scope.package_name:
            return []
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        return FlextInfraUtilitiesDocsApi.docstring_issues(scope.path, contract)

    @staticmethod
    def docs_public_docstring_coverage(
        scope: m.Infra.DocScope,
    ) -> m.Infra.DocstringCoverage | None:
        """Aggregate docstring coverage for a project scope (None at root)."""
        if scope.name == c.Infra.RK_ROOT or not scope.package_name:
            return None
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        return FlextInfraUtilitiesDocsApi.docstring_coverage(scope.path, contract)

    @staticmethod
    def docs_python_codeblock_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Lint embedded ``python`` fenced blocks under one docs scope.

        Captures every ``python`` fenced block via ``c.Infra.PYTHON_FENCE_RE``
        and gates each block through ``ruff check --stdin-filename`` (piped
        body bytes — no temp files). Failures land as ``m.Infra.AuditIssue``
        records flowing through the standard audit report pipeline.
        """
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        for md_file in FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope):
            rel = md_file.relative_to(scope.path).as_posix()
            content = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )
            for index, match in enumerate(c.Infra.PYTHON_FENCE_RE.finditer(content)):
                # flext-o6h5 (agent: kimi) — ruff via running interpreter (venv SSOT);
                # bare "ruff" breaks when .venv/bin is not on PATH (CI docs audit).
                outcome = u.Cli.run_raw(
                    [
                        sys.executable,
                        "-m",
                        c.Infra.RUFF,
                        c.Infra.VERB_CHECK,
                        "--no-fix",
                        "--extend-ignore",
                        ",".join(c.Infra.PYTHON_FENCE_RUFF_EXTEND_IGNORE),
                        "--stdin-filename",
                        f"{rel}#block{index}.py",
                        "-",
                    ],
                    input_data=match.group("body").encode(),
                )
                if outcome.failure:
                    detail = outcome.error
                elif outcome.value.exit_code == 0:
                    continue
                else:
                    # flext-o6h5 (agent: kimi) — ruff reports parse errors on stderr
                    # only; indexing an empty stdout crashes with IndexError.
                    stdout_lines = outcome.value.stdout.strip().splitlines()
                    stderr_lines = outcome.value.stderr.strip().splitlines()
                    detail_lines = stdout_lines or stderr_lines
                    detail = (
                        detail_lines[-1]
                        if detail_lines
                        else f"ruff exit {outcome.value.exit_code}"
                    )
                issues.append(
                    m.Infra.AuditIssue(
                        file=rel,
                        issue_type="python_codeblock",
                        severity="medium",
                        message=f"block #{index}: {detail}",
                    )
                )
        return issues


__all__: list[str] = ["FlextInfraUtilitiesDocsAuditDetectorsMixin"]
