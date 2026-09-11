"""Audit helpers for docs services."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra import c, config, m
from flext_infra._utilities._docs_audit_detectors import (
    FlextInfraUtilitiesDocsAuditDetectorsMixin,
)
from flext_infra._utilities._docs_github_links import FlextInfraUtilitiesDocsGithubLinks
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra._utilities.docs_api import FlextInfraUtilitiesDocsApi
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope

if TYPE_CHECKING:
    from flext_infra.typings import t


class FlextInfraUtilitiesDocsAudit(FlextInfraUtilitiesDocsAuditDetectorsMixin):
    """Reusable audit helpers exposed through ``u.Infra``."""

    @staticmethod
    def docs_is_external(target: str) -> bool:
        """Return whether a docs link target points outside the repository."""
        lower: str = u.norm_str(target, case="lower").lstrip("<")
        return lower.startswith(("http://", "https://", "mailto:", "tel:", "data:"))

    @staticmethod
    def docs_normalize_link(target: str) -> str:
        """Strip fragments and query strings from a markdown link target."""
        value = target.strip()
        if value.startswith("<") and value.endswith(">"):
            value = value[1:-1].strip()
        return value.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]

    @staticmethod
    def docs_should_skip_target(raw: str, target: str) -> bool:
        """Return whether the target should be ignored as prose, not a path."""
        if target.startswith("http"):
            return False
        looks_like_prose = ".md" not in raw and "/" not in raw
        return looks_like_prose and ("," in raw or " " in raw)

    @staticmethod
    def docs_strip_inline_code(line: str) -> str:
        """Remove inline-code spans from one markdown line."""
        pieces: list[str] = []
        in_code = False
        for char in line:
            if char == "`":
                in_code = not in_code
                continue
            if not in_code:
                pieces.append(char)
        return "".join(pieces)

    @staticmethod
    def docs_markdown_link_targets(line: str) -> t.StrSequence:
        """Return markdown link targets from one line."""
        targets: list[str] = []
        index = 0
        while index < len(line):
            close_label = line.find("](", index)
            if close_label < 0:
                break
            close_target = line.find(")", close_label + 2)
            if close_target < 0:
                break
            targets.append(line[close_label + 2 : close_target])
            index = close_target + 1
        return targets

    @staticmethod
    def docs_policy_list(
        scope: m.Infra.DocScope, section: str, key: str
    ) -> t.StrSequence:
        """Read one list of policy tokens from the minimal root docs settings."""
        repository_root = (
            scope.path if scope.name == c.Infra.RK_ROOT else scope.path.parent
        )
        payload = FlextInfraUtilitiesDocsScope.load_config(repository_root)
        container = payload.get(section)
        if not isinstance(container, dict):
            return []
        values = container.get(key)
        return (
            [str(item).strip() for item in values] if isinstance(values, list) else []
        )

    @staticmethod
    def docs_generated_api_reference_path(relative_docs_path: str) -> bool:
        """Return whether a docs path is owned by generated API reference."""
        return relative_docs_path.startswith(
            "api-reference/generated/"
        ) and relative_docs_path.endswith(".md")

    @staticmethod
    def docs_live_public_symbol_names(scope: m.Infra.DocScope) -> set[str]:
        """Return public symbol names that are still exported by one docs scope."""
        if not scope.package_name:
            return set()
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        names: set[str] = set()
        for key in ("exports", "public_symbols"):
            value = contract.get(key)
            if not isinstance(value, list):
                continue
            names.update(item for item in value if isinstance(item, str))
        return names

    @staticmethod
    def docs_broken_link_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect broken internal link issues in one docs scope."""
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        for md_file in FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope):
            rel = md_file.relative_to(scope.path).as_posix()
            content = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )
            in_fenced_code = False
            for number, line in enumerate(content.splitlines(), start=1):
                stripped = line.lstrip()
                if stripped.startswith("```"):
                    in_fenced_code = not in_fenced_code
                    continue
                if in_fenced_code:
                    continue
                clean_line = FlextInfraUtilitiesDocsAudit.docs_strip_inline_code(line)
                for raw in FlextInfraUtilitiesDocsAudit.docs_markdown_link_targets(
                    clean_line
                ):
                    target = FlextInfraUtilitiesDocsAudit.docs_normalize_link(raw)
                    if re.match(
                        config.Infra.codegen.make.docs.cross_project_relative_link_pattern,
                        target,
                    ):
                        issues.append(
                            m.Infra.AuditIssue(
                                file=rel,
                                issue_type="cross_project_relative_link",
                                severity="high",
                                message=(
                                    f"line {number}: cross-project links require an "
                                    f"absolute repository URL -> {raw}"
                                ),
                            )
                        )
                        continue
                    if not target or target.startswith("#"):
                        continue
                    if FlextInfraUtilitiesDocsAudit.docs_is_external(target):
                        issues.extend(
                            FlextInfraUtilitiesDocsGithubLinks.docs_github_link_issues(
                                file=rel, line_number=number, raw=raw, target=target
                            )
                        )
                        continue
                    if FlextInfraUtilitiesDocsAudit.docs_should_skip_target(
                        raw, target
                    ):
                        continue
                    if not (md_file.parent / target).resolve().exists():
                        issues.append(
                            m.Infra.AuditIssue(
                                file=rel,
                                issue_type="broken_link",
                                severity="high",
                                message=f"line {number}: target not found -> {raw}",
                            )
                        )
        return issues

    @staticmethod
    def docs_stale_symbol_issues(
        scope: m.Infra.DocScope,
    ) -> t.SequenceOf[m.Infra.AuditIssue]:
        """Collect stale-symbol issues outside the explicit migration docs."""
        tokens = FlextInfraUtilitiesDocsAudit.docs_policy_list(
            scope, section="audit", key="stale_symbols"
        )
        exempt_paths = set(
            FlextInfraUtilitiesDocsAudit.docs_policy_list(
                scope, section="audit", key="stale_symbol_exempt_paths"
            )
        )
        issues: t.MutableSequenceOf[m.Infra.AuditIssue] = []
        if not tokens:
            return issues
        live_public_symbols = (
            FlextInfraUtilitiesDocsAudit.docs_live_public_symbol_names(scope)
        )
        for md_file in FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope):
            rel = md_file.relative_to(scope.path / c.Infra.DIR_DOCS).as_posix()
            if rel in exempt_paths:
                continue
            is_generated_api_reference = (
                FlextInfraUtilitiesDocsAudit.docs_generated_api_reference_path(rel)
            )
            text = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )
            for token in tokens:
                if token not in text:
                    continue
                if is_generated_api_reference and token in live_public_symbols:
                    continue
                issues.append(
                    m.Infra.AuditIssue(
                        file=md_file.relative_to(scope.path).as_posix(),
                        issue_type="stale_symbol",
                        severity="medium",
                        message=f"contains `{token}`",
                    )
                )
        return issues

    @staticmethod
    def docs_audit_markdown(
        scope: m.Infra.DocScope,
        issues: t.SequenceOf[m.Infra.AuditIssue],
        docstring_coverage: m.Infra.DocstringCoverage | None = None,
    ) -> t.StrSequence:
        """Render the standard markdown audit report."""
        metric_lines: t.MutableSequenceOf[str] = []
        if docstring_coverage is not None:
            metric_lines.append(
                "Docstring coverage: "
                f"{docstring_coverage.percent}% "
                f"({docstring_coverage.documented}/{docstring_coverage.checked})"
            )
        return [
            "# Docs Audit Report",
            "",
            f"Scope: {scope.name}",
            f"Files scanned: {len(FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope))}",
            f"Issues: {len(issues)}",
            *metric_lines,
            "",
            "| file | type | severity | message |",
            "|---|---|---|---|",
            *[
                f"| {issue.file} | {issue.issue_type} | {issue.severity} | {issue.message} |"
                for issue in issues
            ],
        ]


__all__: list[str] = ["FlextInfraUtilitiesDocsAudit"]
