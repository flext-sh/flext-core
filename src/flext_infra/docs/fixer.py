"""Documentation fixer service.

Auto-fixes broken links and inserts/updates TOC in markdown files,
returning structured r reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from flext_core.loggings import FlextLogger
from flext_core.result import r
from flext_core.typings import t
from pydantic import BaseModel, ConfigDict, Field

from flext_infra.constants import c
from flext_infra.docs.shared import (
    DEFAULT_DOCS_OUTPUT_DIR,
    DocScope,
    FlextInfraDocsShared,
)
from flext_infra.patterns import FlextInfraPatterns
from flext_infra.templates import TemplateEngine

logger = FlextLogger.create_module_logger(__name__)


class FixItem(BaseModel):
    """Per-file summary of applied fixes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str = Field(..., description="File path relative to scope.")
    links: int = Field(default=0, description="Number of link fixes applied.")
    toc: int = Field(default=0, description="Number of TOC updates applied.")


class FixReport(BaseModel):
    """Structured fix report for a scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str = Field(..., description="Scope name.")
    changed_files: int = Field(default=0, description="Number of files changed.")
    applied: bool = Field(default=False, description="Whether changes were applied.")
    items: list[FixItem] = Field(default_factory=list, description="List of fix items.")


class DocFixer:
    """Infrastructure service for documentation auto-fixing.

    Fixes broken markdown links and inserts/updates TOC blocks,
    returning structured r reports.
    """

    def fix(
        self,
        root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = DEFAULT_DOCS_OUTPUT_DIR,
        apply: bool = False,
    ) -> r[list[FixReport]]:
        """Run documentation fixes across project scopes.

        Args:
            root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            apply: Actually write changes (dry-run if False).

        Returns:
            r with list of FixReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[FixReport]].fail(scopes_result.error or "scope error")

        reports: list[FixReport] = []
        for scope in scopes_result.value:
            report = self._fix_scope(scope, apply=apply)
            reports.append(report)

        return r[list[FixReport]].ok(reports)

    def _fix_scope(self, scope: DocScope, *, apply: bool) -> FixReport:
        """Run link and TOC fixes across all markdown files in scope."""
        items: list[FixItem] = []
        for md in FlextInfraDocsShared.iter_markdown_files(scope.path):
            item = self._process_file(md, apply=apply)
            if item.links or item.toc:
                rel = md.relative_to(scope.path).as_posix()
                items.append(FixItem(file=rel, links=item.links, toc=item.toc))

        changes_payload: list[Mapping[str, t.ConfigMapValue]] = [
            {
                "file": item.file,
                "links": item.links,
                "toc": item.toc,
            }
            for item in items
        ]
        payload: Mapping[str, t.ConfigMapValue] = {
            "summary": {
                "scope": scope.name,
                "changed_files": len(items),
                "apply": apply,
            },
            "changes": changes_payload,
        }
        _ = FlextInfraDocsShared.write_json(
            scope.report_dir / "fix-summary.json",
            payload,
        )
        lines = [
            "# Docs Fix Report",
            "",
            f"Scope: {scope.name}",
            f"Apply: {int(apply)}",
            f"Changed files: {len(items)}",
            "",
            "| file | link_fixes | toc_updates |",
            "|---|---:|---:|",
            *[f"| {item.file} | {item.links} | {item.toc} |" for item in items],
        ]
        _ = FlextInfraDocsShared.write_markdown(
            scope.report_dir / "fix-report.md",
            lines,
        )

        status = c.Status.OK if apply or not items else c.Status.WARN
        logger.info(
            "docs_fix_scope_completed",
            project=scope.name,
            phase="fix",
            result=status,
            reason=f"changes:{len(items)}",
        )

        return FixReport(
            scope=scope.name,
            changed_files=len(items),
            applied=apply,
            items=items,
        )

    def _process_file(self, md_file: Path, *, apply: bool) -> FixItem:
        """Fix links and TOC in a single markdown file."""
        original = md_file.read_text(encoding=c.Encoding.DEFAULT, errors="ignore")
        link_count = 0

        def replace_link(match: re.Match[str]) -> str:
            nonlocal link_count
            text, link = match.groups()
            fixed = self._maybe_fix_link(md_file, link)
            if fixed is None:
                return match.group(0)
            link_count += 1
            return f"[{text}]({fixed})"

        updated = FlextInfraPatterns.MARKDOWN_LINK_RE.sub(replace_link, original)
        updated, toc_changed = self._update_toc(updated)
        if apply and (link_count > 0 or toc_changed > 0) and updated != original:
            _ = md_file.write_text(updated, encoding=c.Encoding.DEFAULT)
        return FixItem(file=md_file.as_posix(), links=link_count, toc=toc_changed)

    @staticmethod
    def _maybe_fix_link(md_file: Path, raw_link: str) -> str | None:
        """Return a corrected link target or None if no fix is needed."""
        if raw_link.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            return None
        base = raw_link.split("#", maxsplit=1)[0]
        if not base:
            return None
        if (md_file.parent / base).exists():
            return None
        if not base.endswith(".md"):
            md_candidate = md_file.parent / f"{base}.md"
            if md_candidate.exists():
                suffix = raw_link[len(base) :]
                return f"{base}.md{suffix}"
        return None

    @staticmethod
    def _anchorize(text: str) -> str:
        """Convert a heading title to a GitHub-compatible anchor slug."""
        value = text.strip().lower()
        value = re.sub(r"[^a-z0-9\s-]", "", value)
        value = re.sub(r"\s+", "-", value)
        return re.sub(r"-+", "-", value).strip("-")

    def _build_toc(self, content: str) -> str:
        """Generate a TOC block from ## and ### headings in content."""
        items: list[str] = []
        for level, title in FlextInfraPatterns.HEADING_H2_H3_RE.findall(content):
            anchor = self._anchorize(title)
            if not anchor:
                continue
            indent = "  " if level == "###" else ""
            items.append(f"{indent}- [{title}](#{anchor})")
        if not items:
            items = ["- No sections found"]
        return (
            f"{TemplateEngine.TOC_START}\n"
            + "\n".join(items)
            + f"\n{TemplateEngine.TOC_END}"
        )

    def _update_toc(self, content: str) -> tuple[str, int]:
        """Insert or replace the TOC in content, returning (updated, changed)."""
        toc = self._build_toc(content)
        if TemplateEngine.TOC_START in content and TemplateEngine.TOC_END in content:
            updated = re.sub(
                r"<!-- TOC START -->.*?<!-- TOC END -->",
                toc,
                content,
                count=1,
                flags=re.DOTALL,
            )
            return updated, int(updated != content)
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            insert_at = 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, toc)
            lines.insert(insert_at + 2, "")
            return "\n".join(lines) + ("\n" if content.endswith("\n") else ""), 1
        return toc + "\n\n" + content, 1


__all__ = ["DocFixer", "FixItem", "FixReport"]
