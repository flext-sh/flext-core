"""Documentation fixer service.

Auto-fixes broken links and inserts/updates TOC in markdown files,
returning structured r reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import JsonValue

from flext_core import FlextLogger, r
from flext_infra import FlextInfraDocsShared, c, m, u

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraDocFixer:
    """Infrastructure service for documentation auto-fixing.

    Fixes broken markdown links and inserts/updates TOC blocks,
    returning structured r reports.
    """

    @staticmethod
    def _anchorize(text: str) -> str:
        """Convert a heading title to a GitHub-compatible anchor slug."""
        value = text.strip().lower()
        value = re.sub(r"[^a-z0-9\s-]", "", value)
        value = re.sub(r"\s+", "-", value)
        return re.sub(r"-+", "-", value).strip("-")

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

    def fix(
        self,
        workspace_root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR,
        apply: bool = False,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        """Run documentation fixes across project scopes.

        Args:
            workspace_root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            apply: Actually write changes (dry-run if False).

        Returns:
            r with list of FixReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            workspace_root=workspace_root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[m.Infra.Docs.DocsPhaseReport]].fail(
                scopes_result.error or "scope error",
            )
        reports: list[m.Infra.Docs.DocsPhaseReport] = []
        for scope in scopes_result.value:
            report = self._fix_scope(scope, apply=apply)
            reports.append(report)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(reports)

    def _build_toc(self, content: str) -> str:
        """Generate a TOC block from ## and ### headings in content."""
        items: list[str] = []
        for level, title in u.Infra.HEADING_H2_H3_RE.findall(content):
            anchor = self._anchorize(title)
            if not anchor:
                continue
            indent = "  " if level == "###" else ""
            items.append(f"{indent}- [{title}](#{anchor})")
        if not items:
            items = ["- No sections found"]
        return f"{u.Infra.TOC_START}\n" + "\n".join(items) + f"\n{u.Infra.TOC_END}"

    def _fix_scope(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        apply: bool,
    ) -> m.Infra.Docs.DocsPhaseReport:
        """Run link and TOC fixes across all markdown files in scope."""
        items: list[m.Infra.Docs.DocsPhaseItem] = []
        for md in FlextInfraDocsShared.iter_markdown_files(scope.path):
            item = self._process_file(md, apply=apply)
            if item.links or item.toc:
                rel = md.relative_to(scope.path).as_posix()
                items.append(
                    m.Infra.Docs.DocsPhaseItem(
                        phase="fix",
                        file=rel,
                        links=item.links,
                        toc=item.toc,
                    ),
                )
        changes_payload: JsonValue = [
            {c.Infra.ReportKeys.FILE: item.file, "links": item.links, "toc": item.toc}
            for item in items
        ]
        payload: JsonValue = {
            c.Infra.ReportKeys.SUMMARY: {
                c.Infra.ReportKeys.SCOPE: scope.name,
                "changed_files": len(items),
                "apply": apply,
            },
            "changes": changes_payload,
        }
        _ = u.Infra.write_json(scope.report_dir / "fix-summary.json", payload)
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
        status = c.Infra.Status.OK if apply or not items else c.Infra.Status.WARN
        logger.info(
            "docs_fix_scope_completed",
            project=scope.name,
            phase="fix",
            result=status,
            reason=f"changes:{len(items)}",
        )
        return m.Infra.Docs.DocsPhaseReport(
            phase="fix",
            scope=scope.name,
            changed_files=len(items),
            applied=apply,
            items=items,
            result=status,
            reason=f"changes:{len(items)}",
            passed=apply or not items,
        )

    def _process_file(
        self,
        md_file: Path,
        *,
        apply: bool,
    ) -> m.Infra.Docs.DocsPhaseItem:
        """Fix links and TOC in a single markdown file."""
        original = md_file.read_text(
            encoding=c.Infra.Encoding.DEFAULT,
            errors=c.Infra.Toml.IGNORE,
        )
        link_count = 0

        def replace_link(match: re.Match[str]) -> str:
            nonlocal link_count
            text, link = match.groups()
            fixed = self._maybe_fix_link(md_file, link)
            if fixed is None:
                return match.group(0)
            link_count += 1
            return f"[{text}]({fixed})"

        updated = u.Infra.MARKDOWN_LINK_RE.sub(replace_link, original)
        updated, toc_changed = self._update_toc(updated)
        if apply and (link_count > 0 or toc_changed > 0) and (updated != original):
            _ = md_file.write_text(updated, encoding=c.Infra.Encoding.DEFAULT)
        return m.Infra.Docs.DocsPhaseItem(
            phase="fix",
            file=md_file.as_posix(),
            links=link_count,
            toc=toc_changed,
        )

    def _update_toc(self, content: str) -> tuple[str, int]:
        """Insert or replace the TOC in content, returning (updated, changed)."""
        toc = self._build_toc(content)
        if u.Infra.TOC_START in content and u.Infra.TOC_END in content:
            updated = re.sub(
                r"<!-- TOC START -->.*?<!-- TOC END -->",
                toc,
                content,
                count=1,
                flags=re.DOTALL,
            )
            return (updated, int(updated != content))
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            insert_at = 1
            while insert_at < len(lines) and (not lines[insert_at].strip()):
                insert_at += 1
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, toc)
            lines.insert(insert_at + 2, "")
            return ("\n".join(lines) + ("\n" if content.endswith("\n") else ""), 1)
        return (toc + "\n\n" + content, 1)


__all__ = ["FlextInfraDocFixer"]
