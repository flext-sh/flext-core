"""Documentation generator service.

Generates project-level docs from workspace SSOT guides,
returning structured r reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

from flext_core import FlextLogger, r
from flext_infra import FlextInfraDocsShared, c, m, u

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraDocGenerator:
    """Infrastructure service for documentation generation.

    Generates project-level docs from workspace SSOT guides and
    returns structured r reports.
    """

    @staticmethod
    def _normalize_anchor(value: str) -> str:
        """Convert a heading to a GitHub-compatible anchor slug."""
        text = value.strip().lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-")

    @staticmethod
    def _sanitize_internal_anchor_links(content: str) -> str:
        """Normalize generated guides by stripping non-external markdown links."""

        def replace(match: re.Match[str]) -> str:
            label, target = match.groups()
            lower = target.lower().strip()
            if lower.startswith(("http://", "https://", "mailto:", "tel:")):
                return match.group(0)
            return label

        return u.Infra.MARKDOWN_LINK_RE.sub(replace, content)

    @staticmethod
    def _write_if_needed(
        path: Path,
        content: str,
        *,
        apply: bool,
    ) -> m.Infra.Docs.GeneratedFile:
        """Write content to path only when changed and apply is True."""
        exists = path.exists()
        current = path.read_text(encoding=c.Infra.Encoding.DEFAULT) if exists else ""
        if current == content:
            return m.Infra.Docs.GeneratedFile(path=path.as_posix(), written=False)
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(content, encoding=c.Infra.Encoding.DEFAULT)
        return m.Infra.Docs.GeneratedFile(path=path.as_posix(), written=apply)

    def generate(
        self,
        workspace_root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR,
        apply: bool = False,
    ) -> r[list[m.Infra.Docs.DocsPhaseReport]]:
        """Generate docs across project scopes.

        Args:
            workspace_root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            apply: Actually write generated files.

        Returns:
            r with list of GenerateReport objects.

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
            report = self._generate_scope(
                scope, apply=apply, workspace_root=workspace_root
            )
            reports.append(report)
        return r[list[m.Infra.Docs.DocsPhaseReport]].ok(reports)

    def _build_toc(self, content: str) -> str:
        """Build a markdown TOC from level-2 and level-3 headings."""
        items: list[str] = []
        for level, title in u.Infra.HEADING_H2_H3_RE.findall(content):
            anchor = self._normalize_anchor(title)
            if not anchor:
                continue
            indent = "  " if level == "###" else ""
            items.append(f"{indent}- [{title}](#{anchor})")
        if not items:
            items = ["- No sections found"]
        return f"{u.Infra.TOC_START}\n" + "\n".join(items) + f"\n{u.Infra.TOC_END}"

    def _generate_project_guides(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        workspace_root: Path,
        *,
        apply: bool,
    ) -> list[m.Infra.Docs.GeneratedFile]:
        """Copy workspace guides into a project, injecting the project name."""
        source_dir = workspace_root / "docs/guides"
        if not source_dir.exists():
            return []
        files: list[m.Infra.Docs.GeneratedFile] = []
        for source in sorted(source_dir.glob("*.md")):
            rendered = self._project_guide_content(
                content=source.read_text(encoding=c.Infra.Encoding.DEFAULT),
                project=scope.name,
                source_name=source.name,
            )
            files.append(
                self._write_if_needed(
                    scope.path / "docs/guides" / source.name,
                    rendered,
                    apply=apply,
                ),
            )
        return files

    def _generate_project_mkdocs(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        apply: bool,
    ) -> list[m.Infra.Docs.GeneratedFile]:
        """Generate mkdocs.yml for projects that do not have one yet."""
        mkdocs_path = scope.path / "mkdocs.yml"
        if mkdocs_path.exists():
            return []
        site_name = f"{scope.name} Documentation"
        content = (
            "\n".join([
                f"site_name: {site_name}",
                f"site_description: Standard guides for {scope.name}",
                f"site_url: {c.Infra.Github.GITHUB_REPO_URL}",
                f"repo_name: {c.Infra.Github.GITHUB_REPO_NAME}",
                f"repo_url: {c.Infra.Github.GITHUB_REPO_URL}",
                f"edit_uri: edit/main/{scope.name}/docs/guides/",
                "docs_dir: docs/guides",
                f"site_dir: {c.Infra.Docs.DEFAULT_DOCS_OUTPUT_DIR}/site",
                "",
                "theme:",
                "  name: mkdocs",
                "",
                "plugins: []",
                "",
                "nav:",
                "  - Home: README.md",
                "  - Getting Started: getting-started.md",
                "  - Configuration: configuration.md",
                "  - Development: development.md",
                "  - Testing: testing.md",
                "  - Troubleshooting: troubleshooting.md",
                "  - Security: security.md",
                "  - Automation Skill Pattern: skill-automation-pattern.md",
            ])
            + "\n"
        )
        return [self._write_if_needed(mkdocs_path, content, apply=apply)]

    def _generate_root_docs(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        apply: bool,
    ) -> list[m.Infra.Docs.GeneratedFile]:
        """Generate placeholder docs at the workspace root."""
        changelog = self._update_toc(
            "# Changelog\n\nThis file is managed by `make docs DOCS_PHASE=generate`.\n",
        )
        release = self._update_toc(
            "# Latest Release\n\nNo tagged release notes were generated yet.\n",
        )
        roadmap = self._update_toc(
            "# Roadmap\n\nRoadmap updates are generated from docs validation outputs.\n",
        )
        return [
            self._write_if_needed(
                scope.path / "docs/CHANGELOG.md",
                changelog,
                apply=apply,
            ),
            self._write_if_needed(
                scope.path / "docs/releases/latest.md",
                release,
                apply=apply,
            ),
            self._write_if_needed(
                scope.path / "docs/roadmap/index.md",
                roadmap,
                apply=apply,
            ),
        ]

    def _generate_scope(
        self,
        scope: m.Infra.Docs.FlextInfraDocScope,
        *,
        apply: bool,
        workspace_root: Path,
    ) -> m.Infra.Docs.DocsPhaseReport:
        """Generate docs for a single scope and write reports."""
        if scope.name == c.Infra.ReportKeys.ROOT:
            files = self._generate_root_docs(scope=scope, apply=apply)
            source = "root-generated-artifacts"
        else:
            files = self._generate_project_guides(
                scope=scope,
                workspace_root=workspace_root,
                apply=apply,
            )
            files.extend(self._generate_project_mkdocs(scope=scope, apply=apply))
            source = "workspace-docs-guides"
        generated = sum(1 for item in files if item.written)
        _ = u.Infra.write_json(
            scope.report_dir / "generate-summary.json",
            {
                c.Infra.ReportKeys.SUMMARY: {
                    c.Infra.ReportKeys.SCOPE: scope.name,
                    "generated": generated,
                    "apply": apply,
                    "source": source,
                },
                "files": [
                    {c.Infra.Toml.PATH: f.path, "written": f.written} for f in files
                ],
            },
        )
        _ = FlextInfraDocsShared.write_markdown(
            scope.report_dir / "generate-report.md",
            [
                "# Docs Generate Report",
                "",
                f"Scope: {scope.name}",
                f"Apply: {int(apply)}",
                f"Generated files: {generated}",
                f"Source: {source}",
            ],
        )
        result = c.Infra.Status.OK if apply else c.Infra.Status.WARN
        reason = f"generated:{generated}" if apply else "dry-run"
        logger.info(
            "docs_generate_scope_completed",
            project=scope.name,
            phase="generate",
            result=result,
            reason=reason,
        )
        return m.Infra.Docs.DocsPhaseReport(
            phase="generate",
            scope=scope.name,
            generated=generated,
            applied=apply,
            source=source,
            items=[
                m.Infra.Docs.DocsPhaseItem(
                    phase="generate",
                    path=file.path,
                    written=file.written,
                )
                for file in files
            ],
            result=result,
            reason=reason,
            passed=apply,
        )

    def _project_guide_content(
        self,
        content: str,
        project: str,
        source_name: str,
    ) -> str:
        """Render workspace guide content with project-specific heading."""
        lines = content.splitlines()
        out: list[str] = [
            f"<!-- Generated from docs/guides/{source_name} for {project}. -->",
            "<!-- Source of truth: workspace docs/guides/. -->",
            "",
        ]
        heading_done = False
        for line in lines:
            if not heading_done and line.startswith("# "):
                title = line[2:].strip()
                out.extend([
                    f"# {project} - {title}",
                    "",
                    f"> Project profile: `{project}`",
                    "",
                ])
                heading_done = True
                continue
            out.append(line)
        rendered = "\n".join(out).rstrip() + "\n"
        return self._update_toc(self._sanitize_internal_anchor_links(rendered))

    def _update_toc(self, content: str) -> str:
        """Insert or replace TOC markers in markdown content."""
        toc = self._build_toc(content)
        if u.Infra.TOC_START in content and u.Infra.TOC_END in content:
            return re.sub(
                r"<!-- TOC START -->.*?<!-- TOC END -->",
                toc,
                content,
                count=1,
                flags=re.DOTALL,
            )
        lines = content.splitlines()
        if lines and lines[0].startswith("# "):
            insert_at = 1
            while insert_at < len(lines) and (not lines[insert_at].strip()):
                insert_at += 1
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, toc)
            lines.insert(insert_at + 2, "")
            return "\n".join(lines).rstrip() + "\n"
        return toc + "\n\n" + content.rstrip() + "\n"


__all__ = ["FlextInfraDocGenerator"]
