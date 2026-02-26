"""Documentation generator service.

Generates project-level docs from workspace SSOT guides,
returning structured r reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path

import structlog
from flext_core.result import r
from pydantic import BaseModel, ConfigDict, Field

from flext_infra.constants import c
from flext_infra.docs.shared import (
    DocScope,
    FlextInfraDocsShared,
)
from flext_infra.patterns import FlextInfraPatterns
from flext_infra.templates import TemplateEngine

logger = structlog.get_logger(__name__)


class GeneratedFile(BaseModel):
    """Record of a single generated file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(..., description="File path.")
    written: bool = Field(default=False, description="Whether file was written.")


class GenerateReport(BaseModel):
    """Structured generation report for a scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str = Field(..., description="Scope name.")
    generated: int = Field(default=0, description="Number of files generated.")
    applied: bool = Field(default=False, description="Whether generation was applied.")
    source: str = Field(..., description="Source of generated content.")
    files: list[GeneratedFile] = Field(
        default_factory=list, description="List of generated files."
    )


class DocGenerator:
    """Infrastructure service for documentation generation.

    Generates project-level docs from workspace SSOT guides and
    returns structured r reports.
    """

    def generate(
        self,
        root: Path,
        *,
        project: str | None = None,
        projects: str | None = None,
        output_dir: str = ".reports/docs",
        apply: bool = False,
    ) -> r[list[GenerateReport]]:
        """Generate docs across project scopes.

        Args:
            root: Workspace root directory.
            project: Single project name filter.
            projects: Comma-separated project names.
            output_dir: Report output directory.
            apply: Actually write generated files.

        Returns:
            r with list of GenerateReport objects.

        """
        scopes_result = FlextInfraDocsShared.build_scopes(
            root=root,
            project=project,
            projects=projects,
            output_dir=output_dir,
        )
        if scopes_result.is_failure:
            return r[list[GenerateReport]].fail(scopes_result.error or "scope error")

        reports: list[GenerateReport] = []
        for scope in scopes_result.value:
            report = self._generate_scope(scope, apply=apply, workspace_root=root)
            reports.append(report)

        return r[list[GenerateReport]].ok(reports)

    def _generate_scope(
        self,
        scope: DocScope,
        *,
        apply: bool,
        workspace_root: Path,
    ) -> GenerateReport:
        """Generate docs for a single scope and write reports."""
        if scope.name == "root":
            files = self._generate_root_docs(scope=scope, apply=apply)
            source = "root-generated-artifacts"
        else:
            files = self._generate_project_guides(
                scope=scope, workspace_root=workspace_root, apply=apply
            )
            files.extend(self._generate_project_mkdocs(scope=scope, apply=apply))
            source = "workspace-docs-guides"

        generated = sum(1 for item in files if item.written)
        _ = FlextInfraDocsShared.write_json(
            scope.report_dir / "generate-summary.json",
            {
                "summary": {
                    "scope": scope.name,
                    "generated": generated,
                    "apply": apply,
                    "source": source,
                },
                "files": [{"path": f.path, "written": f.written} for f in files],
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
        result = c.Status.OK if apply else c.Status.WARN
        reason = f"generated:{generated}" if apply else "dry-run"
        logger.info(
            "docs_generate_scope_completed",
            project=scope.name,
            phase="generate",
            result=result,
            reason=reason,
        )

        return GenerateReport(
            scope=scope.name,
            generated=generated,
            applied=apply,
            source=source,
            files=files,
        )

    def _generate_root_docs(
        self,
        scope: DocScope,
        *,
        apply: bool,
    ) -> list[GeneratedFile]:
        """Generate placeholder docs at the workspace root."""
        changelog = self._update_toc(
            "# Changelog\n\nThis file is managed by `make docs DOCS_PHASE=generate`.\n"
        )
        release = self._update_toc(
            "# Latest Release\n\nNo tagged release notes were generated yet.\n"
        )
        roadmap = self._update_toc(
            "# Roadmap\n\nRoadmap updates are generated from docs validation outputs.\n"
        )
        return [
            self._write_if_needed(
                scope.path / "docs/CHANGELOG.md", changelog, apply=apply
            ),
            self._write_if_needed(
                scope.path / "docs/releases/latest.md", release, apply=apply
            ),
            self._write_if_needed(
                scope.path / "docs/roadmap/index.md", roadmap, apply=apply
            ),
        ]

    def _generate_project_guides(
        self,
        scope: DocScope,
        workspace_root: Path,
        *,
        apply: bool,
    ) -> list[GeneratedFile]:
        """Copy workspace guides into a project, injecting the project name."""
        source_dir = workspace_root / "docs/guides"
        if not source_dir.exists():
            return []
        files: list[GeneratedFile] = []
        for source in sorted(source_dir.glob("*.md")):
            rendered = self._project_guide_content(
                content=source.read_text(encoding=c.Encoding.DEFAULT),
                project=scope.name,
                source_name=source.name,
            )
            files.append(
                self._write_if_needed(
                    scope.path / "docs/guides" / source.name, rendered, apply=apply
                )
            )
        return files

    def _generate_project_mkdocs(
        self,
        scope: DocScope,
        *,
        apply: bool,
    ) -> list[GeneratedFile]:
        """Generate mkdocs.yml for projects that do not have one yet."""
        mkdocs_path = scope.path / "mkdocs.yml"
        if mkdocs_path.exists():
            return []
        site_name = f"{scope.name} Documentation"
        content = (
            "\n".join([
                f"site_name: {site_name}",
                f"site_description: Standard guides for {scope.name}",
                f"site_url: {c.Github.GITHUB_REPO_URL}",
                f"repo_name: {c.Github.GITHUB_REPO_NAME}",
                f"repo_url: {c.Github.GITHUB_REPO_URL}",
                f"edit_uri: edit/main/{scope.name}/docs/guides/",
                "docs_dir: docs/guides",
                "site_dir: .reports/docs/site",
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

    @staticmethod
    def _sanitize_internal_anchor_links(content: str) -> str:
        """Normalize generated guides by stripping non-external markdown links."""

        def replace(match: re.Match[str]) -> str:
            label, target = match.groups()
            lower = target.lower().strip()
            if lower.startswith(("http://", "https://", "mailto:", "tel:")):
                return match.group(0)
            return label

        return FlextInfraPatterns.MARKDOWN_LINK_RE.sub(replace, content)

    @staticmethod
    def _normalize_anchor(value: str) -> str:
        """Convert a heading to a GitHub-compatible anchor slug."""
        text = value.strip().lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text.strip("-")

    def _build_toc(self, content: str) -> str:
        """Build a markdown TOC from level-2 and level-3 headings."""
        items: list[str] = []
        for level, title in FlextInfraPatterns.HEADING_H2_H3_RE.findall(content):
            anchor = self._normalize_anchor(title)
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

    def _update_toc(self, content: str) -> str:
        """Insert or replace TOC markers in markdown content."""
        toc = self._build_toc(content)
        if TemplateEngine.TOC_START in content and TemplateEngine.TOC_END in content:
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
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, toc)
            lines.insert(insert_at + 2, "")
            return "\n".join(lines).rstrip() + "\n"
        return toc + "\n\n" + content.rstrip() + "\n"

    @staticmethod
    def _write_if_needed(
        path: Path,
        content: str,
        *,
        apply: bool,
    ) -> GeneratedFile:
        """Write content to path only when changed and apply is True."""
        exists = path.exists()
        current = path.read_text(encoding=c.Encoding.DEFAULT) if exists else ""
        if current == content:
            return GeneratedFile(path=path.as_posix(), written=False)
        if apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(content, encoding=c.Encoding.DEFAULT)
        return GeneratedFile(path=path.as_posix(), written=apply)


__all__ = ["DocGenerator", "GenerateReport", "GeneratedFile"]
