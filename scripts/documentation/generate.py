#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Generate project-level docs from workspace SSOT guides."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from scripts.documentation.shared import Scope, build_scopes, write_json, write_markdown
from scripts.libs.config import (
    DEFAULT_ENCODING,
    GITHUB_REPO_NAME,
    GITHUB_REPO_URL,
    STATUS_OK,
    STATUS_WARN,
)
from scripts.libs.doc_patterns import HEADING_H2_H3_RE, MARKDOWN_LINK_RE
from scripts.libs.templates import TOC_END, TOC_START


@dataclass(frozen=True)
class GeneratedFile:
    """Record of a single generated file and whether it was written."""

    path: str
    written: bool


def normalize_anchor(value: str) -> str:
    """Convert a heading to a GitHub-compatible anchor slug."""
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def sanitize_internal_anchor_links(content: str) -> str:
    """Normalize generated guides by stripping non-external markdown links."""

    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        lower = target.lower().strip()
        if lower.startswith(("http://", "https://", "mailto:", "tel:")):
            return match.group(0)
        return label

    return MARKDOWN_LINK_RE.sub(replace, content)


def build_toc(content: str) -> str:
    """Build a markdown TOC from level-2 and level-3 headings."""
    items: list[str] = []
    for level, title in HEADING_H2_H3_RE.findall(content):
        anchor = normalize_anchor(title)
        if not anchor:
            continue
        indent = "  " if level == "###" else ""
        items.append(f"{indent}- [{title}](#{anchor})")
    if not items:
        items = ["- No sections found"]
    return f"{TOC_START}\n" + "\n".join(items) + f"\n{TOC_END}"


def update_toc(content: str) -> str:
    """Insert or replace TOC markers in markdown content."""
    toc = build_toc(content)
    if TOC_START in content and TOC_END in content:
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


def write_if_needed(path: Path, content: str, *, apply: bool) -> GeneratedFile:
    """Write *content* to *path* only when changed and *apply* is True."""
    exists = path.exists()
    current = path.read_text(encoding=DEFAULT_ENCODING) if exists else ""
    if current == content:
        return GeneratedFile(path=path.as_posix(), written=False)
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(content, encoding=DEFAULT_ENCODING)
    return GeneratedFile(path=path.as_posix(), written=apply)


def project_guide_content(content: str, project: str, source_name: str) -> str:
    """Render workspace guide *content* with a project-specific heading."""
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
    return update_toc(sanitize_internal_anchor_links(rendered))


def generate_root_docs(scope: Scope, *, apply: bool) -> list[GeneratedFile]:
    """Generate placeholder docs at the workspace root."""
    changelog = update_toc(
        "# Changelog\n\nThis file is managed by `make docs DOCS_PHASE=generate`.\n"
    )
    release = update_toc(
        "# Latest Release\n\nNo tagged release notes were generated yet.\n"
    )
    roadmap = update_toc(
        "# Roadmap\n\nRoadmap updates are generated from docs validation outputs.\n"
    )
    return [
        write_if_needed(scope.path / "docs/CHANGELOG.md", changelog, apply=apply),
        write_if_needed(scope.path / "docs/releases/latest.md", release, apply=apply),
        write_if_needed(scope.path / "docs/roadmap/index.md", roadmap, apply=apply),
    ]


def generate_project_guides(
    scope: Scope, workspace_root: Path, *, apply: bool
) -> list[GeneratedFile]:
    """Copy workspace guides into a project, injecting the project name."""
    source_dir = workspace_root / "docs/guides"
    if not source_dir.exists():
        return []
    files: list[GeneratedFile] = []
    for source in sorted(source_dir.glob("*.md")):
        rendered = project_guide_content(
            content=source.read_text(encoding=DEFAULT_ENCODING),
            project=scope.name,
            source_name=source.name,
        )
        files.append(
            write_if_needed(
                scope.path / "docs/guides" / source.name, rendered, apply=apply
            )
        )
    return files


def generate_project_mkdocs(scope: Scope, *, apply: bool) -> list[GeneratedFile]:
    """Generate mkdocs.yml for projects that do not have one yet."""
    mkdocs_path = scope.path / "mkdocs.yml"
    if mkdocs_path.exists():
        return []
    site_name = f"{scope.name} Documentation"
    content = (
        "\n".join([
            f"site_name: {site_name}",
            f"site_description: Standard guides for {scope.name}",
            f"site_url: {GITHUB_REPO_URL}",
            f"repo_name: {GITHUB_REPO_NAME}",
            f"repo_url: {GITHUB_REPO_URL}",
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
    return [write_if_needed(mkdocs_path, content, apply=apply)]


def run_scope(scope: Scope, *, apply: bool, workspace_root: Path) -> int:
    """Generate docs for *scope* and write reports."""
    if scope.name == "root":
        files = generate_root_docs(scope=scope, apply=apply)
        source = "root-generated-artifacts"
    else:
        files = generate_project_guides(
            scope=scope, workspace_root=workspace_root, apply=apply
        )
        files.extend(generate_project_mkdocs(scope=scope, apply=apply))
        source = "workspace-docs-guides"

    generated = sum(1 for item in files if item.written)
    write_json(
        scope.report_dir / "generate-summary.json",
        {
            "summary": {
                "scope": scope.name,
                "generated": generated,
                "apply": apply,
                "source": source,
            },
            "files": [item.__dict__ for item in files],
        },
    )
    write_markdown(
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
    result = STATUS_OK if apply else STATUS_WARN
    reason = f"generated:{generated}" if apply else "dry-run"
    print(f"PROJECT={scope.name} PHASE=generate RESULT={result} REASON={reason}")
    return 0


def main() -> int:
    """CLI entry point for the documentation generator."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", default=".")
    _ = parser.add_argument("--project")
    _ = parser.add_argument("--projects")
    _ = parser.add_argument("--output-dir", default=".reports/docs")
    _ = parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scopes = build_scopes(
        root=root,
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
    )
    for scope in scopes:
        _ = run_scope(scope=scope, apply=args.apply, workspace_root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
