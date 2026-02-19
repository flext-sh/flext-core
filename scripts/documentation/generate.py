from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _shared import Scope, build_scopes, write_json, write_markdown


@dataclass(frozen=True)
class GeneratedFile:
    path: str
    written: bool


def write_if_needed(path: Path, content: str, apply: bool) -> GeneratedFile:
    exists = path.exists()
    current = path.read_text(encoding="utf-8") if exists else ""
    if current == content:
        return GeneratedFile(path=path.as_posix(), written=False)
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return GeneratedFile(path=path.as_posix(), written=apply)


def project_guide_content(content: str, project: str, source_name: str) -> str:
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
            out.append(f"# {project} - {title}")
            out.append("")
            out.append(f"> Project profile: `{project}`")
            out.append("")
            heading_done = True
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def generate_root_docs(scope: Scope, apply: bool) -> list[GeneratedFile]:
    changelog = (
        "# Changelog\n\nThis file is managed by `make docs DOCS_PHASE=generate`.\n"
    )
    release = "# Latest Release\n\nNo tagged release notes were generated yet.\n"
    roadmap = (
        "# Roadmap\n\nRoadmap updates are generated from docs validation outputs.\n"
    )
    return [
        write_if_needed(scope.path / "docs/CHANGELOG.md", changelog, apply=apply),
        write_if_needed(scope.path / "docs/releases/latest.md", release, apply=apply),
        write_if_needed(scope.path / "docs/roadmap/index.md", roadmap, apply=apply),
    ]


def generate_project_guides(
    scope: Scope, workspace_root: Path, apply: bool
) -> list[GeneratedFile]:
    source_dir = workspace_root / "docs/guides"
    if not source_dir.exists():
        return []
    files: list[GeneratedFile] = []
    for source in sorted(source_dir.glob("*.md")):
        rendered = project_guide_content(
            content=source.read_text(encoding="utf-8"),
            project=scope.name,
            source_name=source.name,
        )
        files.append(
            write_if_needed(
                scope.path / "docs/guides" / source.name, rendered, apply=apply
            )
        )
    return files


def run_scope(scope: Scope, apply: bool, workspace_root: Path) -> int:
    if scope.name == "root":
        files = generate_root_docs(scope=scope, apply=apply)
        source = "root-generated-artifacts"
    else:
        files = generate_project_guides(
            scope=scope, workspace_root=workspace_root, apply=apply
        )
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
    result = "OK" if apply else "WARN"
    reason = f"generated:{generated}" if apply else "dry-run"
    print(f"PROJECT={scope.name} PHASE=generate RESULT={result} REASON={reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--project")
    parser.add_argument("--projects")
    parser.add_argument("--output-dir", default=".reports/docs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scopes = build_scopes(
        root=root,
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
    )
    for scope in scopes:
        run_scope(scope=scope, apply=args.apply, workspace_root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
