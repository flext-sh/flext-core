from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from _shared import (
    Scope,
    build_scopes,
    iter_markdown_files,
    write_json,
    write_markdown,
)

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(##|###)\s+(.+?)\s*$", re.MULTILINE)
TOC_START = "<!-- TOC START -->"
TOC_END = "<!-- TOC END -->"


@dataclass(frozen=True)
class FixItem:
    file: str
    links: int
    toc: int


def maybe_fix_link(md_file: Path, raw_link: str) -> str | None:
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


def anchorize(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"\s+", "-", value)
    return re.sub(r"-+", "-", value)


def build_toc(content: str) -> str:
    items: list[str] = []
    for level, title in HEADING_RE.findall(content):
        anchor = anchorize(title)
        if not anchor:
            continue
        indent = "  " if level == "###" else ""
        items.append(f"{indent}- [{title}](#{anchor})")
    if not items:
        items = ["- No sections found"]
    return f"{TOC_START}\n" + "\n".join(items) + f"\n{TOC_END}"


def update_toc(content: str) -> tuple[str, int]:
    toc = build_toc(content)
    if TOC_START in content and TOC_END in content:
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


def process_file(md_file: Path, apply: bool) -> FixItem:
    original = md_file.read_text(encoding="utf-8", errors="ignore")
    link_count = 0

    def replace_link(match: re.Match[str]) -> str:
        nonlocal link_count
        text, link = match.groups()
        fixed = maybe_fix_link(md_file, link)
        if fixed is None:
            return match.group(0)
        link_count += 1
        return f"[{text}]({fixed})"

    updated = LINK_RE.sub(replace_link, original)
    updated, toc_changed = update_toc(updated)
    if apply and (link_count > 0 or toc_changed > 0) and updated != original:
        md_file.write_text(updated, encoding="utf-8")
    return FixItem(file=md_file.as_posix(), links=link_count, toc=toc_changed)


def run_scope(scope: Scope, apply: bool) -> int:
    items: list[FixItem] = []
    for md in iter_markdown_files(scope.path):
        item = process_file(md, apply=apply)
        if item.links or item.toc:
            rel = md.relative_to(scope.path).as_posix()
            items.append(FixItem(file=rel, links=item.links, toc=item.toc))

    payload = {
        "summary": {
            "scope": scope.name,
            "changed_files": len(items),
            "apply": apply,
        },
        "changes": [asdict(item) for item in items],
    }
    write_json(scope.report_dir / "fix-summary.json", payload)
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
    write_markdown(scope.report_dir / "fix-report.md", lines)
    result = "OK" if apply or not items else "WARN"
    print(f"PROJECT={scope.name} PHASE=fix RESULT={result} REASON=changes:{len(items)}")
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
        run_scope(scope, apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
