from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _shared import write_json

ECOSYSTEM_LINE = "Part of the [FLEXT](https://github.com/flext-sh/flext) ecosystem."


@dataclass(frozen=True)
class Result:
    file: str
    changed: bool
    issues: int


def target_projects(root: Path, project_names: list[str] | None) -> list[Path]:
    if not project_names:
        return [
            p
            for p in sorted(root.iterdir())
            if p.is_dir() and (p / "README.md").exists()
        ]
    targets: list[Path] = []
    for name in project_names:
        path = root / name
        if path.is_dir():
            targets.append(path)
    return targets


def normalize_content(name: str, content: str) -> tuple[str, int]:
    lines = content.splitlines()
    issues = 0
    title = f"# {name}"
    if not lines or not lines[0].startswith("# "):
        lines = [title, ""] + lines
        issues += 1
    if ECOSYSTEM_LINE not in content:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(ECOSYSTEM_LINE)
        issues += 1
    updated = "\n".join(lines).rstrip() + "\n"
    return updated, issues


def process_readme(path: Path, apply: bool) -> Result:
    original = path.read_text(encoding="utf-8", errors="ignore")
    updated, issues = normalize_content(path.parent.name, original)
    changed = updated != original
    if apply and changed:
        path.write_text(updated, encoding="utf-8")
    return Result(file=path.as_posix(), changed=changed and apply, issues=issues)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--projects", nargs="*")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    apply = args.fix and not args.check
    root = Path(args.root).resolve()
    results: list[Result] = []
    for project in target_projects(root, args.projects):
        readme = project / "README.md"
        if readme.exists():
            result = process_readme(readme, apply=apply)
            results.append(result)
            print(
                f"{project.name}: issues={result.issues} changed={int(result.changed)}"
            )

    report_path = (root / ".reports/docs/readme-standardizer-summary.json").resolve()
    write_json(
        report_path,
        {
            "summary": {
                "projects": len(results),
                "issues": sum(item.issues for item in results),
                "fix_mode": apply,
            },
            "results": [item.__dict__ for item in results],
        },
    )
    return 0 if all(item.issues == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
