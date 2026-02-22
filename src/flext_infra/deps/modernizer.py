from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import tomlkit
from tomlkit.items import Table


def _workspace_root(start: Path) -> Path:
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / ".gitmodules").exists() and (parent / "pyproject.toml").exists():
            return parent
    for parent in (current, *current.parents):
        if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
            return parent
    return start.resolve().parents[4]


ROOT = _workspace_root(Path(__file__))
SKIP_DIRS = frozenset({
    ".archive",
    ".claude.disabled",
    ".flext-deps",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".sisyphus",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "vendor",
})

_DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _dep_name(spec: str) -> str:
    base = spec.strip().split("@", 1)[0].strip()
    match = _DEP_NAME_RE.match(base)
    if match:
        return match.group(1).lower().replace("_", "-")
    return base.lower().replace("_", "-")


def _dedupe_specs(specs: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for spec in specs:
        key = _dep_name(spec)
        if key and key not in seen:
            seen[key] = spec
    return [seen[k] for k in sorted(seen)]


def _as_string_list(value: object) -> list[str]:
    if value is None or isinstance(value, str) or isinstance(value, Mapping):
        return []
    if not isinstance(value, Iterable):
        return []
    items: list[str] = []
    for raw in value:
        normalized = raw.value if hasattr(raw, "value") else raw
        items.append(str(normalized))
    return items


def _array(items: list[str]) -> object:
    arr = tomlkit.array()
    for item in items:
        arr.append(item)
    return arr.multiline(True)


def _ensure_table(parent: Table, key: str) -> Table:
    existing = parent.get(key)
    if isinstance(existing, Table):
        return existing
    table = tomlkit.table()
    parent[key] = table
    return table


def _read_doc(path: Path) -> tomlkit.TOMLDocument | None:
    if not path.exists():
        return None
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _project_dev_groups(doc: tomlkit.TOMLDocument) -> dict[str, list[str]]:
    project = doc.get("project")
    if not isinstance(project, Table):
        return {}
    optional = project.get("optional-dependencies")
    if not isinstance(optional, Table):
        return {}
    return {
        "dev": _as_string_list(optional.get("dev")),
        "docs": _as_string_list(optional.get("docs")),
        "security": _as_string_list(optional.get("security")),
        "test": _as_string_list(optional.get("test")),
        "typings": _as_string_list(optional.get("typings")),
    }


def _canonical_dev_dependencies(root_doc: tomlkit.TOMLDocument) -> list[str]:
    groups = _project_dev_groups(root_doc)
    merged = [
        *groups.get("dev", []),
        *groups.get("docs", []),
        *groups.get("security", []),
        *groups.get("test", []),
        *groups.get("typings", []),
    ]
    return _dedupe_specs(merged)


class ConsolidateGroupsPhase:
    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        canonical_dev: list[str],
    ) -> list[str]:
        changes: list[str] = []

        project = doc.get("project")
        if not isinstance(project, Table):
            project = tomlkit.table()
            doc["project"] = project

        optional = project.get("optional-dependencies")
        if not isinstance(optional, Table):
            optional = tomlkit.table()
            project["optional-dependencies"] = optional

        existing = _project_dev_groups(doc)
        merged_dev = _dedupe_specs([
            *canonical_dev,
            *existing.get("dev", []),
            *existing.get("docs", []),
            *existing.get("security", []),
            *existing.get("test", []),
            *existing.get("typings", []),
        ])
        current_dev = _as_string_list(optional.get("dev"))
        if current_dev != merged_dev:
            optional["dev"] = _array(merged_dev)
            changes.append("project.optional-dependencies.dev consolidated")

        for old_key in ("docs", "security", "test", "typings"):
            if old_key in optional:
                del optional[old_key]
                changes.append(f"project.optional-dependencies.{old_key} removed")

        tool = doc.get("tool")
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc["tool"] = tool

        poetry = _ensure_table(tool, "poetry")
        poetry_group = _ensure_table(poetry, "group")
        poetry_dev = _ensure_table(_ensure_table(poetry_group, "dev"), "dependencies")

        for old_group in ("docs", "security", "test", "typings"):
            old_group_table = poetry_group.get(old_group)
            if not isinstance(old_group_table, Table):
                continue
            old_deps = old_group_table.get("dependencies")
            if isinstance(old_deps, Table):
                for dep_name, dep_value in old_deps.items():
                    if dep_name not in poetry_dev:
                        poetry_dev[dep_name] = dep_value
            del poetry_group[old_group]
            changes.append(f"tool.poetry.group.{old_group} removed")

        deptry = _ensure_table(tool, "deptry")
        current_groups = _as_string_list(deptry.get("pep621_dev_dependency_groups"))
        if current_groups != ["dev"]:
            deptry["pep621_dev_dependency_groups"] = _array(["dev"])
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")

        return changes


class InjectCommentsPhase:
    _BANNER = "\n".join([
        "# [MANAGED] FLEXT pyproject standardization",
        "# Sections with [MANAGED] are enforced by flext_infra.deps.modernizer.",
        "# Sections with [CUSTOM] are project-specific extension points.",
        "# Sections with [AUTO] are derived from workspace layout and dependencies.",
        "",
    ])

    _MARKERS: tuple[tuple[str, str], ...] = (
        ("[build-system]", "# [MANAGED] build system"),
        ("[project]", "# [CUSTOM] project metadata"),
        ("[tool.poetry.group.dev.dependencies]", "# [CUSTOM] poetry dev extensions"),
        ("[tool.deptry]", "# [MANAGED] deptry"),
        ("[tool.ruff]", "# [MANAGED] ruff"),
        ("[tool.codespell]", "# [MANAGED] codespell"),
        ("[tool.pytest]", "# [MANAGED] pytest"),
        ("[tool.coverage]", "# [MANAGED] coverage"),
        ("[tool.mypy]", "# [MANAGED] mypy"),
        ("[tool.pyrefly]", "# [MANAGED] pyrefly"),
        ("[tool.pyright]", "# [MANAGED] pyright"),
    )

    def apply(self, rendered: str) -> tuple[str, list[str]]:
        changes: list[str] = []
        lines = rendered.splitlines()
        existing_text = rendered
        out: list[str] = []

        has_banner = bool(
            lines and "[MANAGED] FLEXT pyproject standardization" in lines[0]
        )
        if not has_banner:
            out.extend(self._BANNER.splitlines())
            changes.append("managed banner injected")

        marker_map = dict(self._MARKERS)
        for line in lines:
            marker = marker_map.get(line.strip())
            if marker:
                recent = out[-3:] if len(out) >= 3 else out
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append(f"marker injected for {line.strip()}")

            if line.strip().startswith("optional-dependencies.dev"):
                recent = out[-4:] if len(out) >= 4 else out
                marker = "# [MANAGED] consolidated development dependencies"
                auto = "# [AUTO] merged from dev/docs/security/test/typings"
                if marker not in recent and marker not in existing_text:
                    out.append(marker)
                    changes.append("marker injected for optional-dependencies.dev")
                if auto not in recent and auto not in existing_text:
                    out.append(auto)
                    changes.append("auto marker injected for optional-dependencies.dev")

            out.append(line)

        return "\n".join(out).rstrip() + "\n", changes


class PyprojectModernizer:
    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.root = root or ROOT

    def find_pyproject_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.root.rglob("pyproject.toml"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            files.append(path)
        return sorted(files)

    def process_file(
        self,
        path: Path,
        *,
        canonical_dev: list[str],
        dry_run: bool,
        skip_comments: bool,
    ) -> list[str]:
        doc = _read_doc(path)
        if doc is None:
            return ["invalid TOML"]

        changes = ConsolidateGroupsPhase().apply(doc, canonical_dev)
        rendered = tomlkit.dumps(doc)
        if not skip_comments:
            rendered, comment_changes = InjectCommentsPhase().apply(rendered)
            changes.extend(comment_changes)

        if changes and not dry_run:
            _ = path.write_text(rendered, encoding="utf-8")

        return changes

    def run(self, args: argparse.Namespace) -> int:
        dry_run = bool(args.dry_run or args.audit)
        files = self.find_pyproject_files()

        root_doc = _read_doc(self.root / "pyproject.toml")
        if root_doc is None:
            print("invalid root pyproject.toml")
            return 2
        canonical_dev = _canonical_dev_dependencies(root_doc)

        violations: dict[str, list[str]] = {}
        total = 0
        for file_path in files:
            changes = self.process_file(
                file_path,
                canonical_dev=canonical_dev,
                dry_run=dry_run,
                skip_comments=bool(args.skip_comments),
            )
            if not changes:
                continue
            rel = str(file_path.relative_to(self.root))
            violations[rel] = changes
            total += len(changes)

        mode = "AUDIT" if args.audit else ("DRY-RUN" if dry_run else "APPLY")
        print(f"[{mode}] modernize pyproject files: {len(files)}")
        if violations:
            for rel, changes in violations.items():
                print(f"\n{rel}:")
                for item in changes:
                    print(f"  - {item}")
        print(f"\nTotal changes: {total}")

        if args.audit and total > 0:
            return 1

        if not dry_run and not args.skip_fmt:
            self._run_pyproject_fmt(files)
        if not dry_run and not args.skip_check:
            return self._run_poetry_check(files)
        return 0

    def _run_pyproject_fmt(self, files: list[Path]) -> None:
        fmt_bin = self.root / ".venv" / "bin" / "pyproject-fmt"
        if not fmt_bin.exists():
            return
        _ = subprocess.run(
            [str(fmt_bin), *[str(path) for path in files]],
            check=False,
            capture_output=True,
            text=True,
        )

    def _run_poetry_check(self, files: list[Path]) -> int:
        has_warning = False
        for path in files:
            project_dir = path.parent
            result = subprocess.run(
                ["poetry", "check"],
                cwd=project_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                has_warning = True
        return 1 if has_warning else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modernize workspace pyproject files")
    _ = parser.add_argument("--audit", action="store_true")
    _ = parser.add_argument("--dry-run", action="store_true")
    _ = parser.add_argument("--skip-comments", action="store_true")
    _ = parser.add_argument("--skip-fmt", action="store_true")
    _ = parser.add_argument("--skip-check", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    return PyprojectModernizer().run(args)


if __name__ == "__main__":
    raise SystemExit(main())
