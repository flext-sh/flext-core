#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-dependencies/SKILL.md
"""Declarative pyproject.toml standardization for the FLEXT monorepo.

Enforces Poetry 2.x + PEP 621 compliance and cross-project tool config consistency
across all workspace projects via a ProjectSpec auto-derived from each project's
directory layout and Makefile.

Usage:
    .venv/bin/python scripts/dependencies/modernize_pyproject.py [OPTIONS]

Options:
    --audit       Report violations without fixing (exit 1 if any found)
    --dry-run     Show what would change without writing files
    --skip-fmt    Skip pyproject-fmt formatting pass
    --skip-check  Skip poetry check validation pass
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import tomlkit
from tomlkit.items import Table

from scripts.libs.config import (
    DEFAULT_ENCODING,
    MAKEFILE_FILENAME,
    PYPROJECT_FILENAME,
    PYPROJECT_SKIP_DIRS as SKIP_DIRS,
    VENV_BIN_REL,
)
from scripts.libs.discovery import find_all_pyproject_files
from scripts.libs.paths import workspace_root_from_file
from scripts.libs.toml_io import (
    build_table,
    read_toml_document,
    sync_section,
    write_toml_document,
)

_TableLike = Table | MutableMapping[str, object]

ROOT = workspace_root_from_file(__file__)
VENV_BIN = ROOT / VENV_BIN_REL

TEST_PACKAGES = frozenset({
    "pytest",
    "pytest-benchmark",
    "pytest-clarity",
    "pytest-cov",
    "pytest-deadfixtures",
    "pytest-env",
    "pytest-httpx",
    "pytest-mock",
    "pytest-randomly",
    "pytest-sugar",
    "pytest-timeout",
    "pytest-xdist",
    "pytest-asyncio",
    "pytest-json-report",
    "factory-boy",
    "faker",
    "hypothesis",
    "coverage",
})

LICENSE_CLASSIFIERS = frozenset({
    "License :: OSI Approved :: MIT License",
    "License :: OSI Approved",
})

# ── ProjectSpec ──────────────────────────────────────────────────────


@dataclass
class ProjectSpec:
    """Auto-derived spec for what a project's pyproject.toml SHOULD contain.

    Values are computed from:
    - src/ directory layout → pkg_name, coverage_source
    - legacy Makefile coverage variable → min_coverage (pyproject.toml is the source of truth,
      but we read existing Makefile values to migrate them correctly)
    - Project path relative to ROOT → ruff_extend path
    """

    project_dir: Path
    pkg_name: str | None = None
    min_coverage: int = 100
    is_root: bool = False

    ruff_extend: str = field(init=False)
    coverage_source: str | None = field(init=False)

    def __post_init__(self) -> None:
        """Derive project metadata from filesystem and config."""
        self.is_root = self.project_dir == ROOT
        self.ruff_extend = (
            "./ruff-shared.toml" if self.is_root else "../ruff-shared.toml"
        )
        if self.pkg_name is None:
            self.pkg_name = _detect_src_package(self.project_dir)
        if self.min_coverage == 100:
            self.min_coverage = _read_min_coverage(self.project_dir)
        self.coverage_source = f"src/{self.pkg_name}" if self.pkg_name else None


def _detect_src_package(project_dir: Path) -> str | None:
    src_dir = project_dir / "src"
    if not src_dir.is_dir():
        return None
    packages = [
        d.name
        for d in src_dir.iterdir()
        if d.is_dir() and not d.name.startswith((".", "__"))
    ]
    return packages[0] if len(packages) == 1 else None


def _read_min_coverage(project_dir: Path) -> int:
    """Read min coverage: pyproject.toml is source of truth, Makefile is migration fallback."""
    pyproject = project_dir / PYPROJECT_FILENAME
    if pyproject.exists():
        doc = read_toml_document(pyproject)
        if doc is None:
            doc = tomlkit.parse(pyproject.read_text())
        tool = doc.get("tool")
        if tool:
            cov = tool.get("coverage")
            if cov:
                report = cov.get("report") if hasattr(cov, "get") else None
                if report:
                    val = report.get("fail_under") if hasattr(report, "get") else None
                    if val is not None:
                        return int(val)

    makefile = project_dir / MAKEFILE_FILENAME
    if makefile.exists():
        match = re.search(r"MIN_COVERAGE\s*:?=\s*(\d+)", makefile.read_text())
        if match:
            return int(match.group(1))

    return 100


# ── Helpers ──────────────────────────────────────────────────────────


def _norm(dep_str: str) -> str:
    """Normalize a PEP 508 dependency string to its bare package name."""
    name = dep_str.strip().split(">=")[0].split("(")[0].split("<")[0]
    name = name.split("==")[0].split("~=")[0].split("!=")[0].split("[")[0].strip()
    return name.lower().replace("_", "-")


def _replace_inplace(arr: list[object], new_items: list[object]) -> None:
    """Replace contents of a tomlkit array in-place (preserves TOML structure)."""
    while arr:
        _ = arr.pop()
    arr.extend(new_items)


def _ensure_tool(doc: tomlkit.TOMLDocument) -> Table:
    if "tool" not in doc:
        _ = doc.add("tool", tomlkit.table())
    tool = doc["tool"]
    if not isinstance(tool, Table):
        msg = "Invalid [tool] table structure"
        raise TypeError(msg)
    return tool


def _require_table(value: object, msg: str = "expected Table") -> Table:
    """Return value as Table after runtime check; raise if not."""
    if not value or not isinstance(value, Table):
        raise RuntimeError(msg)
    return value


@functools.cache
def _root_tool_doc() -> Table:
    """Parse workspace root pyproject.toml ONCE and return the ``[tool]`` table."""
    root_doc = read_toml_document(ROOT / PYPROJECT_FILENAME)
    if root_doc is None:
        msg = "workspace pyproject.toml is missing or invalid"
        raise RuntimeError(msg)
    tool = root_doc.get("tool")
    return _require_table(tool, "workspace pyproject.toml missing valid [tool] section")


def _ssot_section(*keys: str) -> object:
    """Navigate nested keys under the cached root ``[tool]`` table.

    ``_ssot_section("pytest", "ini_options")`` returns ``[tool.pytest.ini_options]``.
    Raises ``RuntimeError`` if any key is missing.
    """
    current: object = _root_tool_doc()
    path_so_far = "tool"
    for key in keys:
        if not hasattr(current, "get"):
            msg = f"workspace SSOT: [tool.{path_so_far}] is not a table"
            raise RuntimeError(msg)
        current = current.get(key)  # type: ignore[union-attr]
        path_so_far = f"{path_so_far}.{key}"
        if current is None:
            msg = f"workspace SSOT: [{path_so_far}] not found"
            raise RuntimeError(msg)
    return current


# ── Declarative SSOT enforcement ─────────────────────────────────────

_SKIP = object()


@dataclass(frozen=True)
class SSOTRule:
    """Rule for SSOT section: root_only keys, per_project overrides, entry_point, guard."""

    from_root: bool = True
    prune_extras: bool = True
    root_only: frozenset[str] = frozenset()
    per_project: dict[str, object] = field(default_factory=dict)
    entry_point: str | None = None
    guard: str | None = None


SSOT_RULES: dict[str, SSOTRule] = {
    "bandit": SSOTRule(),
    "pyright": SSOTRule(root_only=frozenset({"executionEnvironments"})),
    "coverage": SSOTRule(
        from_root=False,
        prune_extras=False,
        guard="coverage_source",
        per_project={
            "run.source": "coverage_source_list",
            "report.fail_under": "min_coverage",
            "report.precision": "coverage_precision",
        },
    ),
    "pytest": SSOTRule(entry_point="ini_options"),
    "pyrefly": SSOTRule(
        prune_extras=False,
        root_only=frozenset({"sub-config"}),
        per_project={"search-path": "pyrefly_sub_path"},
    ),
    "ruff": SSOTRule(
        per_project={
            "extend": "ruff_extend",
            "lint.isort.known-first-party": "project_pkg_list",
        }
    ),
    "codespell": SSOTRule(),
    "complexipy": SSOTRule(),
    "vulture": SSOTRule(),
    "deptry": SSOTRule(prune_extras=False),
}

_PYREFLY_SUB_PATH: list[str] = [
    "../typings",
    "../typings/generated",
    "src",
    "tests",
    "examples",
    "scripts",
]


def _resolve_override(key: str, spec: ProjectSpec) -> object:
    overrides: dict[str, object] = {
        "coverage_source_list": (
            [spec.coverage_source] if spec.coverage_source else _SKIP
        ),
        "min_coverage": spec.min_coverage,
        "coverage_precision": 2,
        "pyrefly_sub_path": _PYREFLY_SUB_PATH,
        "ruff_extend": spec.ruff_extend,
        "project_pkg_list": [spec.pkg_name] if spec.pkg_name else [],
    }
    return overrides.get(key, _SKIP)


def _enforce_one(
    doc: tomlkit.TOMLDocument,
    tool_name: str,
    rule: SSOTRule,
    spec: ProjectSpec,
) -> str | None:
    if spec.is_root:
        return None
    if rule.guard and not getattr(spec, rule.guard, None):
        return None

    canonical: dict[str, object] = {}
    if rule.from_root:
        root_keys = list((rule.entry_point and [rule.entry_point]) or [])
        try:
            root_sec = cast(
                "MutableMapping[str, object]",
                _ssot_section(tool_name, *root_keys),
            )
        except RuntimeError:
            return None
        for k, v in root_sec.items():
            if k in rule.root_only:
                continue
            if hasattr(v, "items") and not isinstance(v, (list, str)):
                canonical[k] = dict(cast("MutableMapping[str, object]", v).items())
            else:
                canonical[k] = v

    for dotted, resolver_key in rule.per_project.items():
        value = _resolve_override(str(resolver_key), spec)
        if value is _SKIP:
            continue
        parts = dotted.split(".")
        target: dict[str, object] = canonical
        for p in parts[:-1]:
            nxt = target.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                target[p] = nxt
            target = nxt
        target[parts[-1]] = value

    tool = _ensure_tool(doc)
    section = tool.get(tool_name)

    if section is None:
        outer = build_table(canonical)
        if rule.entry_point:
            wrapper = tomlkit.table()
            wrapper[rule.entry_point] = outer
            cast("MutableMapping[str, object]", tool)[tool_name] = wrapper
        else:
            cast("MutableMapping[str, object]", tool)[tool_name] = outer
        return f"{tool_name}: added from workspace SSOT"

    if rule.entry_point:
        inner = section.get(rule.entry_point) if hasattr(section, "get") else None
        if inner is None:
            section[rule.entry_point] = build_table(canonical)
            return f"{tool_name}: added {rule.entry_point} from workspace SSOT"
        target_sec = cast("_TableLike", inner)
    else:
        target_sec = cast("_TableLike", section)

    added, updated, removed = sync_section(
        target_sec,
        canonical,
        prune_extras=rule.prune_extras,
    )
    if not added and not updated and not removed:
        return None

    parts_msg: list[str] = []
    if added:
        parts_msg.append(f"added {added}")
    if updated:
        parts_msg.append(f"updated {updated}")
    if removed:
        parts_msg.append(f"removed {removed}")
    return f"{tool_name}: enforced SSOT ({', '.join(parts_msg)})"


def _enforce_ssot_tools(
    doc: tomlkit.TOMLDocument,
    spec: ProjectSpec,
) -> list[str]:
    results: list[str] = []
    for tool_name, rule in SSOT_RULES.items():
        result = _enforce_one(doc, tool_name, rule, spec)
        if result:
            results.append(result)
    return results


# ── Fix functions ────────────────────────────────────────────────────
# Each returns a description string if a fix was applied, None otherwise.


@dataclass(frozen=True)
class NormalizeRule:
    """Describes one declarative normalization rule for pyproject.toml."""

    kind: str
    message: str
    path: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()


NORMALIZE_RULES: tuple[NormalizeRule, ...] = (
    NormalizeRule(
        kind="license_table_to_string",
        message="license: table → SPDX string",
        path=("project",),
    ),
    NormalizeRule(
        kind="remove_license_classifiers",
        message="classifiers: removed deprecated License classifier",
        path=("project",),
    ),
    NormalizeRule(
        kind="remove_poetry_duplicates",
        message="tool.poetry: removed fields duplicated in [project]",
        path=("tool", "poetry"),
        fields=("name", "version", "description", "authors", "readme", "license"),
    ),
    NormalizeRule(
        kind="extract_license_from_people",
        message="maintainers: extracted misplaced license key",
        path=("project",),
    ),
    NormalizeRule(
        kind="normalize_poetry_core",
        message="build-system: poetry-core>=2",
        path=("build-system",),
    ),
)


def _get_path_table(doc: tomlkit.TOMLDocument, path: tuple[str, ...]) -> object:
    current: object = doc
    for key in path:
        if not hasattr(current, "get"):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _apply_normalize_rule(doc: tomlkit.TOMLDocument, rule: NormalizeRule) -> str | None:
    target = _get_path_table(doc, rule.path)
    if target is None:
        return None

    if rule.kind == "license_table_to_string":
        project = cast("MutableMapping[str, object]", target)
        val = project.get("license")
        if val is None or isinstance(val, str):
            return None
        if isinstance(val, dict):
            project["license"] = str(val.get("text", "MIT"))
            return rule.message
        return None

    if rule.kind == "remove_license_classifiers":
        project = cast("MutableMapping[str, object]", target)
        cls_obj = project.get("classifiers")
        if not isinstance(cls_obj, list):
            return None
        cls = cast("list[object]", cls_obj)
        filtered = [c for c in cls if str(c) not in LICENSE_CLASSIFIERS]
        if len(filtered) == len(cls):
            return None
        _replace_inplace(cls, filtered)
        return rule.message

    if rule.kind == "remove_poetry_duplicates":
        project_table = _get_path_table(doc, ("project",))
        if project_table is None:
            return None
        project = cast("MutableMapping[str, object]", project_table)
        poetry = cast("MutableMapping[str, object]", target)
        changed = False
        for field_name in rule.fields:
            if field_name in poetry and field_name in project:
                del poetry[field_name]
                changed = True
        return rule.message if changed else None

    if rule.kind == "extract_license_from_people":
        project = cast("MutableMapping[str, object]", target)
        changed = False
        for field_name in ("maintainers", "authors"):
            people_obj = project.get(field_name)
            if not isinstance(people_obj, list):
                continue
            for person in people_obj:
                if isinstance(person, dict) and "license" in person:
                    license_val = person.pop("license")
                    if "license" not in project:
                        project["license"] = str(license_val)
                    changed = True
        return rule.message if changed else None

    if rule.kind == "normalize_poetry_core":
        build_system = cast("MutableMapping[str, object]", target)
        requires_obj = build_system.get("requires")
        if not isinstance(requires_obj, list):
            return None
        requires = cast("list[object]", requires_obj)
        for i, req in enumerate(requires):
            s = str(req)
            if _norm(s) == "poetry-core" and ">=2" not in s:
                requires[i] = "poetry-core>=2"
                return rule.message
        return None

    return None


def _apply_normalize_rules(doc: tomlkit.TOMLDocument) -> list[str]:
    results: list[str] = []
    for rule in NORMALIZE_RULES:
        result = _apply_normalize_rule(doc, rule)
        if result:
            results.append(result)
    return results


def fix_test_deps_in_runtime(doc: tomlkit.TOMLDocument) -> tuple[str | None, list[str]]:
    """Move test-only packages out of runtime dependencies."""
    project = doc.get("project")
    if not project:
        return None, []
    deps = project.get("dependencies")
    if not deps:
        return None, []
    removed: list[object] = []
    kept: list[object] = []
    for dep in deps:
        (_removed_list := removed if _norm(str(dep)) in TEST_PACKAGES else kept).append(
            dep
        )
    if not removed:
        return None, []
    names = sorted(_norm(str(d)) for d in removed)
    _replace_inplace(deps, kept)
    return f"dependencies: removed {len(removed)} test deps: {', '.join(names)}", names


def fix_deptry_ignores(
    doc: tomlkit.TOMLDocument, removed_pkgs: list[str]
) -> str | None:
    """Clean deptry ignore entries for removed packages."""
    if not removed_pkgs:
        return None
    tool = doc.get("tool")
    if not tool:
        return None
    deptry_obj = tool.get("deptry")
    if not isinstance(deptry_obj, MutableMapping):
        return None
    per_rule_obj = deptry_obj.get("per_rule_ignores")
    if not isinstance(per_rule_obj, MutableMapping):
        return None
    dep002 = per_rule_obj.get("DEP002")
    if not dep002:
        return None
    removed_set = set(removed_pkgs)
    filtered = [p for p in dep002 if _norm(str(p)) not in removed_set]
    if len(filtered) == len(dep002):
        return None
    _replace_inplace(dep002, filtered)
    return "deptry: cleaned DEP002 ignores for moved deps"


# ── Pipeline ─────────────────────────────────────────────────────────


def _has_array_of_tables(path: Path) -> bool:
    """Detect ``[[array.of.tables]]`` that pyproject-fmt corrupts on reformat."""
    return "[[tool.pyrefly.sub-config]]" in path.read_text(encoding=DEFAULT_ENCODING)


def process_file(path: Path, spec: ProjectSpec, *, dry_run: bool = False) -> list[str]:
    """Run all fix functions on a single pyproject.toml. Returns fix descriptions."""
    doc = read_toml_document(path)
    if doc is None:
        msg = f"invalid TOML: {path}"
        raise RuntimeError(msg)
    fixes: list[str] = []

    def _apply(result: str | None) -> None:
        if result:
            fixes.append(result)

    fixes.extend(_apply_normalize_rules(doc))
    test_msg, removed_pkgs = fix_test_deps_in_runtime(doc)
    _apply(test_msg)
    _apply(fix_deptry_ignores(doc, removed_pkgs))
    fixes.extend(_enforce_ssot_tools(doc, spec))

    if fixes and not dry_run:
        write_toml_document(path, doc)

    return fixes


# ── Makefile cleanup ─────────────────────────────────────────────────


def cleanup_makefiles(*, dry_run: bool = False) -> list[str]:
    """Remove legacy coverage and COV_DIR vars from project Makefiles.

    pyproject.toml [tool.coverage] is now the single source of truth for these values.
    base.mk will be updated separately to read from pyproject.toml.
    """
    fixes: list[str] = []
    for makefile in sorted(ROOT.glob("*/Makefile")):
        text = makefile.read_text()
        original = text

        text = re.sub(r"^MIN_COVERAGE\s*:?=\s*\d+\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^COV_DIR\s*:?=\s*\S+\s*\n", "", text, flags=re.MULTILINE)

        if text != original:
            name = makefile.parent.name
            fixes.append(name)
            if not dry_run:
                _ = makefile.write_text(text)

    return fixes


# ── Discovery ────────────────────────────────────────────────────────


def find_pyproject_files() -> list[Path]:
    """Discover pyproject.toml files under ROOT, excluding SKIP_DIRS."""
    return find_all_pyproject_files(ROOT, skip_dirs=SKIP_DIRS)


# ── Formatting + validation ──────────────────────────────────────────


def run_pyproject_fmt(paths: list[Path], *, dry_run: bool = False) -> int:
    """Run pyproject-fmt over safe files and return exit code."""
    fmt_bin = VENV_BIN / "pyproject-fmt"
    if not fmt_bin.exists():
        print(f"  ⚠ pyproject-fmt not found at {fmt_bin}")
        return 1
    safe = [p for p in paths if not _has_array_of_tables(p)]
    skipped = len(paths) - len(safe)
    if skipped:
        print(
            f"  ⚠ Skipping {skipped} files with [[array.of.tables]] (pyproject-fmt corrupts them)"
        )
    if not safe:
        return 0
    args = [str(fmt_bin)]
    if dry_run:
        args.append("--check")
    args.extend(str(p) for p in safe)
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0 and dry_run:
        print(f"  ℹ {len(safe)} files would be reformatted")
    elif result.returncode == 0:
        print(f"  ✓ {len(safe)} files formatted")
    return result.returncode


def run_poetry_check(paths: list[Path]) -> dict[str, list[str]]:
    """Run poetry check for each project and collect warnings."""
    results: dict[str, list[str]] = {}
    for path in paths:
        project_dir = path.parent
        label = "root" if project_dir == ROOT else project_dir.name
        result = subprocess.run(
            ["poetry", "check"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=project_dir,
            check=False,
        )
        warnings = [
            line.strip()
            for line in (result.stdout + result.stderr).splitlines()
            if line.strip().startswith("Warning:")
        ]
        if warnings:
            results[label] = warnings
    return results


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    """Run pyproject modernization phases across workspace files."""
    audit = "--audit" in sys.argv
    dry_run = "--dry-run" in sys.argv or audit
    skip_fmt = "--skip-fmt" in sys.argv
    skip_check = "--skip-check" in sys.argv

    mode = "[AUDIT]" if audit else ("[DRY RUN]" if dry_run else "")
    print(f"{mode} Modernizing pyproject.toml files...".strip())
    print(f"  Root: {ROOT}")

    files = find_pyproject_files()
    print(f"  Found {len(files)} pyproject.toml files\n")

    # Phase 1: ProjectSpec-driven fixes
    print("Phase 1: Declarative fixes (ProjectSpec-driven)")
    print("=" * 60)
    total_fixes = 0
    violations: dict[str, list[str]] = {}

    for path in files:
        spec = ProjectSpec(project_dir=path.parent)
        fixes = process_file(path, spec, dry_run=dry_run)
        if fixes:
            rel = str(path.relative_to(ROOT))
            total_fixes += len(fixes)
            violations[rel] = fixes
            marker = "⚠" if audit else "✓"
            print(f"\n  {rel}:")
            for fix in fixes:
                print(f"    {marker} {fix}")

    if total_fixes == 0:
        print("  ✓ All projects comply with spec.")
    else:
        word = "violations" if audit else "fixes"
        print(f"\n  Total: {total_fixes} {word} across {len(violations)} files")

    # Phase 1b: Makefile cleanup (remove legacy coverage var/COV_DIR)
    print("\n\nPhase 1b: Makefile cleanup")
    print("=" * 60)
    mk_fixes = cleanup_makefiles(dry_run=dry_run)
    if mk_fixes:
        marker = "⚠" if audit else "✓"
        print(
            f"  {marker} Removed legacy coverage var/COV_DIR from: {', '.join(mk_fixes)}"
        )
    else:
        print("  ✓ All Makefiles clean.")

    if audit and (total_fixes > 0 or mk_fixes):
        total = total_fixes + len(mk_fixes)
        print(f"\n✗ Audit failed: {total} violations found.")
        print("  Run without --audit to auto-fix.")
        return 1

    # Phase 2: pyproject-fmt
    if not skip_fmt:
        label = "(check only)" if dry_run else "(apply)"
        print(f"\n\nPhase 2: pyproject-fmt {label}")
        print("=" * 60)
        _ = run_pyproject_fmt(files, dry_run=dry_run)

    # Phase 3: poetry check
    if not skip_check and not dry_run:
        print("\n\nPhase 3: poetry check")
        print("=" * 60)
        warnings = run_poetry_check(files)
        if warnings:
            print(f"\n  ⚠ {len(warnings)} projects have warnings:\n")
            for project, warns in sorted(warnings.items()):
                print(f"  {project}:")
                for w in warns:
                    print(f"    {w}")
            return 1
        print(f"  ✓ All {len(files)} projects pass poetry check")

    print("\n✓ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
