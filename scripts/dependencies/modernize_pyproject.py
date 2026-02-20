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

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = ROOT / ".venv" / "bin"

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

SKIP_DIRS = frozenset({
    ".claude.disabled",
    ".flext-deps",
    ".venv",
    "node_modules",
    "__pycache__",
    ".git",
})

COV_FLAG = re.compile(r"^--cov")


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
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
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

    makefile = project_dir / "Makefile"
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


def _replace_inplace(arr: Any, new_items: list[Any]) -> None:
    """Replace contents of a tomlkit array in-place (preserves TOML structure)."""
    while arr:
        arr.pop()
    for item in new_items:
        arr.append(item)


def _ensure_tool(doc: tomlkit.TOMLDocument) -> Any:
    if "tool" not in doc:
        doc.add("tool", tomlkit.table())
    return doc["tool"]


def _get_ssot_bandit_skips() -> list[str]:
    root_doc = tomlkit.parse((ROOT / "pyproject.toml").read_text())
    tool = root_doc.get("tool")
    if not tool:
        msg = "workspace pyproject missing [tool] for Bandit SSOT"
        raise RuntimeError(msg)
    bandit = tool.get("bandit")
    if not bandit:
        msg = "workspace pyproject missing [tool.bandit] for Bandit SSOT"
        raise RuntimeError(msg)
    skips = bandit.get("skips")
    if not skips or not isinstance(skips, list):
        msg = "workspace pyproject missing [tool.bandit].skips list for Bandit SSOT"
        raise RuntimeError(msg)
    unique: list[str] = []
    for item in skips:
        value = str(item).strip()
        if value and value not in unique:
            unique.append(value)
    if not unique:
        msg = "workspace pyproject defines empty [tool.bandit].skips"
        raise RuntimeError(msg)
    return unique


def _get_ssot_pytest_addopts() -> list[str]:
    """Read canonical pytest addopts from workspace pyproject SSOT."""
    root_doc = tomlkit.parse((ROOT / "pyproject.toml").read_text())
    tool = root_doc.get("tool")
    if not tool:
        msg = "workspace pyproject missing [tool] for pytest SSOT"
        raise RuntimeError(msg)
    pytest_section = tool.get("pytest")
    if not pytest_section:
        msg = "workspace pyproject missing [tool.pytest] for pytest SSOT"
        raise RuntimeError(msg)
    ini_options = pytest_section.get("ini_options")
    if not ini_options:
        msg = "workspace pyproject missing [tool.pytest.ini_options] for pytest SSOT"
        raise RuntimeError(msg)
    addopts = ini_options.get("addopts")
    if not addopts or not isinstance(addopts, list):
        msg = "workspace pyproject missing [tool.pytest.ini_options].addopts list for pytest SSOT"
        raise RuntimeError(msg)
    normalized: list[str] = [str(item) for item in addopts if str(item).strip()]
    if not normalized:
        msg = "workspace pyproject defines empty pytest addopts SSOT"
        raise RuntimeError(msg)
    return normalized


# ── Fix functions ────────────────────────────────────────────────────
# Each returns a description string if a fix was applied, None otherwise.


def fix_license_format(doc: tomlkit.TOMLDocument) -> str | None:
    project = doc.get("project")
    if not project:
        return None
    val = project.get("license")
    if val is None or isinstance(val, str):
        return None
    if isinstance(val, dict):
        project["license"] = str(val.get("text", "MIT"))
        return "license: table → SPDX string"
    return None


def fix_license_classifiers(doc: tomlkit.TOMLDocument) -> str | None:
    project = doc.get("project")
    if not project:
        return None
    cls = project.get("classifiers")
    if not cls:
        return None
    filtered = [c for c in cls if c not in LICENSE_CLASSIFIERS]
    if len(filtered) == len(cls):
        return None
    _replace_inplace(cls, filtered)
    return "classifiers: removed deprecated License classifier"


def fix_test_deps_in_runtime(doc: tomlkit.TOMLDocument) -> tuple[str | None, list[str]]:
    project = doc.get("project")
    if not project:
        return None, []
    deps = project.get("dependencies")
    if not deps:
        return None, []
    removed, kept = [], []
    for dep in deps:
        (_removed_list := removed if _norm(str(dep)) in TEST_PACKAGES else kept).append(
            dep
        )
    if not removed:
        return None, []
    names = sorted(_norm(str(d)) for d in removed)
    _replace_inplace(deps, kept)
    return f"dependencies: removed {len(removed)} test deps: {', '.join(names)}", names


def fix_duplicate_poetry_metadata(doc: tomlkit.TOMLDocument) -> str | None:
    project = doc.get("project")
    tool = doc.get("tool")
    if not project or not tool:
        return None
    poetry: dict[str, Any] | None = tool.get("poetry")  # type: ignore[assignment]
    if not poetry:
        return None
    changed = False
    for f in ("name", "version", "description", "authors", "readme", "license"):
        if f in poetry and f in project:
            del poetry[f]
            changed = True
    return "tool.poetry: removed fields duplicated in [project]" if changed else None


def fix_license_in_maintainers(doc: tomlkit.TOMLDocument) -> str | None:
    project = doc.get("project")
    if not project:
        return None
    changed = False
    for field_name in ("maintainers", "authors"):
        people = project.get(field_name)
        if not people:
            continue
        for person in people:
            if isinstance(person, dict) and "license" in person:
                license_val = person.pop("license")
                if "license" not in project:
                    project["license"] = str(license_val)
                changed = True
    return "maintainers: extracted misplaced license key" if changed else None


def fix_deptry_ignores(
    doc: tomlkit.TOMLDocument, removed_pkgs: list[str]
) -> str | None:
    if not removed_pkgs:
        return None
    tool = doc.get("tool")
    if not tool:
        return None
    deptry = tool.get("deptry")  # type: ignore[union-attr]
    if not deptry:
        return None
    per_rule = deptry.get("per_rule_ignores")
    if not per_rule:
        return None
    dep002 = per_rule.get("DEP002")
    if not dep002:
        return None
    removed_set = set(removed_pkgs)
    filtered = [p for p in dep002 if _norm(str(p)) not in removed_set]
    if len(filtered) == len(dep002):
        return None
    _replace_inplace(dep002, filtered)
    return "deptry: cleaned DEP002 ignores for moved deps"


def fix_poetry_core_version(doc: tomlkit.TOMLDocument) -> str | None:
    bs = doc.get("build-system")
    if not bs:
        return None
    requires = bs.get("requires")
    if not requires:
        return None
    for i, req in enumerate(requires):
        s = str(req)
        if _norm(s) == "poetry-core" and ">=2" not in s:
            requires[i] = "poetry-core>=2"
            return "build-system: poetry-core>=2"
    return None


def fix_ruff_extend(doc: tomlkit.TOMLDocument, spec: ProjectSpec) -> str | None:
    tool = doc.get("tool")
    if not tool:
        return None
    ruff = tool.get("ruff")
    if not ruff or "extend" in ruff:
        return None

    first_party = None
    lint = ruff.get("lint")
    if lint:
        isort = lint.get("isort")
        if isort:
            first_party = isort.get("known-first-party")

    for key in list(ruff.keys()):
        del ruff[key]
    ruff["extend"] = spec.ruff_extend

    if first_party:
        ruff.add("lint", tomlkit.table())
        ruff["lint"].add("isort", tomlkit.table())
        ruff["lint"]["isort"]["known-first-party"] = first_party

    return "ruff: replaced inline config with extend"


def fix_coverage_config(doc: tomlkit.TOMLDocument, spec: ProjectSpec) -> str | None:
    """Ensure [tool.coverage] has correct run.source and report.fail_under from ProjectSpec."""
    if not spec.coverage_source:
        return None

    tool = _ensure_tool(doc)
    changes: list[str] = []

    if "coverage" not in tool:
        cov = tomlkit.table()
        run_table = tomlkit.table()
        run_table["source"] = [spec.coverage_source]
        cov.add("run", run_table)
        report_table = tomlkit.table()
        report_table["fail_under"] = spec.min_coverage
        report_table["precision"] = 2
        cov.add("report", report_table)
        tool.add("coverage", cov)  # type: ignore[union-attr]
        return f"coverage: added run.source=[{spec.coverage_source}] fail_under={spec.min_coverage}"

    cov = tool["coverage"]  # type: ignore[index]

    # Ensure run.source exists
    run_sec = cov.get("run") if hasattr(cov, "get") else None  # type: ignore[union-attr]
    if not run_sec:
        cov.add("run", tomlkit.table())  # type: ignore[union-attr]
        cov["run"]["source"] = [spec.coverage_source]  # type: ignore[index]
        changes.append("run.source")
    elif "source" not in run_sec:
        run_sec["source"] = [spec.coverage_source]
        changes.append("run.source")

    # Sync fail_under with Makefile-derived spec
    report_sec = cov.get("report") if hasattr(cov, "get") else None  # type: ignore[union-attr]
    if not report_sec:
        cov.add("report", tomlkit.table())  # type: ignore[union-attr]
        cov["report"]["fail_under"] = spec.min_coverage  # type: ignore[index]
        cov["report"]["precision"] = 2  # type: ignore[index]
        changes.append(f"fail_under={spec.min_coverage}")
    else:
        current = report_sec.get("fail_under")
        if current is not None and int(current) != spec.min_coverage:
            report_sec["fail_under"] = spec.min_coverage
            changes.append(f"fail_under: {current}→{spec.min_coverage}")
        elif current is None:
            report_sec["fail_under"] = spec.min_coverage
            changes.append(f"fail_under={spec.min_coverage}")

    return f"coverage: synced {', '.join(changes)}" if changes else None


def fix_pytest_section(doc: tomlkit.TOMLDocument) -> str | None:
    """Add [tool.pytest] with standard addopts if missing entirely."""
    standard_addopts = _get_ssot_pytest_addopts()
    tool = doc.get("tool")
    if tool and "pytest" in tool:
        return None
    tool = _ensure_tool(doc)
    pt = tomlkit.table()
    ini = tomlkit.table()
    ini["addopts"] = list(standard_addopts)
    pt.add("ini_options", ini)
    tool.add("pytest", pt)  # type: ignore[union-attr]
    return "pytest: added standard ini_options"


def fix_pytest_addopts_sync(doc: tomlkit.TOMLDocument, spec: ProjectSpec) -> str | None:
    """Sync pytest addopts to workspace SSOT for all non-root projects."""
    if spec.is_root:
        return None

    standard_addopts = _get_ssot_pytest_addopts()
    tool = doc.get("tool")
    if not tool:
        return None

    pytest_cfg = tool.get("pytest")
    if not pytest_cfg:
        return None

    ini = pytest_cfg.get("ini_options") if hasattr(pytest_cfg, "get") else None
    if ini is None:
        ini = pytest_cfg

    addopts = ini.get("addopts") if hasattr(ini, "get") else None
    if addopts is None:
        ini["addopts"] = list(standard_addopts)
        return "pytest: set addopts from workspace SSOT"

    if not isinstance(addopts, list):
        ini["addopts"] = list(standard_addopts)
        return "pytest: normalized non-list addopts from workspace SSOT"

    current = [str(item) for item in addopts]
    if current == standard_addopts:
        return None

    _replace_inplace(addopts, list(standard_addopts))
    return "pytest: synced addopts from workspace SSOT"


def fix_pytest_cov_flags(doc: tomlkit.TOMLDocument) -> str | None:
    """Remove --cov* flags from pytest addopts (coverage is owned by [tool.coverage])."""
    tool = doc.get("tool")
    if not tool:
        return None
    pytest_cfg = tool.get("pytest")
    if not pytest_cfg:
        return None

    ini = pytest_cfg.get("ini_options") if hasattr(pytest_cfg, "get") else None
    if ini is None:
        ini = pytest_cfg  # handle [tool.pytest.ini_options] parsed as subtable

    addopts = ini.get("addopts") if hasattr(ini, "get") else None
    if not addopts or not isinstance(addopts, list):
        return None

    filtered = [opt for opt in addopts if not COV_FLAG.match(str(opt))]
    if len(filtered) == len(addopts):
        return None
    removed_count = len(addopts) - len(filtered)
    _replace_inplace(addopts, filtered)
    return f"pytest: removed {removed_count} --cov flags from addopts"


def fix_pyrefly_search_path(doc: tomlkit.TOMLDocument) -> str | None:
    tool = doc.get("tool")
    if not tool:
        return None
    pyrefly = tool.get("pyrefly")
    if not pyrefly:
        return None
    sp = pyrefly.get("search-path")
    if not sp:
        return None
    strs = [str(p) for p in sp]
    changed = False
    if "../typings" not in strs:
        sp.insert(0, "../typings")
        changed = True
    strs = [str(p) for p in sp]
    if "../typings/generated" not in strs:
        sp.insert(strs.index("../typings") + 1, "../typings/generated")
        changed = True
    return "pyrefly: added ../typings to search-path" if changed else None


def fix_bandit_skips(doc: tomlkit.TOMLDocument, spec: ProjectSpec) -> str | None:
    if spec.is_root:
        return None
    standard_skips = _get_ssot_bandit_skips()
    tool = _ensure_tool(doc)
    bandit = tool.get("bandit")
    if bandit is None:
        bandit = tomlkit.table()
        bandit["skips"] = list(standard_skips)
        tool.add("bandit", bandit)  # type: ignore[union-attr]
        return "bandit: added [tool.bandit] from workspace SSOT"
    skips = bandit.get("skips")
    if not skips or not isinstance(skips, list):
        bandit["skips"] = list(standard_skips)
        return "bandit: set skips from workspace SSOT"
    existing = {str(s).strip() for s in skips}
    missing = [s for s in standard_skips if s not in existing]
    if not missing:
        return None
    for s in missing:
        skips.append(s)
    return f"bandit: added SSOT skips {missing}"


# ── Pipeline ─────────────────────────────────────────────────────────


def _has_array_of_tables(path: Path) -> bool:
    """Detect [[array.of.tables]] that tomlkit corrupts on write (indented sub-configs)."""
    return "[[tool.pyrefly.sub-config]]" in path.read_text(encoding="utf-8")


def process_file(path: Path, spec: ProjectSpec, *, dry_run: bool = False) -> list[str]:
    """Run all fix functions on a single pyproject.toml. Returns fix descriptions."""
    text = path.read_text(encoding="utf-8")
    unsafe_write = _has_array_of_tables(path)

    doc = tomlkit.parse(text)
    fixes: list[str] = []

    def _apply(result: str | None) -> None:
        if result:
            fixes.append(result)

    _apply(fix_license_format(doc))
    _apply(fix_license_classifiers(doc))
    test_msg, removed_pkgs = fix_test_deps_in_runtime(doc)
    _apply(test_msg)
    _apply(fix_deptry_ignores(doc, removed_pkgs))
    _apply(fix_duplicate_poetry_metadata(doc))
    _apply(fix_license_in_maintainers(doc))
    _apply(fix_poetry_core_version(doc))
    _apply(fix_ruff_extend(doc, spec))
    _apply(fix_coverage_config(doc, spec))
    _apply(fix_pytest_section(doc))
    _apply(fix_pytest_addopts_sync(doc, spec))
    _apply(fix_pytest_cov_flags(doc))
    _apply(fix_pyrefly_search_path(doc))
    _apply(fix_bandit_skips(doc, spec))

    if fixes and not dry_run:
        if unsafe_write:
            _apply_regex_fixes(path, spec, fixes)
        else:
            path.write_text(tomlkit.dumps(doc), encoding="utf-8")

    return fixes


def _apply_regex_fixes(path: Path, spec: ProjectSpec, fixes: list[str]) -> None:
    """Apply fixes via regex for files where tomlkit corrupts [[array.of.tables]].

    Comprehensive handler for all fix types since tomlkit cannot safely write these files.
    """
    text = path.read_text(encoding="utf-8")
    original = text

    # 1. poetry-core version bump
    text = re.sub(r"poetry-core>=1\.\d+", "poetry-core>=2", text)

    # 2. License: [project.license] table → PEP 639 SPDX string
    if any("license" in f and "SPDX" in f for f in fixes):
        license_match = re.search(
            r"^\s*\[project\.license\]\s*\n\s*text\s*=\s*\"([^\"]+)\"\s*\n",
            text,
            flags=re.MULTILINE,
        )
        if license_match:
            license_val = license_match.group(1)
            text = text[: license_match.start()] + text[license_match.end() :]
            text = re.sub(
                r"(^\s*\[\[project\.(authors|maintainers)\]\])",
                f'    license = "{license_val}"\n\n\\1',
                text,
                count=1,
                flags=re.MULTILINE,
            )

    # 3. License classifier removal
    if any("License classifier" in f for f in fixes):
        text = re.sub(
            r'^\s*"License :: [^"]*",?\s*\n',
            "",
            text,
            flags=re.MULTILINE,
        )

    # 4. Test deps in runtime dependencies
    if any("test deps" in f for f in fixes):
        test_pkg_pattern = "|".join(re.escape(p) for p in sorted(TEST_PACKAGES))
        text = re.sub(
            rf'^\s*"({test_pkg_pattern})\s*[\(@>=<~!].*",?\s*\n',
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # 4b. Deptry DEP002 cleanup for removed test deps
    if any("DEP002" in f for f in fixes):
        test_pkg_pattern = "|".join(re.escape(p) for p in sorted(TEST_PACKAGES))
        text = re.sub(
            rf'^\s*"({test_pkg_pattern})",?\s*\n',
            "",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    # 4c. Misplaced license key inside [[project.maintainers]] or [[project.authors]]
    if any("misplaced license" in f for f in fixes):
        text = re.sub(
            r"(^\s*\[\[project\.(maintainers|authors)\]\]\s*\n(?:\s+\w+\s*=\s*[^\n]+\n)*)\s+license\s*=\s*[^\n]+\n",
            r"\1",
            text,
            flags=re.MULTILINE,
        )

    # 5. Duplicate poetry metadata removal
    if any("duplicated in [project]" in f for f in fixes):
        # In [tool.poetry] section, remove name/version/description/authors/readme/license
        # that duplicate [project] fields. This is complex with regex; only remove
        # simple key = value lines within [tool.poetry] block.
        for field in ("name", "version", "description", "authors", "readme", "license"):
            text = re.sub(
                rf"^(\[tool\.poetry\]\s*\n(?:.*\n)*?)\s+{field}\s*=\s*[^\n]+\n",
                r"\1",
                text,
                count=1,
                flags=re.MULTILINE,
            )

    # 6. pyrefly search-path: add ../typings entries
    if any("pyrefly" in f for f in fixes) and '"../typings"' not in text:
        text = re.sub(
            r"(search-path\s*=\s*\[)\s*\n",
            r'\1\n            "../typings",\n            "../typings/generated",\n',
            text,
            count=1,
        )

    # 7. Coverage: sync existing fail_under value
    if any("fail_under" in f and "→" in f for f in fixes):
        text = re.sub(
            r"(fail_under\s*=\s*)\d+",
            rf"\g<1>{spec.min_coverage}",
            text,
            count=1,
        )

    # 8. Coverage: add fail_under to existing [tool.coverage.report] section
    if any("fail_under=" in f for f in fixes) and "fail_under" not in text:
        if "[tool.coverage.report]" in text:
            # Add fail_under after the section header
            text = re.sub(
                r"(\[tool\.coverage\.report\]\s*\n)",
                rf"\1        fail_under = {spec.min_coverage}\n",
                text,
                count=1,
            )
        elif "[tool.coverage" in text:
            # Has [tool.coverage.run] but no report section — append report section
            text = re.sub(
                r"(\[tool\.coverage\.run\](?:.*\n)*?)((?=\s*\[)|\Z)",
                rf"\1\n    [tool.coverage.report]\n        fail_under = {spec.min_coverage}\n        precision = 2\n\n\2",
                text,
                count=1,
            )
        else:
            # No coverage section at all — append before last section or at end
            coverage_block = (
                f"\n[tool.coverage]\n"
                f"    [tool.coverage.run]\n"
                f'        source = ["{spec.coverage_source}"]\n'
                f"\n"
                f"    [tool.coverage.report]\n"
                f"        fail_under = {spec.min_coverage}\n"
                f"        precision = 2\n"
            )
            # Insert before [dependency-groups] if it exists, otherwise append
            if "[dependency-groups]" in text:
                text = text.replace(
                    "[dependency-groups]",
                    f"{coverage_block}\n[dependency-groups]",
                )
            else:
                text = text.rstrip() + "\n" + coverage_block + "\n"

    # 9. --cov flag removal from pytest addopts
    if any("--cov flags" in f for f in fixes):
        lines = text.split("\n")
        filtered = [
            l
            for l in lines
            if not COV_FLAG.search(l.strip().strip('"').strip("'").strip(","))
        ]
        text = "\n".join(filtered)

    if any(
        "pytest: synced addopts" in f
        or "pytest: set addopts" in f
        or "pytest: normalized non-list addopts" in f
        for f in fixes
    ):
        ssot_addopts = _get_ssot_pytest_addopts()
        addopts_literal = "[ " + ", ".join(f'"{opt}"' for opt in ssot_addopts) + " ]"
        text, replaced = re.subn(
            r"(^\s*ini_options\.addopts\s*=\s*)(\[[\s\S]*?\]|\"[^\"]*\")",
            rf"\1{addopts_literal}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if replaced == 0:

            def _replace_pytest_ini_options_section(match: re.Match[str]) -> str:
                section_header = match.group(1)
                section_body = match.group(2)
                updated_body, _ = re.subn(
                    r"(^\s*addopts\s*=\s*)(\[[\s\S]*?\]|\"[^\"]*\")",
                    rf"\1{addopts_literal}",
                    section_body,
                    count=1,
                    flags=re.MULTILINE,
                )
                return f"{section_header}{updated_body}"

            text = re.sub(
                r"(?ms)(^\s*\[tool\.pytest\.ini_options\]\s*\n)(.*?)(?=^\s*\[|\Z)",
                _replace_pytest_ini_options_section,
                text,
                count=1,
            )

    if any(f.startswith("bandit:") for f in fixes):
        ssot_skips = _get_ssot_bandit_skips()
        section_match = re.search(
            r"(?ms)^\s*\[tool\.bandit\]\s*\n.*?(?=^\s*\[|\Z)",
            text,
        )
        if not section_match:
            block = "[tool.bandit]\n    skips = [\n"
            for item in ssot_skips:
                block += f'        "{item}",\n'
            block += "    ]\n"
            if "[dependency-groups]" in text:
                text = text.replace(
                    "[dependency-groups]", f"{block}\n[dependency-groups]"
                )
            else:
                text = text.rstrip() + "\n\n" + block + "\n"
        else:
            section = section_match.group(0)
            skips_match = re.search(
                r"(?ms)^\s*skips\s*=\s*\[(?P<body>.*?)^\s*\]",
                section,
            )
            if not skips_match:
                insert = "    skips = [\n"
                for item in ssot_skips:
                    insert += f'        "{item}",\n'
                insert += "    ]\n"
                section = re.sub(
                    r"(^\s*\[tool\.bandit\]\s*\n)",
                    rf"\1{insert}",
                    section,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                existing = set(re.findall(r'"([^"]+)"', skips_match.group("body")))
                missing = [item for item in ssot_skips if item not in existing]
                if missing:
                    append_lines = "".join(f'        "{item}",\n' for item in missing)
                    body = skips_match.group("body") + append_lines
                    section = (
                        section[: skips_match.start("body")]
                        + body
                        + section[skips_match.end("body") :]
                    )
            text = text[: section_match.start()] + section + text[section_match.end() :]

    if text != original:
        path.write_text(text, encoding="utf-8")


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
                makefile.write_text(text)

    return fixes


# ── Discovery ────────────────────────────────────────────────────────


def find_pyproject_files() -> list[Path]:
    results = []
    for p in sorted(ROOT.rglob("pyproject.toml")):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        results.append(p)
    return results


# ── Formatting + validation ──────────────────────────────────────────


def run_pyproject_fmt(paths: list[Path], *, dry_run: bool = False) -> int:
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
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 and dry_run:
        print(f"  ℹ {len(safe)} files would be reformatted")
    elif result.returncode == 0:
        print(f"  ✓ {len(safe)} files formatted")
    return result.returncode


def run_poetry_check(paths: list[Path]) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for path in paths:
        project_dir = path.parent
        label = "root" if project_dir == ROOT else project_dir.name
        result = subprocess.run(
            ["poetry", "check"], capture_output=True, text=True, cwd=project_dir
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
        run_pyproject_fmt(files, dry_run=dry_run)

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
