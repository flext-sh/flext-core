#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-dependencies/SKILL.md
"""Runtime and dev dependency detection helpers using deptry, pip check, and related tools.

Used by detect_runtime_dev_deps.py and by setup/upgrade automation.
Does not modify lockfiles or pyproject.toml unless explicitly requested by callers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from scripts.libs.selection import resolve_projects  # noqa: E402

# Mypy output patterns for typing library detection (aligned with stub_supply_chain)
MYPY_HINT_RE = re.compile(r'note: Hint: "python3 -m pip install ([^"]+)"')
MYPY_STUB_RE = re.compile(r'Library stubs not installed for "([^"]+)"')
# Internal FLEXT modules: do not suggest types-* for these (fix in code)
INTERNAL_PREFIXES = ("flext_", "flext_", "flext_")

# Default module name -> types-* PyPI package (overridable via dependency_limits.toml)
DEFAULT_MODULE_TO_TYPES_PACKAGE: dict[str, str] = {
    "yaml": "types-pyyaml",
    "ldap3": "types-ldap3",
    "redis": "types-redis",
    "requests": "types-requests",
    "setuptools": "types-setuptools",
    "toml": "types-toml",
    "dateutil": "types-python-dateutil",
    "psutil": "types-psutil",
    "psycopg2": "types-psycopg2",
    "protobuf": "types-protobuf",
    "pyyaml": "types-pyyaml",
    "decorator": "types-decorator",
    "jsonschema": "types-jsonschema",
    "openpyxl": "types-openpyxl",
    "xlrd": "types-xlrd",
}

# Skip directories when discovering projects (same as sync_dependencies / modernize)
SKIP_DIRS = frozenset({
    "archive",
    "backup",
    "node_modules",
    ".git",
    ".venv",
    "cmd",
    "scripts",
    "docs",
    "typings",
    ".claude.disabled",
})


def discover_projects(
    workspace_root: Path,
    projects_filter: list[str] | None = None,
) -> list[Path]:
    """Discover workspace projects eligible for dependency checks."""
    projects = [
        project.path
        for project in resolve_projects(workspace_root, names=[])
        if (project.path / "pyproject.toml").exists()
        and not any(skip in project.name for skip in SKIP_DIRS)
    ]
    if projects_filter is not None:
        filter_set = set(projects_filter)
        projects = [path for path in projects if path.name in filter_set]
    return sorted(projects)


def run_deptry(
    project_path: Path,
    venv_bin: Path,
    *,
    config_path: Path | None = None,
    json_output_path: Path | None = None,
    extend_exclude: list[str] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Run deptry on a project. Returns (parsed JSON issues, exit_code).

    If json_output_path is set, deptry writes JSON there and we read it.
    Otherwise we run with a temp file. Exit code 0 = no issues, 1 = issues found.
    """
    config = config_path or (project_path / "pyproject.toml")
    if not config.exists():
        return [], 0

    out_file = json_output_path or (project_path / ".deptry-report.json")
    cmd: list[str] = [
        str(venv_bin / "deptry"),
        ".",
        "--config",
        str(config),
        "--json-output",
        str(out_file),
        "--no-ansi",
    ]
    if extend_exclude:
        for exc in extend_exclude:
            cmd.extend(["--extend-exclude", exc])

    result = subprocess.run(
        cmd,
        cwd=project_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    issues: list[dict[str, object]] = []
    if out_file.exists():
        try:
            raw = out_file.read_text(encoding="utf-8")
            issues = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            pass
        if json_output_path is None:
            with contextlib.suppress(OSError):
                out_file.unlink()

    return issues, result.returncode


def run_pip_check(workspace_root: Path, venv_bin: Path) -> tuple[list[str], int]:
    """Run 'pip check' for the workspace venv (shared env). Returns (lines of output, exit_code)."""
    pip = venv_bin / "pip"
    if not pip.exists():
        return [], 0
    result = subprocess.run(
        [str(pip), "check"],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "VIRTUAL_ENV": str(venv_bin.parent)},
        check=False,
    )
    out = (result.stdout or "").strip().splitlines() if result.stdout else []
    return out, result.returncode


def classify_issues(
    issues: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group deptry issues by error code (DEP001=missing, DEP002=unused, DEP003=transitive, DEP004=dev in runtime)."""
    groups: dict[str, list[dict[str, object]]] = {
        "DEP001": [],
        "DEP002": [],
        "DEP003": [],
        "DEP004": [],
    }
    for item in issues:
        error_obj = item.get("error")
        code = error_obj.get("code") if isinstance(error_obj, dict) else None
        if isinstance(code, str) and code in groups:
            groups[code].append(item)
    return groups


def build_project_report(
    project_name: str,
    deptry_issues: list[dict[str, object]],
) -> dict[str, object]:
    """Build a single-project report for runtime/dev dependency detection."""
    classified = classify_issues(deptry_issues)
    return {
        "project": project_name,
        "deptry": {
            "missing": [i.get("module") for i in classified["DEP001"]],
            "unused": [i.get("module") for i in classified["DEP002"]],
            "transitive": [i.get("module") for i in classified["DEP003"]],
            "dev_in_runtime": [i.get("module") for i in classified["DEP004"]],
            "raw_count": len(deptry_issues),
        },
    }


def load_dependency_limits(limits_path: Path | None = None) -> dict[str, object]:
    """Load dependency_limits.toml. Returns empty dict if file missing or invalid."""
    path = limits_path or (Path(__file__).resolve().parent / "dependency_limits.toml")
    if not path.exists():
        return {}
    try:
        return dict(tomllib.loads(path.read_text(encoding="utf-8")))
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def run_mypy_stub_hints(
    project_path: Path,
    venv_bin: Path,
    *,
    timeout: int = 300,
) -> tuple[list[str], list[str]]:
    """Run mypy on project src and parse stub hints. Returns (hinted_packages, missing_module_names).

    hinted_packages: pip-installable names from mypy hint (e.g. types-requests).
    missing_module_names: module names from 'Library stubs not installed for X'.
    """
    mypy_bin = venv_bin / "mypy"
    if not mypy_bin.exists():
        return [], []

    cmd: list[str] = [
        str(mypy_bin),
        "src",
        "--config-file",
        "pyproject.toml",
        "--no-error-summary",
    ]
    env = {
        **os.environ,
        "VIRTUAL_ENV": str(venv_bin.parent),
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
    }
    result = subprocess.run(
        cmd,
        cwd=project_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    hinted: set[str] = set()
    for m in MYPY_HINT_RE.finditer(output):
        pkg = m.group(1).strip()
        if pkg:
            hinted.add(pkg)

    missing: set[str] = set()
    for m in MYPY_STUB_RE.finditer(output):
        lib = m.group(1).strip()
        if lib:
            missing.add(lib)

    return sorted(hinted), sorted(missing)


def module_to_types_package(module_name: str, limits: dict[str, object]) -> str | None:
    """Map importable module name to types-* PyPI package. Returns None if internal or unknown."""
    root = module_name.split(".", 1)[0]
    if root.startswith(INTERNAL_PREFIXES):
        return None
    typing_libraries = limits.get("typing_libraries")
    overrides: object = None
    if isinstance(typing_libraries, dict):
        overrides = typing_libraries.get("module_to_package")
    if isinstance(overrides, dict) and root in overrides:
        return str(overrides[root])
    return DEFAULT_MODULE_TO_TYPES_PACKAGE.get(root.lower())


def get_current_typings_from_pyproject(project_path: Path) -> list[str]:
    """Return list of package names in [tool.poetry.group.typings.dependencies] or optional-dependencies.typings."""
    pyproject = project_path / "pyproject.toml"
    if not pyproject.exists():
        return []
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return []

    names: set[str] = set()

    # Poetry group typings
    tool = data.get("tool")
    if isinstance(tool, dict):
        poetry = tool.get("poetry")
        if isinstance(poetry, dict):
            group = poetry.get("group")
            if isinstance(group, dict):
                typings_group = group.get("typings")
                if isinstance(typings_group, dict):
                    deps = typings_group.get("dependencies")
                    if isinstance(deps, dict):
                        names.update(str(k) for k in deps)

    # PEP 621 optional-dependencies typings
    project = data.get("project")
    if isinstance(project, dict):
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            typings = optional.get("typings")
            if isinstance(typings, list):
                for spec in typings:
                    if isinstance(spec, str):
                        names.add(
                            spec.split("[")[0].split(">=")[0].split("==")[0].strip()
                        )
            elif isinstance(typings, dict):
                names.update(str(k) for k in typings)

    return sorted(names)


def get_required_typings(
    project_path: Path,
    venv_bin: Path,
    limits_path: Path | None = None,
    *,
    include_mypy: bool = True,
) -> dict[str, object]:
    """Detect required typing libraries (types-*) for a project using mypy stub hints and limits.

    Returns dict with: required_packages, hinted, missing_modules, current, to_add, to_remove, limits_applied, exclude.
    """
    limits = load_dependency_limits(limits_path)
    exclude_set = set()
    typing_limits = limits.get("typing_libraries")
    if isinstance(typing_limits, dict):
        excluded_obj = typing_limits.get("exclude")
        if isinstance(excluded_obj, list):
            exclude_set = {str(item) for item in excluded_obj}

    hinted: list[str] = []
    missing_modules: list[str] = []
    if include_mypy:
        hinted, missing_modules = run_mypy_stub_hints(project_path, venv_bin)

    required_set: set[str] = set(hinted)
    for mod in missing_modules:
        pkg = module_to_types_package(mod, limits)
        if pkg:
            required_set.add(pkg)
    required_set -= exclude_set
    required_packages = sorted(required_set)

    current = get_current_typings_from_pyproject(project_path)
    current_set = set(current)
    to_add = sorted(required_set - current_set)
    to_remove = sorted(
        current_set - required_set
    )  # optional: suggest removing unused typings

    python_cfg = limits.get("python")
    python_version = python_cfg.get("version") if isinstance(python_cfg, dict) else None

    return {
        "required_packages": required_packages,
        "hinted": hinted,
        "missing_modules": missing_modules,
        "current": current,
        "to_add": to_add,
        "to_remove": to_remove,
        "limits_applied": bool(limits),
        "python_version": python_version,
    }
