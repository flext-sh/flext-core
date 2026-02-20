#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-infra/SKILL.md

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

MISSING_IMPORT_RE = re.compile(r"Cannot find module `([^`]+)` \[missing-import\]")
MYPY_HINT_RE = re.compile(r'note: Hint: "python3 -m pip install ([^"]+)"')
MYPY_STUB_RE = re.compile(r'Library stubs not installed for "([^"]+)"')
INTERNAL_PREFIXES = ("flext_", "flext_", "flext_")


@dataclass(frozen=True)
class ProjectResult:
    project: str
    internal_missing_imports: list[str]
    unresolved_missing_imports: list[str]
    generated_files: list[str]
    types_packages_added: list[str]
    types_packages_missing: list[str]
    mypy_stub_hints: list[str]


def run_cmd(
    command: list[str], cwd: Path, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    workspace_venv: Path | None = None
    if (cwd / ".venv").exists():
        workspace_venv = cwd / ".venv"
    elif (cwd.parent / ".venv").exists():
        workspace_venv = cwd.parent / ".venv"

    if workspace_venv is not None:
        env["VIRTUAL_ENV"] = str(workspace_venv)
        env["POETRY_VIRTUALENVS_CREATE"] = "false"
        env["POETRY_VIRTUALENVS_IN_PROJECT"] = "false"
        env["PATH"] = f"{workspace_venv / 'bin'}:{env.get('PATH', '')}"

    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        msg = f"Command not found: {command[0]}"
        raise RuntimeError(msg) from exc


def discover_projects(root: Path) -> list[Path]:
    projects: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "pyproject.toml").exists() and (entry / "src").is_dir():
            projects.append(entry)
    return projects


def load_pyproject(project_dir: Path) -> dict[str, object]:
    pyproject = project_dir / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data


def get_poetry_dependencies(data: dict[str, object]) -> set[str]:
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return set()
    poetry = tool.get("poetry")
    if not isinstance(poetry, dict):
        return set()

    deps: set[str] = set()

    main_deps = poetry.get("dependencies")
    if isinstance(main_deps, dict):
        deps.update(str(k) for k in main_deps)

    group = poetry.get("group")
    if isinstance(group, dict):
        for group_def in group.values():
            if not isinstance(group_def, dict):
                continue
            group_deps = group_def.get("dependencies")
            if isinstance(group_deps, dict):
                deps.update(str(k) for k in group_deps)
    return deps


def parse_mypy_signal(output: str) -> tuple[list[str], list[str]]:
    hinted: set[str] = set()
    missing_libs: set[str] = set()

    for match in MYPY_HINT_RE.finditer(output):
        pkg = match.group(1).strip()
        if pkg:
            hinted.add(pkg)

    for match in MYPY_STUB_RE.finditer(output):
        lib = match.group(1).strip()
        if lib:
            missing_libs.add(lib)

    return sorted(hinted), sorted(missing_libs)


def run_mypy_signal(project_dir: Path) -> tuple[list[str], list[str], str]:
    command = [
        "poetry",
        "run",
        "mypy",
        "src",
        "--config-file",
        "pyproject.toml",
        "--no-error-summary",
    ]
    result = run_cmd(command, cwd=project_dir, timeout=1200)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    hinted, missing_libs = parse_mypy_signal(output)
    return hinted, missing_libs, output


def install_types_packages(
    project_dir: Path,
    packages: list[str],
    apply: bool,
) -> tuple[list[str], list[str]]:
    existing = get_poetry_dependencies(load_pyproject(project_dir))
    missing = [pkg for pkg in packages if pkg not in existing]
    if not missing:
        return [], []
    if not apply:
        return [], missing

    added: list[str] = []
    for pkg in sorted(missing):
        add_result = run_cmd(
            ["poetry", "add", "--group", "typings", pkg],
            cwd=project_dir,
            timeout=1200,
        )
        if add_result.returncode != 0:
            msg = (
                f"Failed to add {pkg} in {project_dir.name}:\n"
                f"{(add_result.stdout or '')}\n{(add_result.stderr or '')}"
            )
            raise RuntimeError(msg)
        added.append(pkg)
    return added, []


def parse_pyrefly_missing_imports(output: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in MISSING_IMPORT_RE.finditer(output):
        module_name = match.group(1).strip()
        if module_name and module_name not in seen:
            seen.add(module_name)
            ordered.append(module_name)
    return ordered


def project_missing_imports(project_dir: Path) -> list[str]:
    result = run_cmd(
        ["poetry", "run", "pyrefly", "check", "src", "--config", "pyproject.toml"],
        cwd=project_dir,
        timeout=1200,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return parse_pyrefly_missing_imports(output)


def is_internal_module(module_name: str, project_name: str) -> bool:
    root = module_name.split(".", 1)[0]
    project_root = project_name.replace("-", "_")
    if root.startswith(INTERNAL_PREFIXES):
        return True
    return root == project_root


def existing_stub(module_name: str, root: Path) -> bool:
    rel = module_name.replace(".", "/")
    manual = root / "typings" / rel
    generated = root / "typings" / "generated" / rel
    candidates = [
        manual.with_suffix(".pyi"),
        generated.with_suffix(".pyi"),
        manual / "__init__.pyi",
        generated / "__init__.pyi",
    ]
    return any(path.exists() for path in candidates)


def stub_presence(module_name: str, root: Path) -> tuple[bool, bool]:
    rel = module_name.replace(".", "/")
    manual = root / "typings" / rel
    generated = root / "typings" / "generated" / rel
    manual_candidates = [
        manual.with_suffix(".pyi"),
        manual / "__init__.pyi",
    ]
    generated_candidates = [
        generated.with_suffix(".pyi"),
        generated / "__init__.pyi",
    ]
    has_manual = any(path.exists() for path in manual_candidates)
    has_generated = any(path.exists() for path in generated_candidates)
    return has_manual, has_generated


def snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*.pyi")):
        rel = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[rel] = digest
    return snapshot


def merge_generated_stubs(temp_root: Path, dest_root: Path) -> list[str]:
    created: list[str] = []
    if not temp_root.exists():
        return created
    for source in sorted(temp_root.rglob("*.pyi")):
        rel = source.relative_to(temp_root)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            _ = shutil.copy2(source, dest)
            created.append(str(Path("typings/generated") / rel))
    return created


def run_stubgen_for_module(
    module_name: str, temp_root: Path, project_dir: Path
) -> bool:
    pkg_result = run_cmd(
        [
            "poetry",
            "run",
            "stubgen",
            "-p",
            module_name,
            "-o",
            str(temp_root),
            "--ignore-errors",
            "--inspect-mode",
            "--export-less",
        ],
        cwd=project_dir,
        timeout=1200,
    )
    if pkg_result.returncode == 0:
        return True

    module_result = run_cmd(
        [
            "poetry",
            "run",
            "stubgen",
            "-m",
            module_name,
            "-o",
            str(temp_root),
            "--ignore-errors",
            "--inspect-mode",
            "--export-less",
        ],
        cwd=project_dir,
        timeout=1200,
    )
    return module_result.returncode == 0


def generate_stubgen_stubs(
    project_dir: Path,
    modules: list[str],
    root: Path,
    apply: bool,
) -> list[str]:
    if not modules or not apply:
        return []

    generated_root = root / "typings" / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    with tempfile.TemporaryDirectory(prefix="stubgen_") as tmp:
        tmp_root = Path(tmp)
        for module_name in sorted(modules):
            ok = run_stubgen_for_module(module_name, tmp_root, project_dir)
            if not ok:
                msg = f"stubgen failed for module '{module_name}' in {project_dir.name}"
                raise RuntimeError(msg)
        created.extend(merge_generated_stubs(tmp_root, generated_root))
    return sorted(created)


def process_project(project_dir: Path, root: Path, apply: bool) -> ProjectResult:
    mypy_hints, _, _ = run_mypy_signal(project_dir)
    types_added, types_missing = install_types_packages(project_dir, mypy_hints, apply)

    missing_imports = project_missing_imports(project_dir)
    internal_missing = [
        module_name
        for module_name in missing_imports
        if is_internal_module(module_name, project_dir.name)
    ]
    third_party_missing = [
        module_name
        for module_name in missing_imports
        if not is_internal_module(module_name, project_dir.name)
    ]
    unresolved: list[str] = []
    for module_name in third_party_missing:
        _, has_generated = stub_presence(module_name, root)
        if not has_generated:
            unresolved.append(module_name)

    generated_files: list[str] = []
    if apply:
        generated_files = generate_stubgen_stubs(project_dir, unresolved, root, apply)
        unresolved = [
            module_name
            for module_name in unresolved
            if not existing_stub(module_name, root)
        ]

    return ProjectResult(
        project=project_dir.name,
        internal_missing_imports=sorted(internal_missing),
        unresolved_missing_imports=sorted(unresolved),
        generated_files=sorted(generated_files),
        types_packages_added=sorted(types_added),
        types_packages_missing=sorted(types_missing),
        mypy_stub_hints=sorted(mypy_hints),
    )


def write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Typing stub supply-chain gate")
    _ = parser.add_argument("--all", action="store_true", help="Process all projects")
    _ = parser.add_argument(
        "--project", action="append", default=[], help="Project name"
    )
    _ = parser.add_argument("--apply", action="store_true", help="Apply updates")
    _ = parser.add_argument(
        "--report",
        default=".reports/validate/stub-supply-chain.json",
        help="JSON report path",
    )
    _ = parser.add_argument(
        "--idempotency-check",
        action="store_true",
        help="Require second-pass zero pending work after apply (enabled automatically with --apply)",
    )
    args = parser.parse_args()

    root = Path.cwd()
    projects = discover_projects(root)
    if args.project:
        wanted = set(args.project)
        projects = [project for project in projects if project.name in wanted]
    if not projects:
        print("ERROR: no projects found", file=sys.stderr)
        return 2

    generated_root = root / "typings" / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    before_snapshot = snapshot_tree(generated_root)

    results: list[ProjectResult] = []
    for project in projects:
        results.append(process_project(project, root, args.apply))

    after_snapshot = snapshot_tree(generated_root)

    internal_total = sum(len(result.internal_missing_imports) for result in results)
    unresolved_total = sum(len(result.unresolved_missing_imports) for result in results)
    generated_total = sum(len(result.generated_files) for result in results)
    types_added_total = sum(len(result.types_packages_added) for result in results)
    types_missing_total = sum(len(result.types_packages_missing) for result in results)

    idempotency_enabled = bool(args.apply or args.idempotency_check)
    idempotency_pending = 0
    if idempotency_enabled:
        second_pass: list[ProjectResult] = []
        for project in projects:
            second_pass.append(process_project(project, root, apply=False))
        idempotency_pending = sum(
            len(item.unresolved_missing_imports) + len(item.types_packages_missing)
            for item in second_pass
        )

    report = {
        "projects": [
            {
                "project": result.project,
                "internal_missing_imports": result.internal_missing_imports,
                "unresolved_missing_imports": result.unresolved_missing_imports,
                "generated_files": result.generated_files,
                "types_packages_added": result.types_packages_added,
                "types_packages_missing": result.types_packages_missing,
                "mypy_stub_hints": result.mypy_stub_hints,
            }
            for result in results
        ],
        "summary": {
            "project_count": len(results),
            "internal_missing_imports": internal_total,
            "unresolved_missing_imports": unresolved_total,
            "generated_stub_files": generated_total,
            "types_packages_added": types_added_total,
            "types_packages_missing": types_missing_total,
            "generated_snapshot_before": len(before_snapshot),
            "generated_snapshot_after": len(after_snapshot),
            "generated_snapshot_changed": before_snapshot != after_snapshot,
            "idempotency_pending": idempotency_pending,
            "apply": bool(args.apply),
            "idempotency_check": idempotency_enabled,
        },
    }
    write_report(Path(args.report), report)

    for result in results:
        line = (
            f"{result.project}: internal={len(result.internal_missing_imports)} "
            + f"unresolved={len(result.unresolved_missing_imports)} "
            + f"types_added={len(result.types_packages_added)} generated={len(result.generated_files)}"
        )
        print(line)

    print(f"Report: {args.report}")
    if internal_total > 0:
        print(
            f"FAIL: internal missing imports detected ({internal_total})",
            file=sys.stderr,
        )
        return 1
    if types_missing_total > 0:
        print(
            f"FAIL: missing types-* packages remain ({types_missing_total})",
            file=sys.stderr,
        )
        return 1
    if unresolved_total > 0:
        print(
            f"FAIL: unresolved third-party missing imports remain ({unresolved_total})",
            file=sys.stderr,
        )
        return 1
    if idempotency_enabled and idempotency_pending > 0:
        print(
            f"FAIL: idempotency pending work remains ({idempotency_pending})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
