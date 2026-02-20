#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-dependencies/SKILL.md
"""Sync internal FLEXT path dependencies for workspace/standalone modes."""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

GIT_BIN = shutil.which("git") or "git"


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [GIT_BIN, *args], cwd=cwd, text=True, capture_output=True, check=False
    )


def _ssh_to_https(url: str) -> str:
    if url.startswith("git@github.com:"):
        return f"https://github.com/{url.removeprefix('git@github.com:')}"
    return url


def _parse_gitmodules(path: Path) -> dict[str, dict[str, str]]:
    parser = configparser.RawConfigParser()
    parser.read(path)
    mapping: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        repo_name = section.split('"')[1]
        repo_url = parser.get(section, "url", fallback="").strip()
        if not repo_url:
            continue
        mapping[repo_name] = {
            "ssh_url": repo_url,
            "https_url": _ssh_to_https(repo_url),
        }
    return mapping


def _parse_repo_map(path: Path) -> dict[str, dict[str, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    repos = data.get("repo", {})
    result: dict[str, dict[str, str]] = {}
    for repo_name, values in repos.items():
        ssh_url = str(values.get("ssh_url", ""))
        https_url = str(values.get("https_url", _ssh_to_https(ssh_url)))
        if ssh_url:
            result[repo_name] = {"ssh_url": ssh_url, "https_url": https_url}
    return result


def _resolve_ref(project_root: Path) -> str:
    if os.getenv("GITHUB_ACTIONS") == "true":
        for key in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
            value = os.getenv(key)
            if value:
                return value
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    if branch.returncode == 0:
        current = branch.stdout.strip()
        if current and current != "HEAD":
            return current
    tag = _run_git(["describe", "--tags", "--exact-match"], project_root)
    if tag.returncode == 0:
        return tag.stdout.strip()
    return "main"


def _is_workspace_mode(project_root: Path) -> tuple[bool, Path | None]:
    if os.getenv("FLEXT_STANDALONE") == "1":
        return False, None
    superproject = _run_git(
        ["rev-parse", "--show-superproject-working-tree"], project_root
    )
    if superproject.returncode == 0:
        value = superproject.stdout.strip()
        if value:
            return True, Path(value)
    if (project_root / ".gitmodules").exists():
        return True, project_root
    return False, None


def _ensure_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source.resolve(), target_is_directory=True)


def _ensure_checkout(dep_path: Path, repo_url: str, ref_name: str) -> None:
    dep_path.parent.mkdir(parents=True, exist_ok=True)
    if not (dep_path / ".git").exists():
        cloned = subprocess.run(
            [
                GIT_BIN,
                "clone",
                "--depth",
                "1",
                "--branch",
                ref_name,
                repo_url,
                str(dep_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if cloned.returncode == 0:
            return
        fallback = subprocess.run(
            [
                GIT_BIN,
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                repo_url,
                str(dep_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if fallback.returncode != 0:
            error_msg = f"clone failed for {dep_path.name}: {fallback.stderr.strip()}"
            raise RuntimeError(error_msg)
        print(
            f"[sync-deps] warning: {dep_path.name} missing ref '{ref_name}', using 'main'"
        )
        return

    fetch = _run_git(["fetch", "origin", "--tags"], dep_path)
    if fetch.returncode != 0:
        error_msg = f"fetch failed for {dep_path.name}: {fetch.stderr.strip()}"
        raise RuntimeError(error_msg)

    checkout = _run_git(["checkout", ref_name], dep_path)
    if checkout.returncode == 0:
        _run_git(["pull", "--ff-only", "origin", ref_name], dep_path)
        return

    fallback_checkout = _run_git(["checkout", "main"], dep_path)
    if fallback_checkout.returncode != 0:
        error_msg = f"checkout failed for {dep_path.name}: {checkout.stderr.strip()}"
        raise RuntimeError(error_msg)
    _run_git(["pull", "--ff-only", "origin", "main"], dep_path)
    print(
        f"[sync-deps] warning: {dep_path.name} missing ref '{ref_name}', using 'main'"
    )


def _collect_internal_deps(project_root: Path) -> dict[str, Path]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    result: dict[str, Path] = {}
    for dep_name, dep_value in deps.items():
        if not isinstance(dep_value, dict):
            continue
        dep_path = dep_value.get("path")
        if not isinstance(dep_path, str):
            continue
        if not dep_path.startswith(".flext-deps/"):
            continue
        result[dep_name] = project_root / dep_path

    project_deps = data.get("project", {}).get("dependencies", [])
    dep_pattern = re.compile(r"@\s*\.\.?/\.flext-deps/([A-Za-z0-9_.-]+)")
    for dep in project_deps:
        if not isinstance(dep, str):
            continue
        match = dep_pattern.search(dep)
        if not match:
            continue
        repo_name = match.group(1)
        result.setdefault(repo_name, project_root / ".flext-deps" / repo_name)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    deps = _collect_internal_deps(project_root)
    if not deps:
        return 0

    workspace_mode, workspace_root = _is_workspace_mode(project_root)
    repo_map: dict[str, dict[str, str]]
    map_file = project_root / "flext-repo-map.toml"
    if workspace_mode and workspace_root and (workspace_root / ".gitmodules").exists():
        repo_map = _parse_gitmodules(workspace_root / ".gitmodules")
        if map_file.exists():
            repo_map = {**_parse_repo_map(map_file), **repo_map}
    else:
        if not map_file.exists():
            error_msg = (
                "missing flext-repo-map.toml for standalone dependency resolution"
            )
            raise RuntimeError(error_msg)
        repo_map = _parse_repo_map(map_file)

    ref_name = _resolve_ref(project_root)
    force_https = (
        os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("FLEXT_USE_HTTPS") == "1"
    )

    for dep_path in deps.values():
        repo_name = dep_path.name
        if repo_name not in repo_map:
            error_msg = f"missing repo mapping for {repo_name}"
            raise RuntimeError(error_msg)

        if workspace_mode and workspace_root:
            sibling = workspace_root / repo_name
            if sibling.exists():
                _ensure_symlink(dep_path, sibling)
                continue

        urls = repo_map[repo_name]
        selected_url = urls["https_url"] if force_https else urls["ssh_url"]
        _ensure_checkout(dep_path, selected_url, ref_name)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except Exception as exc:
        print(f"[sync-deps] error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
