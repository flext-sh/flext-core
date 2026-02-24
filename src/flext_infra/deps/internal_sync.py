"""Internal dependency synchronization service for managing FLEXT submodule dependencies."""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

import structlog
from flext_core import r
from pydantic import Field

from flext_infra.constants import ic
from flext_infra.models import im
from flext_infra.subprocess import CommandRunner
from flext_infra.toml_io import TomlService

logger = structlog.get_logger(__name__)

GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
GITHUB_REPO_URL_RE = re.compile(
    r"^(?:git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?|https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?)$"
)
_PEP621_PATH_RE = re.compile(r"@\s*(?:file:)?(?P<path>.+)$")


class RepoUrls(im.ArbitraryTypesModel):
    """Repository URL pair with SSH and HTTPS variants."""

    ssh_url: str = Field(default="")
    https_url: str = Field(default="")


class InternalDependencySyncService:
    """Synchronize internal FLEXT dependencies via git clone or workspace symlinks."""

    def __init__(self) -> None:
        """Initialize the internal dependency sync service."""
        self._runner = CommandRunner()
        self._toml = TomlService()

    def _run_git(self, args: list[str], cwd: Path) -> r[im.CommandOutput]:
        return self._runner.run_raw(["git", *args], cwd=cwd)

    @staticmethod
    def _validate_git_ref(ref_name: str) -> r[str]:
        if not GIT_REF_RE.fullmatch(ref_name):
            return r[str].fail(f"invalid git ref: {ref_name!r}")
        return r[str].ok(ref_name)

    @staticmethod
    def _validate_repo_url(repo_url: str) -> r[str]:
        if not GITHUB_REPO_URL_RE.fullmatch(repo_url):
            return r[str].fail(f"invalid repository URL: {repo_url!r}")
        return r[str].ok(repo_url)

    @staticmethod
    def _ssh_to_https(url: str) -> str:
        if url.startswith("git@github.com:"):
            return f"https://github.com/{url.removeprefix('git@github.com:')}"
        return url

    def _parse_gitmodules(self, path: Path) -> Mapping[str, RepoUrls]:
        parser = configparser.RawConfigParser()
        _ = parser.read(path)
        mapping: MutableMapping[str, RepoUrls] = {}
        for section in parser.sections():
            if not section.startswith("submodule "):
                continue
            repo_name = section.split('"')[1]
            repo_url = parser.get(section, "url", fallback="").strip()
            if not repo_url:
                continue
            mapping[repo_name] = RepoUrls(
                ssh_url=repo_url,
                https_url=self._ssh_to_https(repo_url),
            )
        return mapping

    def _parse_repo_map(self, path: Path) -> r[Mapping[str, RepoUrls]]:
        data_result = self._toml.read(path)
        if data_result.is_failure:
            return r[Mapping[str, RepoUrls]].fail(
                data_result.error or "failed to read repository map"
            )
        data = data_result.value
        repos_obj = data.get("repo", {})
        if type(repos_obj) is not dict:
            return r[Mapping[str, RepoUrls]].ok({})
        result: MutableMapping[str, RepoUrls] = {}
        for repo_name, values in repos_obj.items():
            if type(values) is not dict:
                continue
            ssh_url = str(values.get("ssh_url", ""))
            https_url = str(values.get("https_url", self._ssh_to_https(ssh_url)))
            if ssh_url:
                result[repo_name] = RepoUrls(ssh_url=ssh_url, https_url=https_url)
        return r[Mapping[str, RepoUrls]].ok(result)

    def _resolve_ref(self, project_root: Path) -> str:
        if os.getenv("GITHUB_ACTIONS") == "true":
            for key in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
                value = os.getenv(key)
                if value:
                    return value

        branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
        if branch.is_success and branch.value.exit_code == 0:
            current = branch.value.stdout.strip()
            if current and current != "HEAD":
                return current

        tag = self._run_git(["describe", "--tags", "--exact-match"], project_root)
        if tag.is_success and tag.value.exit_code == 0:
            return tag.value.stdout.strip()
        return "main"

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            _ = path.relative_to(parent)
        except ValueError:
            return False
        return True

    def _workspace_root_from_env(self, project_root: Path) -> Path | None:
        env_root = os.getenv("FLEXT_WORKSPACE_ROOT")
        if not env_root:
            return None
        candidate = Path(env_root).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            return None
        if self._is_relative_to(project_root, candidate):
            return candidate
        return None

    @staticmethod
    def _workspace_root_from_parents(project_root: Path) -> Path | None:
        for candidate in (project_root, *project_root.parents):
            if (candidate / ".gitmodules").exists():
                return candidate
        return None

    def _is_workspace_mode(self, project_root: Path) -> tuple[bool, Path | None]:
        if os.getenv("FLEXT_STANDALONE") == "1":
            return False, None

        env_workspace_root = self._workspace_root_from_env(project_root)
        if env_workspace_root is not None:
            return True, env_workspace_root

        superproject = self._run_git(
            ["rev-parse", "--show-superproject-working-tree"],
            project_root,
        )
        if superproject.is_success and superproject.value.exit_code == 0:
            value = superproject.value.stdout.strip()
            if value:
                return True, Path(value)

        heuristic_workspace_root = self._workspace_root_from_parents(project_root)
        if heuristic_workspace_root is not None:
            return True, heuristic_workspace_root

        return False, None

    @staticmethod
    def _owner_from_remote_url(remote_url: str) -> str | None:
        patterns = (
            r"^git@github\.com:(?P<owner>[^/]+)/[^/]+(?:\.git)?$",
            r"^https://github\.com/(?P<owner>[^/]+)/[^/]+(?:\.git)?$",
            r"^http://github\.com/(?P<owner>[^/]+)/[^/]+(?:\.git)?$",
        )
        for pattern in patterns:
            match = re.match(pattern, remote_url)
            if match:
                return match.group("owner")
        return None

    def _infer_owner_from_origin(self, project_root: Path) -> str | None:
        remote = self._run_git(["config", "--get", "remote.origin.url"], project_root)
        if remote.is_failure or remote.value.exit_code != 0:
            return None
        return self._owner_from_remote_url(remote.value.stdout.strip())

    def _synthesized_repo_map(
        self,
        owner: str,
        repo_names: set[str],
    ) -> Mapping[str, RepoUrls]:
        result: MutableMapping[str, RepoUrls] = {}
        for repo_name in sorted(repo_names):
            ssh_url = f"git@github.com:{owner}/{repo_name}.git"
            result[repo_name] = RepoUrls(
                ssh_url=ssh_url,
                https_url=self._ssh_to_https(ssh_url),
            )
        return result

    @staticmethod
    def _ensure_symlink(target: Path, source: Path) -> r[bool]:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == source.resolve():
                return r[bool].ok(True)
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.symlink_to(source.resolve(), target_is_directory=True)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"failed to ensure symlink for {target}: {exc}")

    def _ensure_checkout(self, dep_path: Path, repo_url: str, ref_name: str) -> r[bool]:
        safe_repo_url_result = self._validate_repo_url(repo_url)
        if safe_repo_url_result.is_failure:
            return r[bool].fail(safe_repo_url_result.error or "invalid repository URL")
        safe_ref_name_result = self._validate_git_ref(ref_name)
        if safe_ref_name_result.is_failure:
            return r[bool].fail(safe_ref_name_result.error or "invalid git ref")

        safe_repo_url = safe_repo_url_result.value
        safe_ref_name = safe_ref_name_result.value

        dep_path.parent.mkdir(parents=True, exist_ok=True)
        if not (dep_path / ".git").exists():
            try:
                if dep_path.exists() or dep_path.is_symlink():
                    if dep_path.is_dir() and not dep_path.is_symlink():
                        shutil.rmtree(dep_path)
                    else:
                        dep_path.unlink()
            except OSError as exc:
                return r[bool].fail(f"cleanup failed for {dep_path.name}: {exc}")

            cloned = self._runner.run_raw([
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                safe_ref_name,
                safe_repo_url,
                str(dep_path),
            ])
            if cloned.is_failure or cloned.value.exit_code != 0:
                stderr = cloned.value.stderr.strip() if cloned.is_success else ""
                return r[bool].fail(f"clone failed for {dep_path.name}: {stderr}")
            return r[bool].ok(True)

        fetch = self._run_git(["fetch", "origin", "--tags"], dep_path)
        if fetch.is_failure or fetch.value.exit_code != 0:
            stderr = fetch.value.stderr.strip() if fetch.is_success else ""
            return r[bool].fail(f"fetch failed for {dep_path.name}: {stderr}")

        checkout = self._run_git(["checkout", safe_ref_name], dep_path)
        if checkout.is_failure or checkout.value.exit_code != 0:
            stderr = checkout.value.stderr.strip() if checkout.is_success else ""
            return r[bool].fail(f"checkout failed for {dep_path.name}: {stderr}")
        _ = self._run_git(["pull", "--ff-only", "origin", safe_ref_name], dep_path)
        return r[bool].ok(True)

    @staticmethod
    def _is_internal_path_dep(raw_path: str) -> str | None:
        normalized = raw_path.strip().removeprefix("./")
        if normalized.startswith(".flext-deps/"):
            return normalized.removeprefix(".flext-deps/")
        if normalized.startswith("../"):
            candidate = normalized.removeprefix("../")
            if candidate and "/" not in candidate:
                return candidate
        if normalized and "/" not in normalized and normalized not in {".", ".."}:
            return normalized
        return None

    def _collect_internal_deps(self, project_root: Path) -> r[Mapping[str, Path]]:
        pyproject = project_root / ic.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            return r[Mapping[str, Path]].ok({})

        data_result = self._toml.read(pyproject)
        if data_result.is_failure:
            return r[Mapping[str, Path]].fail(
                data_result.error or f"failed to read {pyproject}"
            )
        data = data_result.value

        tool = data.get("tool")
        poetry = tool.get("poetry") if type(tool) is dict else None
        deps = poetry.get("dependencies") if type(poetry) is dict else {}
        if type(deps) is not dict:
            deps = {}

        result: MutableMapping[str, Path] = {}
        for dep_name, dep_value in deps.items():
            if type(dep_value) is not dict:
                continue
            dep_path = dep_value.get("path")
            if type(dep_path) is not str:
                continue
            repo_name = self._is_internal_path_dep(dep_path)
            if repo_name is None:
                continue
            result[dep_name] = project_root / ".flext-deps" / repo_name

        project_obj = data.get("project")
        project_deps = (
            project_obj.get("dependencies", []) if type(project_obj) is dict else []
        )
        if type(project_deps) is not list:
            project_deps = []

        for dep in project_deps:
            if type(dep) is not str or " @ " not in dep:
                continue
            match = _PEP621_PATH_RE.search(dep)
            if not match:
                continue
            repo_name = self._is_internal_path_dep(match.group("path"))
            if repo_name is None:
                continue
            _ = result.setdefault(repo_name, project_root / ".flext-deps" / repo_name)

        return r[Mapping[str, Path]].ok(result)

    def sync(self, project_root: Path) -> r[int]:
        """Synchronize internal dependencies via git clone or workspace symlinks."""
        deps_result = self._collect_internal_deps(project_root)
        if deps_result.is_failure:
            return r[int].fail(deps_result.error or "dependency collection failed")
        deps = deps_result.value
        if not deps:
            return r[int].ok(0)

        workspace_mode, workspace_root = self._is_workspace_mode(project_root)
        map_file = project_root / "flext-repo-map.toml"
        repo_map: Mapping[str, RepoUrls]

        if (
            workspace_mode
            and workspace_root
            and (workspace_root / ".gitmodules").exists()
        ):
            repo_map = self._parse_gitmodules(workspace_root / ".gitmodules")
            if map_file.exists():
                parsed_map_result = self._parse_repo_map(map_file)
                if parsed_map_result.is_failure:
                    return r[int].fail(
                        parsed_map_result.error or "failed to parse standalone map"
                    )
                repo_map = {**parsed_map_result.value, **repo_map}
        elif not map_file.exists():
            owner = self._infer_owner_from_origin(project_root)
            if owner is None:
                return r[int].fail(
                    "missing flext-repo-map.toml for standalone dependency resolution "
                    "and unable to infer GitHub owner from remote.origin.url"
                )
            repo_map = self._synthesized_repo_map(
                owner,
                {dep_path.name for dep_path in deps.values()},
            )
            logger.warning(
                "sync_deps_synthesized_repo_map",
                owner=owner,
            )
        else:
            parsed_map_result = self._parse_repo_map(map_file)
            if parsed_map_result.is_failure:
                return r[int].fail(
                    parsed_map_result.error or "failed to parse repo map"
                )
            repo_map = parsed_map_result.value

        ref_name = self._resolve_ref(project_root)
        force_https = (
            os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("FLEXT_USE_HTTPS") == "1"
        )

        for dep_path in deps.values():
            repo_name = dep_path.name
            if repo_name not in repo_map:
                return r[int].fail(f"missing repo mapping for {repo_name}")

            if workspace_mode and workspace_root:
                sibling = workspace_root / repo_name
                if sibling.exists():
                    symlink_result = self._ensure_symlink(dep_path, sibling)
                    if symlink_result.is_failure:
                        return r[int].fail(
                            symlink_result.error or f"failed symlink for {repo_name}"
                        )
                    continue

            urls = repo_map[repo_name]
            selected_url = urls.https_url if force_https else urls.ssh_url
            checkout_result = self._ensure_checkout(dep_path, selected_url, ref_name)
            if checkout_result.is_failure:
                return r[int].fail(
                    checkout_result.error or f"checkout failed for {repo_name}"
                )

        return r[int].ok(0)


def main() -> int:
    """Entry point for internal dependency synchronization CLI."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    result = InternalDependencySyncService().sync(project_root)
    if result.is_success:
        return result.value

    logger.error("sync_internal_deps_failed", error=result.error)
    _ = sys.stderr.write(f"[sync-deps] error: {result.error}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["InternalDependencySyncService", "RepoUrls", "main"]
