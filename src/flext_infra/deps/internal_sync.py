"""Internal dependency synchronization service for managing FLEXT submodule dependencies."""

from __future__ import annotations

import configparser
import os
import re
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_core import FlextLogger, r
from flext_infra import (
    c,
    m,
    p,
    t,
    u,
)


class FlextInfraInternalDependencySyncService:
    """Synchronize internal FLEXT dependencies via git clone or workspace symlinks."""

    log = FlextLogger.create_module_logger(__name__)

    def __init__(self) -> None:
        """Initialize the internal dependency sync service."""
        self.toml: p.Infra.TomlReader | None = None

    def _read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
        if self.toml is not None:
            return self.toml.read_plain(path)
        return u.Infra.read_plain(path)

    @staticmethod
    def ensure_symlink(target: Path, source: Path) -> r[bool]:
        """Ensure target points to source via directory symlink."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() and target.resolve() == source.resolve():
                return r[bool].ok(True)
            if target.exists() or target.is_symlink():
                if target.is_dir() and (not target.is_symlink()):
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.symlink_to(source.resolve(), target_is_directory=True)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"failed to ensure symlink for {target}: {exc}")

    @staticmethod
    def is_internal_path_dep(raw_path: str) -> str | None:
        """Resolve repository name from internal path dependency notation."""
        normalized = raw_path.strip().removeprefix("./")
        if normalized.startswith(".flext-deps/"):
            return normalized.removeprefix(".flext-deps/")
        if normalized.startswith("../"):
            candidate = normalized.removeprefix("../")
            if candidate and "/" not in candidate:
                return candidate
        if normalized and "/" not in normalized and (normalized not in {".", ".."}):
            return normalized
        return None

    @staticmethod
    def is_relative_to(path: Path, parent: Path) -> bool:
        """Return whether path is located under parent."""
        try:
            _ = path.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def owner_from_remote_url(remote_url: str) -> str | None:
        """Extract GitHub owner from supported remote URL formats."""
        patterns = (
            "^git@github\\.com:(?P<owner>[^/]+)/[^/]+(?:\\.git)?$",
            "^https://github\\.com/(?P<owner>[^/]+)/[^/]+(?:\\.git)?$",
            "^http://github\\.com/(?P<owner>[^/]+)/[^/]+(?:\\.git)?$",
        )
        for pattern in patterns:
            match = re.match(pattern, remote_url)
            if match:
                return match.group("owner")
        return None

    @staticmethod
    def ssh_to_https(url: str) -> str:
        """Convert GitHub SSH URL to HTTPS URL when needed."""
        if url.startswith("git@github.com:"):
            return f"https://github.com/{url.removeprefix('git@github.com:')}"
        return url

    @staticmethod
    def validate_git_ref(ref_name: str) -> r[str]:
        """Validate git reference name using project-safe regex."""
        if not c.Infra.Deps.GIT_REF_RE.fullmatch(ref_name):
            return r[str].fail(f"invalid git ref: {ref_name!r}")
        return r[str].ok(ref_name)

    @staticmethod
    def validate_repo_url(repo_url: str) -> r[str]:
        """Validate repository URL against allowed GitHub format."""
        if not c.Infra.Deps.GITHUB_REPO_URL_RE.fullmatch(repo_url):
            return r[str].fail(f"invalid repository URL: {repo_url!r}")
        return r[str].ok(repo_url)

    @staticmethod
    def workspace_root_from_parents(project_root: Path) -> Path | None:
        """Locate workspace root by scanning parent directories for .gitmodules."""
        for candidate in (project_root, *project_root.parents):
            if (candidate / c.Infra.Files.GITMODULES).exists():
                return candidate
        return None

    def sync(self, project_root: Path) -> r[int]:
        """Synchronize internal dependencies via git clone or workspace symlinks."""
        deps_result = self.collect_internal_deps(project_root)
        if deps_result.is_failure:
            return r[int].fail(deps_result.error or "dependency collection failed")
        deps = deps_result.value
        if not deps:
            return r[int].ok(0)
        workspace_mode, workspace_root = self.is_workspace_mode(project_root)
        map_file = project_root / "flext-repo-map.toml"
        repo_map: Mapping[str, m.Infra.Github.RepoUrls]
        if (
            workspace_mode
            and workspace_root
            and (workspace_root / c.Infra.Files.GITMODULES).exists()
        ):
            repo_map = self.parse_gitmodules(workspace_root / c.Infra.Files.GITMODULES)
            if map_file.exists():
                parsed_map_result = self.parse_repo_map(map_file)
                if parsed_map_result.is_failure:
                    return r[int].fail(
                        parsed_map_result.error or "failed to parse standalone map",
                    )
                repo_map = {**parsed_map_result.value, **repo_map}
        elif not map_file.exists():
            owner = self.infer_owner_from_origin(project_root)
            if owner is None:
                return r[int].fail(
                    "missing flext-repo-map.toml for standalone dependency resolution and unable to infer GitHub owner from remote.origin.url",
                )
            repo_map = self.synthesized_repo_map(
                owner,
                {dep_path.name for dep_path in deps.values()},
            )
            self.log.warning("sync_deps_synthesized_repo_map", owner=owner)
        else:
            parsed_map_result = self.parse_repo_map(map_file)
            if parsed_map_result.is_failure:
                return r[int].fail(
                    parsed_map_result.error or "failed to parse repo map",
                )
            repo_map = parsed_map_result.value
        ref_name = self.resolve_ref(project_root)
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
                    symlink_result = self.ensure_symlink(dep_path, sibling)
                    if symlink_result.is_failure:
                        return r[int].fail(
                            symlink_result.error or f"failed symlink for {repo_name}",
                        )
                    continue
            urls = repo_map[repo_name]
            selected_url = urls.https_url if force_https else urls.ssh_url
            checkout_result = self.ensure_checkout(dep_path, selected_url, ref_name)
            if checkout_result.is_failure:
                return r[int].fail(
                    checkout_result.error or f"checkout failed for {repo_name}",
                )
        return r[int].ok(0)

    def collect_internal_deps(self, project_root: Path) -> r[Mapping[str, Path]]:
        """Collect internal path dependencies from pyproject metadata."""
        pyproject = project_root / c.Infra.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            return r[Mapping[str, Path]].ok({})
        data_result = self._read_plain(pyproject)
        if data_result.is_failure:
            return r[Mapping[str, Path]].fail(
                data_result.error or f"failed to read {pyproject}",
            )
        data = data_result.value
        tool = self._normalize_str_object_mapping(data.get(c.Infra.Toml.TOOL))
        poetry = self._normalize_str_object_mapping(tool.get(c.Infra.Toml.POETRY))
        deps = self._normalize_str_object_mapping(poetry.get(c.Infra.Toml.DEPENDENCIES))
        result: MutableMapping[str, Path] = {}
        for dep_name, dep_value in deps.items():
            dep_value_map = self._normalize_str_object_mapping(dep_value)
            if not dep_value_map:
                continue
            dep_path = dep_value_map.get(c.Infra.Toml.PATH)
            if not isinstance(dep_path, str):
                continue
            repo_name = self.is_internal_path_dep(dep_path)
            if repo_name is None:
                continue
            result[dep_name] = project_root / ".flext-deps" / repo_name
        project_obj = self._normalize_str_object_mapping(data.get(c.Infra.Toml.PROJECT))
        project_deps_raw = project_obj.get(c.Infra.Toml.DEPENDENCIES)
        project_deps = self._normalize_string_list(project_deps_raw)
        for dep in project_deps:
            if " @ " not in dep:
                continue
            match = c.Infra.Deps.PEP621_PATH_RE.search(dep)
            if not match:
                continue
            repo_name = self.is_internal_path_dep(match.group("path"))
            if repo_name is None:
                continue
            _ = result.setdefault(repo_name, project_root / ".flext-deps" / repo_name)
        return r[Mapping[str, Path]].ok(result)

    def ensure_checkout(self, dep_path: Path, repo_url: str, ref_name: str) -> r[bool]:
        """Ensure dependency checkout exists and matches requested ref."""
        safe_repo_url_result = self.validate_repo_url(repo_url)
        if safe_repo_url_result.is_failure:
            return r[bool].fail(safe_repo_url_result.error or "invalid repository URL")
        safe_ref_name_result = self.validate_git_ref(ref_name)
        if safe_ref_name_result.is_failure:
            return r[bool].fail(safe_ref_name_result.error or "invalid git ref")
        safe_repo_url = safe_repo_url_result.value
        safe_ref_name = safe_ref_name_result.value
        dep_path.parent.mkdir(parents=True, exist_ok=True)
        if not (dep_path / c.Infra.Git.DIR).exists():
            try:
                if dep_path.exists() or dep_path.is_symlink():
                    if dep_path.is_dir() and (not dep_path.is_symlink()):
                        shutil.rmtree(dep_path)
                    else:
                        dep_path.unlink()
            except OSError as exc:
                return r[bool].fail(f"cleanup failed for {dep_path.name}: {exc}")
            cloned = u.Infra.git_run_checked([
                "clone",
                "--depth",
                "1",
                "--branch",
                safe_ref_name,
                safe_repo_url,
                str(dep_path),
            ])
            if cloned.is_failure:
                return r[bool].fail(f"clone failed for {dep_path.name}: {cloned.error}")
            return r[bool].ok(True)
        fetch = u.Infra.git_fetch(dep_path, c.Infra.Git.ORIGIN)
        if fetch.is_failure:
            return r[bool].fail(f"fetch failed for {dep_path.name}: {fetch.error}")
        checkout = u.Infra.git_checkout(dep_path, safe_ref_name)
        if checkout.is_failure:
            return r[bool].fail(
                f"checkout failed for {dep_path.name}: {checkout.error}",
            )
        _ = u.Infra.git_pull(dep_path, remote=c.Infra.Git.ORIGIN, branch=safe_ref_name)
        return r[bool].ok(True)

    def infer_owner_from_origin(self, project_root: Path) -> str | None:
        """Infer GitHub owner from remote origin URL."""
        remote = u.Infra.git_run(
            ["config", "--get", "remote.origin.url"], cwd=project_root
        )
        if remote.is_failure:
            return None
        remote_val = remote.value
        if isinstance(remote_val, str):
            return self.owner_from_remote_url(remote_val.strip())
        return None

    def is_workspace_mode(self, project_root: Path) -> tuple[bool, Path | None]:
        """Determine workspace mode and return resolved workspace root."""
        if os.getenv("FLEXT_STANDALONE") == "1":
            u.Infra.info("Standalone mode: skipping workspace dependency sync")
            return (False, None)
        env_workspace_root = self.workspace_root_from_env(project_root)
        if env_workspace_root is not None:
            return (True, env_workspace_root)
        superproject = u.Infra.git_run(
            ["rev-parse", "--show-superproject-working-tree"],
            cwd=project_root,
        )
        if superproject.is_success:
            sp_val = superproject.value
            value = sp_val.strip() if isinstance(sp_val, str) else ""
            if value:
                return (True, Path(value))
        heuristic_workspace_root = self.workspace_root_from_parents(project_root)
        if heuristic_workspace_root is not None:
            return (True, heuristic_workspace_root)
        return (False, None)

    def parse_gitmodules(self, path: Path) -> Mapping[str, m.Infra.Github.RepoUrls]:
        """Parse .gitmodules file into repo URL mapping."""
        parser = configparser.RawConfigParser()
        _ = parser.read(path)
        mapping: MutableMapping[str, m.Infra.Github.RepoUrls] = {}
        for section in parser.sections():
            if not section.startswith("submodule "):
                continue
            repo_name = section.split('"')[1]
            repo_url = parser.get(section, c.Infra.ReportKeys.URL, fallback="").strip()
            if not repo_url:
                continue
            mapping[repo_name] = m.Infra.Github.RepoUrls(
                ssh_url=repo_url,
                https_url=self.ssh_to_https(repo_url),
            )
        return mapping

    def parse_repo_map(self, path: Path) -> r[Mapping[str, m.Infra.Github.RepoUrls]]:
        """Parse flext-repo-map TOML into repository URL entries."""
        data_result = self._read_plain(path)
        if data_result.is_failure:
            return r[Mapping[str, m.Infra.Github.RepoUrls]].fail(
                data_result.error or "failed to read repository map",
            )
        data = data_result.value
        repos_obj = self._normalize_str_object_mapping(data.get("repo", {}))
        if not repos_obj:
            return r[Mapping[str, m.Infra.Github.RepoUrls]].ok({})
        result: MutableMapping[str, m.Infra.Github.RepoUrls] = {}
        for repo_name, values in repos_obj.items():
            values_map = self._normalize_str_object_mapping(values)
            if not values_map:
                continue
            ssh_url = str(values_map.get("ssh_url", ""))
            https_url = str(values_map.get("https_url", self.ssh_to_https(ssh_url)))
            if ssh_url:
                result[repo_name] = m.Infra.Github.RepoUrls(
                    ssh_url=ssh_url,
                    https_url=https_url,
                )
        return r[Mapping[str, m.Infra.Github.RepoUrls]].ok(result)

    @staticmethod
    def _normalize_str_object_mapping(
        value: t.Infra.InfraValue,
    ) -> dict[str, t.Infra.InfraValue]:
        try:
            adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            )
            return adapter.validate_python(value)
        except ValidationError:
            return {}

    @staticmethod
    def _normalize_string_list(value: t.Infra.InfraValue) -> list[str]:
        try:
            adapter: TypeAdapter[list[str]] = TypeAdapter(list[str])
            return adapter.validate_python(value)
        except ValidationError:
            if not isinstance(value, list):
                return []
            raw_items: list[JsonValue] = TypeAdapter(
                list[JsonValue],
            ).validate_python(value)
            return [str(item) for item in raw_items]

    def resolve_ref(self, project_root: Path) -> str:
        """Resolve dependency sync git reference for current environment."""
        if os.getenv("GITHUB_ACTIONS") == "true":
            for key in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
                value = os.getenv(key)
                if value:
                    return value
        branch = u.Infra.git_current_branch(project_root)
        if branch.is_success:
            branch_val = branch.value
            current = branch_val.strip() if isinstance(branch_val, str) else ""
            if current and current != c.Infra.Git.HEAD:
                return current
        tag = u.Infra.git_run(["describe", "--tags", "--exact-match"], cwd=project_root)
        if tag.is_success:
            tag_val = tag.value
            return tag_val.strip() if isinstance(tag_val, str) else c.Infra.Git.MAIN
        return c.Infra.Git.MAIN

    def synthesized_repo_map(
        self,
        owner: str,
        repo_names: set[str],
    ) -> Mapping[str, m.Infra.Github.RepoUrls]:
        """Build default repository URL mapping from owner and repo set."""
        result: MutableMapping[str, m.Infra.Github.RepoUrls] = {}
        for repo_name in sorted(repo_names):
            ssh_url = f"git@github.com:{owner}/{repo_name}.git"
            result[repo_name] = m.Infra.Github.RepoUrls(
                ssh_url=ssh_url,
                https_url=self.ssh_to_https(ssh_url),
            )
        return result

    def workspace_root_from_env(self, project_root: Path) -> Path | None:
        """Resolve workspace root from environment when valid for project root."""
        env_root = os.getenv("FLEXT_WORKSPACE_ROOT")
        if not env_root:
            return None
        candidate = Path(env_root).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            return None
        if self.is_relative_to(project_root, candidate):
            return candidate
        return None


def main() -> int:
    """Entry point for internal dependency synchronization CLI."""
    parser = u.Infra.create_parser(
        prog="flext-infra deps internal-sync",
        description="Synchronize internal FLEXT dependencies via git clone or workspace symlinks",
        include_apply=False,
    )
    args = parser.parse_args()
    cli_args = u.Infra.resolve(args)
    service = FlextInfraInternalDependencySyncService()
    result = service.sync(cli_args.workspace)
    if result.is_success:
        return result.value
    sync_error = result.error or "sync_internal_deps_failed"
    service.log.error("sync_internal_deps_failed", error=sync_error)
    u.Infra.error(f"[sync-deps] {sync_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
__all__ = ["FlextInfraInternalDependencySyncService", "shutil"]
