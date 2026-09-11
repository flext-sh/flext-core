"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from configparser import Error as ConfigParserError
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from git import GitCommandError, GitConfigParser

from flext_core import r
from flext_infra._utilities._git.worktree_roots import (
    FlextInfraUtilitiesGitWorktreeRootsMixin,
)
from flext_infra._utilities._sort_keys import path_depth_then_text
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitWorktreeDiscoveryMixin(
    FlextInfraUtilitiesGitWorktreeRootsMixin
):
    """Own worktree discovery operations."""

    @staticmethod
    def git_remote_identity(url: str) -> str:
        """Normalize remotes to owner/repo identity across HTTPS, SSH, and aliases.

        CI deploy-key init rewrites private member ``origin`` to an SSH URL that
        may use a Host alias (for example ``git@charts-github:org/repo.git``)
        while the workspace manifest and ``.gitmodules`` keep HTTPS on
        ``github.com``. Compare the repository path only so gen does not
        false-fail after a successful private checkout.
        """
        value = url.strip().removesuffix(".git")
        remote_path = ""
        if value.startswith("git@"):
            host_path = value.removeprefix("git@")
            if ":" in host_path:
                _host, remote_path = host_path.split(":", 1)
        else:
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https", "ssh"} and parsed.netloc:
                remote_path = parsed.path.lstrip("/")
            else:
                remote_path = value
        parts = [part for part in remote_path.split("/") if part]
        match parts:
            case [*_, owner, repo]:
                return f"{owner}/{repo}".lower()
            case _:
                return remote_path.lower()

    @classmethod
    def git_declared_submodule_paths(
        cls, repository_root: Path
    ) -> p.Result[t.SequenceOf[Path]]:
        """Read every valid path declared by the repository's ``.gitmodules``.

        Unlike ``git submodule status``, this contract includes uninitialized
        submodules and treats an empty file as an empty topology. Malformed,
        duplicate, absolute, or escaping paths fail closed.
        """
        gitmodules = repository_root / c.Infra.GITMODULES
        if not gitmodules.exists():
            return r[t.SequenceOf[Path]].ok(())
        if not gitmodules.is_file():
            return r[t.SequenceOf[Path]].fail(
                f"Git submodule manifest is not a regular file: {gitmodules}"
            )
        try:
            with GitConfigParser(file_or_files=gitmodules, read_only=True) as config:
                raw_paths = tuple(
                    str(config.get_value(section, "path"))
                    for section in config.sections()
                    if section.startswith("submodule ")
                    and config.has_option(section, "path")
                )
        except (ConfigParserError, OSError, TypeError, ValueError) as exc:
            return r[t.SequenceOf[Path]].fail(
                f"failed to read Git submodule declarations: {exc}"
            )
        paths: t.MutableSequenceOf[Path] = []
        for raw_path in raw_paths:
            relative = Path(raw_path)
            if relative.is_absolute() or relative == Path() or ".." in relative.parts:
                return r[t.SequenceOf[Path]].fail(
                    f"invalid Git submodule path: {raw_path}"
                )
            if relative in paths:
                return r[t.SequenceOf[Path]].fail(
                    f"duplicate Git submodule path: {raw_path}"
                )
            paths.append(relative)
        return r[t.SequenceOf[Path]].ok(tuple(paths))

    @classmethod
    def gitmodule_contract(
        cls, request: m.Infra.GitSubmoduleContractRequest
    ) -> p.Result[m.Infra.GitSubmoduleContractReport]:
        """Read the exact declared URL and branch for one submodule path.

        The path must be declared exactly once in ``.gitmodules``; a missing
        URL or branch fails closed.
        """
        gitmodules = request.repo_root / c.Infra.GITMODULES
        try:
            url, branch = cls._read_gitmodule_contract(gitmodules, request.member_path)
        except (ConfigParserError, OSError, TypeError, ValueError) as exc:
            return r[m.Infra.GitSubmoduleContractReport].fail(
                f"failed to read Git submodule paths: {exc}"
            )
        if not url:
            return r[m.Infra.GitSubmoduleContractReport].fail(
                f"Git submodule URL is missing: {request.member_path}"
            )
        if not branch:
            return r[m.Infra.GitSubmoduleContractReport].fail(
                f"Git submodule branch is missing: {request.member_path}"
            )
        return r[m.Infra.GitSubmoduleContractReport].ok(
            m.Infra.GitSubmoduleContractReport(url=url, branch=branch)
        )

    @staticmethod
    def _read_gitmodule_contract(gitmodules: Path, member_path: str) -> tuple[str, str]:
        """Read URL and branch for one submodule from .gitmodules."""
        with GitConfigParser(file_or_files=gitmodules, read_only=True) as config:
            matching_sections = tuple(
                section
                for section in config.sections()
                if section.startswith("submodule ")
                and config.has_option(section, "path")
                and str(config.get_value(section, "path")) == member_path
            )
            if len(matching_sections) != 1:
                msg = f"Git submodule path must be declared exactly once: {member_path}"
                raise ValueError(msg)
            section = matching_sections[0]
            url = (
                str(config.get_value(section, "url")).strip()
                if config.has_option(section, "url")
                else ""
            )
            branch = (
                str(config.get_value(section, "branch")).strip()
                if config.has_option(section, "branch")
                else ""
            )
        return url, branch

    @classmethod
    def git_submodule_paths(cls, repository_root: Path) -> p.Result[t.SequenceOf[Path]]:
        """Resolve every initialized recursive submodule path."""
        try:
            repo = cls._repo(repository_root)
            status = repo.git.submodule("status", "--recursive")
        except GitCommandError as exc:
            return r[t.SequenceOf[Path]].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[t.SequenceOf[Path]].fail(
                f"failed to discover Git submodules: {exc}"
            )
        paths: t.MutableSequenceOf[Path] = []
        for raw_line in status.splitlines():
            normalized = raw_line.strip()
            if not normalized:
                continue
            try:
                _status_and_sha, relative_path_text, *_description = normalized.split(
                    maxsplit=2
                )
            except ValueError:
                continue
            relative_path = Path(relative_path_text)
            if (repository_root / relative_path / ".git").exists():
                paths.append(relative_path)
        return r[t.SequenceOf[Path]].ok(tuple(sorted(paths, key=path_depth_then_text)))


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeDiscoveryMixin"]
