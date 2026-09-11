"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from git import BaseIndexEntry, GitCommandError, Repo

from flext_core import r
from flext_infra._utilities._git.semantic_paths import (
    FlextInfraUtilitiesGitSemanticPathsMixin,
)
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p

_GITLINK_MODE = "160000"
_STAGED_GITLINK_FIELDS = 2


class FlextInfraUtilitiesGitSemanticIndexMixin(
    FlextInfraUtilitiesGitSemanticPathsMixin
):
    """Own semantic index operations."""

    @classmethod
    def git_head_numstat(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitNumstatReport]:
        """Capture HEAD subject and ``HEAD~1..HEAD`` numstat."""
        try:
            repo = cls._repo(request.repo_root)
            subject = repo.git.log("-1", "--format=%s")
            numstat = repo.git.diff("--numstat", "HEAD~1", c.Infra.GIT_HEAD)
        except GitCommandError as exc:
            return r[m.Infra.GitNumstatReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitNumstatReport].fail(f"git numstat read failed: {exc}")
        return r[m.Infra.GitNumstatReport].ok(
            m.Infra.GitNumstatReport(subject=subject.strip(), numstat=numstat)
        )

    @classmethod
    def git_fingerprint_inputs(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[m.Infra.GitFingerprintInputsReport]:
        """Capture byte-exact fingerprint inputs for one worktree."""
        try:
            repo = cls._repo(request.repo_root)
            paths_z, index_z, head = cls._git_capture_fingerprint(repo)
        except GitCommandError as exc:
            return r[m.Infra.GitFingerprintInputsReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitFingerprintInputsReport].fail(
                f"failed to capture fingerprint inputs: {exc}"
            )
        return r[m.Infra.GitFingerprintInputsReport].ok(
            m.Infra.GitFingerprintInputsReport(
                paths_z=paths_z, index_z=index_z, head=head
            )
        )

    @staticmethod
    def _git_capture_fingerprint(repo: Repo) -> tuple[bytes, bytes, bytes]:
        """Capture (paths_z, index_z, head) bytes for fingerprinting."""
        paths_z = repo.git.ls_files(
            "-z", "--cached", "--others", "--exclude-standard"
        ).encode(c.Cli.ENCODING_DEFAULT)
        index_z = repo.git.ls_files("--stage", "-z").encode(c.Cli.ENCODING_DEFAULT)
        try:
            head = repo.head.commit.hexsha.encode(c.Cli.ENCODING_DEFAULT)
        except (ValueError, OSError):
            head = b"UNBORN"
        return paths_z, index_z, head

    @classmethod
    def git_update_index_gitlink(
        cls, request: m.Infra.GitUpdateIndexGitlinkRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Stage one gitlink (mode 160000) into the index."""
        try:
            repo = cls._repo(request.repo_root)
            entry = BaseIndexEntry((
                c.Infra.GIT_MODE_GITLINK,
                bytes.fromhex(request.oid),
                c.Infra.GIT_STAGE_NORMAL,
                request.relative_path,
            ))
            repo.index.add([entry])
        except GitCommandError as exc:
            return r[m.Infra.GitBoolReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(
                f"failed to update-index gitlink: {exc}"
            )
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_gitlink_spec(
        cls, request: m.Infra.GitRefRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Resolve the indexed gitlink oid for one submodule path.

        The ``ls-files --stage`` entry is validated structurally: mode
        ``160000``, stage ``0``, and an exact path match.
        """
        try:
            repo = cls._repo(request.repo_root)
            output = repo.git.ls_files("--stage", "--", request.reference)
        except GitCommandError as exc:
            return r[m.Infra.GitOidReport].fail(str(exc))
        except (OSError, ValueError) as exc:
            return r[m.Infra.GitOidReport].fail(f"failed to read gitlink spec: {exc}")
        if not output.strip():
            return r[m.Infra.GitOidReport].fail(
                f"Git gitlink is missing from the index: {request.reference}"
            )
        match output.split():
            case [mode, oid, stage, indexed_path] if (
                mode == _GITLINK_MODE
                and stage == str(c.Infra.GIT_STAGE_NORMAL)
                and indexed_path == request.reference
            ):
                return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=oid))
            case _:
                return r[m.Infra.GitOidReport].fail(
                    f"Git gitlink entry is malformed: {request.reference}"
                )

    @classmethod
    def git_staged_gitlink_oid(
        cls, request: m.Infra.GitRefRequest
    ) -> p.Result[m.Infra.GitOidReport]:
        """Return the gitlink OID the index records for one submodule path."""
        try:
            repo = cls._repo(request.repo_root)
            staged = repo.git.ls_files("--stage", "--", request.reference)
        except (GitCommandError, OSError, ValueError) as exc:
            return r[m.Infra.GitOidReport].fail(
                f"failed to read the staged gitlink for {request.reference}: {exc}"
            )
        for line in staged.splitlines():
            fields = line.split()
            if len(fields) >= _STAGED_GITLINK_FIELDS and fields[0] == _GITLINK_MODE:
                return r[m.Infra.GitOidReport].ok(m.Infra.GitOidReport(oid=fields[1]))
        return r[m.Infra.GitOidReport].fail(
            f"governed gitlink is absent from the index: {request.reference}"
        )


__all__: list[str] = ["FlextInfraUtilitiesGitSemanticIndexMixin"]
