"""Canonical Git responsibility mixin for ``u.Infra``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from git import GitCommandError

from flext_core import r
from flext_infra._utilities._git.semantic_identity import (
    FlextInfraUtilitiesGitSemanticIdentityMixin,
)
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesGitSemanticSubmoduleMixin(
    FlextInfraUtilitiesGitSemanticIdentityMixin
):
    """Own semantic submodule operations."""

    @classmethod
    def git_submodule_init(
        cls, request: m.Infra.GitRefRequest
    ) -> p.Result[m.Infra.GitBoolReport]:
        """Initialize one declared submodule at its recorded gitlink."""
        try:
            repo = cls._repo(request.repo_root)
            repo.git.submodule("update", "--init", "--", request.reference)
        except (GitCommandError, OSError, ValueError) as exc:
            return r[m.Infra.GitBoolReport].fail(
                f"could not initialize the governed gitlink {request.reference}: {exc}"
            )
        return r[m.Infra.GitBoolReport].ok(m.Infra.GitBoolReport(value=True))

    @classmethod
    def git_submodule_config_value(
        cls, request: m.Infra.GitSubmoduleConfigRequest
    ) -> p.Result[m.Infra.GitTextReport]:
        """Read one ``.gitmodules`` value, returning empty text when unset."""
        try:
            repo = cls._repo(request.repo_root)
            # The declaration file is addressed absolutely: the resolved Git
            # handle may sit in an enclosing repository whose own working tree
            # carries no (or another) ``.gitmodules``.
            value = repo.git.config(
                "-f",
                str(request.repo_root / c.Infra.GITMODULES),
                "--get",
                "--default",
                "",
                f"{request.section}.{request.key}",
            )
        except (GitCommandError, OSError, ValueError) as exc:
            return r[m.Infra.GitTextReport].fail(
                f"failed to read {request.section}.{request.key}: {exc}"
            )
        return r[m.Infra.GitTextReport].ok(m.Infra.GitTextReport(text=value.strip()))

    @classmethod
    def git_submodule_sections(
        cls, request: m.Infra.GitRepoRequest
    ) -> p.Result[t.StrMapping]:
        """Map every declared submodule path to its ``.gitmodules`` section.

        A duplicated path is a declaration defect rather than a state defect, so
        it fails here instead of silently resolving to the last writer.
        """
        gitmodules = request.repo_root / c.Infra.GITMODULES
        if not gitmodules.is_file():
            return r[t.StrMapping].ok({})
        try:
            repo = cls._repo(request.repo_root)
            listed = repo.git.config(
                "-f",
                str(gitmodules),
                "--name-only",
                "--get-regexp",
                r"^submodule\..*\.path$",
            )
        except (GitCommandError, OSError, ValueError) as exc:
            return r[t.StrMapping].fail(f"failed to read submodule declarations: {exc}")
        sections: dict[str, str] = {}
        for key in listed.split():
            section = key.removesuffix(".path")
            value = cls.git_submodule_config_value(
                m.Infra.GitSubmoduleConfigRequest(
                    repo_root=request.repo_root, section=section, key="path"
                )
            )
            if value.failure:
                return r[t.StrMapping].fail(
                    value.error or f"failed to read {section}.path"
                )
            declared = value.value.text
            if not declared:
                continue
            if declared in sections:
                return r[t.StrMapping].fail(
                    f"governed gitlink path is duplicated: {declared}"
                )
            sections[declared] = section
        return r[t.StrMapping].ok(sections)


__all__: list[str] = ["FlextInfraUtilitiesGitSemanticSubmoduleMixin"]
