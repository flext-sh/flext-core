"""Public Git orchestration service for flext-infra consumers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_infra import m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraGitService(s[m.Infra.GitStatusReport]):
    """Thin status-only orchestrator over ``u.Infra.git_status``."""

    repository: Annotated[
        Path | None, m.Field(description="Repository path; defaults to repository_root")
    ] = None

    def _repo(self) -> Path:
        """Resolve the single repository root for this invocation."""
        return (self.repository or self.repository_root).expanduser().resolve()

    @override
    def execute(self) -> p.Result[m.Infra.GitStatusReport]:
        """Capture porcelain status for the selected repository."""
        return u.Infra.git_status(m.Infra.GitStatusRequest(repo_root=self._repo()))


__all__: list[str] = ["FlextInfraGitService"]
