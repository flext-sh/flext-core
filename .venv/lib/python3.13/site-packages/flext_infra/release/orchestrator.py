"""Release orchestration service: one repository, one phase, one typed result."""

from __future__ import annotations

from typing import Annotated

from flext_infra import c, m
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.release._orchestrator_dispatch import (
    FlextInfraReleaseOrchestratorDispatchMixin,
)
from flext_infra.release.orchestrator_phases import FlextInfraReleaseOrchestratorPhases


class FlextInfraReleaseOrchestrator(
    FlextInfraReleaseOrchestratorDispatchMixin,
    FlextInfraReleaseOrchestratorPhases,
    FlextInfraProjectSelectionServiceBase[bool],
):
    """Run one phase of the release protocol against the workspace root.

    The version lives only in ``pyproject.toml``; the protocol derives every
    change from merged pull-request titles and is the sole writer.
    """

    phase: Annotated[
        c.Infra.ReleasePhase, m.Field(description="Release phase to execute")
    ] = c.Infra.ReleasePhase.PLAN
    index: Annotated[
        bool,
        m.Field(description="Publish receipt-verified artifacts to the package index"),
    ] = False
    pr_title: Annotated[
        str, m.Field(description="Pull-request title to validate against the protocol")
    ] = ""


__all__: list[str] = ["FlextInfraReleaseOrchestrator"]
