"""Project-selection service base for flext-infra command services."""

from __future__ import annotations

from flext_infra import m, t
from flext_infra._base_projects import FlextInfraProjectSelectionMixin
from flext_infra.base import FlextInfraServiceBase

type _InfraResultValue = t.Cli.ResultValue


class FlextInfraProjectSelectionServiceBase[TDomainResult: _InfraResultValue](
    FlextInfraServiceBase[TDomainResult], FlextInfraProjectSelectionMixin
):
    """Shared service foundation for commands that target workspace projects."""

    selected_projects: t.StrSequence | None = m.Field(
        default=None, alias="projects", description="Projects to process"
    )


__all__: list[str] = ["FlextInfraProjectSelectionServiceBase"]
