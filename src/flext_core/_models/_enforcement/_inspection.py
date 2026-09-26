"""Rule-level accounting for source-proven alias deferrals."""

from __future__ import annotations

from ._base import EnforcementModelBase
from ._resolution import FlextModelsEnforcementResolution


class FlextModelsEnforcementInspection(FlextModelsEnforcementResolution):
    """Bind rule context to an already-defined alias resolution contract."""

    class DeferredInspection(EnforcementModelBase):
        """A rule whose required alias value belongs to static type checking."""

        tag: str
        location: str
        alias: FlextModelsEnforcementResolution.DeferredAlias
