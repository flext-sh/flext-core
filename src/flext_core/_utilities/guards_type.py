from __future__ import annotations

from flext_core import (
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
)


class FlextUtilitiesGuardsType(
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
):
    pass


__all__ = ["FlextUtilitiesGuardsType"]
