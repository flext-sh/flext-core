from __future__ import annotations

from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core._utilities.guards_type_protocol import FlextUtilitiesGuardsTypeProtocol


class FlextUtilitiesGuardsType(
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
):
    pass


__all__ = ["FlextUtilitiesGuardsType"]
