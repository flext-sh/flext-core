from __future__ import annotations

from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure
from flext_core._utilities.guards_validation import FlextUtilitiesGuardsValidation


class FlextUtilitiesGuards(
    FlextUtilitiesGuardsEnsure,
    FlextUtilitiesGuardsValidation,
):
    pass


__all__ = ["FlextUtilitiesGuards"]
