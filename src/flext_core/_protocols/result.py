"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols._result_parts.flextprotocolsresult_part_04 import (
    FlextProtocolsResult as FlextProtocolsResultPartFinal,
)


class FlextProtocolsResult(FlextProtocolsResultPartFinal):
    """Public facade for FlextProtocolsResult."""


__all__: list[str] = ["FlextProtocolsResult"]
