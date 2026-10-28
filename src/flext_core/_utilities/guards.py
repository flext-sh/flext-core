"""Extracted facade class for guards.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from flext_core._utilities._guards import (
    FlextUtilitiesGuardsTypeProtocolSpecs,
    FlextUtilitiesGuardsTypeProtocolString,
)


class FlextUtilitiesGuards(
    FlextUtilitiesGuardsTypeProtocolSpecs,
    FlextUtilitiesGuardsTypeProtocolString,
):
    """Facade for guards."""


__all__: list[str] = ["FlextUtilitiesGuards"]
