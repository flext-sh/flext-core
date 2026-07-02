"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .flexthandlers_part_01 import (
    FlextHandlers as FlextHandlersPart01,
)


class FlextHandlers[MessageT_contra, ResultT](
    FlextHandlersPart01[MessageT_contra, ResultT],
):
    """Generated MRO anchor for handler parts."""


__all__: list[str] = ["FlextHandlers"]
