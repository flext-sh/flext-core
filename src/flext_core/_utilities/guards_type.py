"""Combined type guards aggregating core, model, and protocol type guards.

Consolidates type checking functions from FlextUtilitiesGuardsTypeCore,
FlextUtilitiesGuardsTypeModel, and FlextUtilitiesGuardsTypeProtocol into
a single unified namespace via multiple inheritance.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

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
    """Unified type guards combining core, model, and protocol checkers.

    Aggregates all type guard methods from three specialized classes into
    a single interface for convenient access to all type narrowing utilities.
    """

    pass


__all__ = ["FlextUtilitiesGuardsType"]
