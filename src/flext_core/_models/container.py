"""Container models - Dependency Injection registry models.

TIER 0.5: Uses only stdlib + pydantic + models/metadata.py
(avoids cycles via __init__.py).

This module contains Pydantic models for FlextContainer that implement
ServiceRegistry and FactoryProvider Protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._container_parts.flextmodelscontainer_part_04 import (
    FlextModelsContainer as FlextModelsContainerPartFinal,
)


class FlextModelsContainer(FlextModelsContainerPartFinal):
    """Public facade for FlextModelsContainer."""


__all__: list[str] = ["FlextModelsContainer"]
