"""Handler state models - Pydantic v2, state-only surface.

Only fields, validators, and computed properties that are consumed by
``src/`` (handlers, registry, utilities). Orchestration mutations live in
``FlextUtilitiesHandler``. All helper methods, dead factories, dict-style
accessors, and redundant wrapper classes have been removed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._handler_parts.flextmodelshandler_part_02 import (
    FlextModelsHandler as FlextModelsHandlerPartFinal,
)


class FlextModelsHandler(FlextModelsHandlerPartFinal):
    """Public facade for FlextModelsHandler."""


__all__: list[str] = ["FlextModelsHandler"]
