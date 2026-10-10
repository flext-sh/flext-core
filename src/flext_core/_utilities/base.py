"""Utilities base facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import TypeAdapter as PydanticTypeAdapter


class FlextUtilitiesBase:
    """Bootstrap-safe constructors inherited by the public utilities facade.

    Adapter construction preserves Pydantic's signature without importing
    models or runtime. Composed consumers use ``u.type_adapter``; lower
    bootstrap owners consume this same constructor before ``u`` exists.
    """

    # Runtime/model bootstrap consumers cannot depend on the composed facade.
    TypeAdapter = PydanticTypeAdapter
    type_adapter = TypeAdapter


__all__: list[str] = ["FlextUtilitiesBase"]
