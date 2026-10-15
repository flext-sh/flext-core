"""Centralized tp.TypeAdapter cache for FLEXT typing aliases.

The composed adapter surface lives here; the JSON-shape factories live in
``typeadapter_json.py`` and the scalar/binary factories in
``typeadapter_scalars.py`` — one top-level class per module (NS-000).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities import (
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
)


class FlextTypesTypeAdapters(
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
):
    """Cached ``FlextTypesPydantic.TypeAdapter`` factories.

    JSON-shape factories live in ``FlextUtilitiesTypingAdapterJson`` and
    scalar/binary factories in ``FlextUtilitiesTypingAdapterScalars``; this
    class composes both so the public ``FlextTypingBase`` surface stays
    unchanged.
    """


__all__: list[str] = ["FlextTypesTypeAdapters"]
