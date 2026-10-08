"""Centralized tp.TypeAdapter cache for FLEXT typing aliases.

The composed adapter surface lives here; the JSON-shape factories live in
``typeadapter_json.py`` and the scalar/binary factories in
``typeadapter_scalars.py`` — one top-level class per module (NS-000).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings.typeadapter_json import FlextTypesTypeAdapterJson
from flext_core._typings.typeadapter_scalars import FlextTypesTypeAdapterScalars


class FlextTypesTypeAdapters(FlextTypesTypeAdapterJson, FlextTypesTypeAdapterScalars):
    """Cached ``FlextTypesPydantic.TypeAdapter`` factories.

    JSON-shape factories live in ``FlextTypesTypeAdapterJson`` and
    scalar/binary factories in ``FlextTypesTypeAdapterScalars``; this
    class composes both so the public ``FlextTypingBase`` surface stays
    unchanged.
    """


__all__: list[str] = ["FlextTypesTypeAdapters"]
