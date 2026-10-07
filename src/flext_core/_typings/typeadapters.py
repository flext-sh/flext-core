"""Centralized tp.TypeAdapter cache for FLEXT typing aliases.

Each adapter is a `@classmethod @cache` that returns the canonical
``tp.TypeAdapter[...]``. The ``functools.cache`` wrapper memoizes per-cls,
so each adapter is constructed exactly once across the process — the
same caching contract the previous ClassVar pattern provided, with
~6 LOC eliminated per adapter (no per-adapter ``ClassVar`` slot, no
``if cls._x is None: cls._x = …`` boilerplate).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._typings._typeadapters_parts import (
    FlextTypesTypeAdapterJson,
    FlextTypesTypeAdapterScalars,
)


class FlextTypesTypeAdapters(FlextTypesTypeAdapterJson, FlextTypesTypeAdapterScalars):
    """Cached ``FlextTypesPydantic.TypeAdapter`` factories.

    JSON-shape factories live in ``FlextTypesTypeAdapterJson`` and
    scalar/binary factories in ``FlextTypesTypeAdapterScalars``; this
    class composes both so the public ``FlextTypingBase`` surface stays
    unchanged.
    """


__all__: list[str] = ["FlextTypesTypeAdapters"]
