"""FlextSettingsContext — context-scoped overrides storage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core import t


class FlextSettingsContext:
    """Mixin that carries the class-level override registry.

    Actual resolution (``for_context``) lives in the façade because it
    needs ``model_copy`` and ``fetch_global``, which are only available
    once ``BaseSettings`` is part of the MRO.  Registration, however,
    stays here so it is inherited by every settings class.
    """

    _context_overrides: ClassVar[t.ScopedScalarRegistry] = {}

    @classmethod
    def register_context_overrides(cls, context_id: str, **overrides: t.Scalar) -> None:
        """Register context-specific settings overrides.

        Registers overrides that will be automatically applied when using
        `for_context()` with the same context_id.
        """
        cls._context_overrides.setdefault(context_id, {}).update(overrides)


__all__: list[str] = ["FlextSettingsContext"]
