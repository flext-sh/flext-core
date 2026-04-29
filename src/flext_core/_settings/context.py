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
    once ``BaseSettings`` is part of the MRO.
    """

    _context_overrides: ClassVar[t.ScopedScalarRegistry] = {}


__all__: list[str] = ["FlextSettingsContext"]
