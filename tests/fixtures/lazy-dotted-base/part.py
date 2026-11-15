"""Part extending the nested class of another package's export.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .outer import Outer


class Part(Outer.Inner):
    """Deferred part whose base is ``Outer.Inner``."""

    PART_VALUE = 3


__all__: list[str] = ["Part"]
