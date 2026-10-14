"""Base facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextBase:
    """Base facade for flext-core.

    All flext-core namespaces (``c``, ``t``, ``p``, ``m``, ``u`` and the
    operational ``r``, ``e``, ``x``, ``h``, ``d``, ``s``) compose through
    this root, so every shared behavior added here is inherited by the whole
    facade tree via MRO.
    """


__all__: list[str] = ["FlextBase"]
