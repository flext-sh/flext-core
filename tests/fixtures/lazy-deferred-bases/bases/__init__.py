"""Initializer that publishes ``Base`` as a deferred stand-in.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports

install_lazy_exports(
    __name__,
    globals(),
    {"Base": ".base"},
    deferred_members={"Base": ("VALUE", "twice")},
)
