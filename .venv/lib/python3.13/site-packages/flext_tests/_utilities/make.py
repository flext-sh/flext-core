"""Composed Make command utilities for flext-tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests._utilities.make_rendering import FlextTestsMakeRenderingUtilitiesMixin


class FlextTestsMakeUtilitiesMixin(FlextTestsMakeRenderingUtilitiesMixin):
    """Generic registry-driven Make command utility namespace."""


__all__: list[str] = ["FlextTestsMakeUtilitiesMixin"]
