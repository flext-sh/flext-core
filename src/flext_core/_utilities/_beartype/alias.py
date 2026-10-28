"""Extracted facade class for alias.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._beartype._alias import FlextUtilitiesBeartypeAliasVisitor


class FlextBeartypeAlias(
    FlextUtilitiesBeartypeAliasVisitor,
):
    """Facade for alias."""


__all__: list[str] = ["FlextBeartypeAlias"]
