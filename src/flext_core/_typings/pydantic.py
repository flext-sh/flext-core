"""Pydantic v2 type annotation helpers exported via FlextTypes.

This module provides public aliases for pydantic v2 annotation/discriminator
helpers that are used in type aliases across the flext ecosystem. All projects
consuming these must import from flext_core.t.* instead of directly from pydantic.

Architecture: Abstraction boundary - typings layer
Boundary: flext-core is sole owner of pydantic v2 integration. All other
projects receive pydantic type helpers ONLY through public facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import Discriminator


class FlextTypesPydantic:
    """Public type annotation helpers from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use these aliases via t.* instead: t.Discriminator

    Available helpers (accessible as t.HELPER_NAME):
        Discriminator: Function for discriminated union annotations in PEP 695 type aliases
    """

    # Public Pydantic v2 annotation helpers available via t.*
    Discriminator = Discriminator
