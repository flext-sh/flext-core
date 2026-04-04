"""FlextTypesValidation - constrained validation aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextTypesValidation:
    """Portable validation type aliases with annotated-types constraints.

    All types use ``annotated-types`` (Gt, Ge, Le, Len) for framework-independent
    constraints that Pydantic v2 and other checkers understand natively.
    """
