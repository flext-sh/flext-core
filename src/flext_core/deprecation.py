"""FlextDeprecationWarning - Custom deprecation warning.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Custom deprecation warning for FLEXT framework components.
"""

from __future__ import annotations


class FlextDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for FLEXT framework components."""


__all__ = ["FlextDeprecationWarning"]
