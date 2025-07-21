"""DEPRECATED: LDIF Writer utilities - USE flext-ldif instead.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

⚠️  DEPRECATION WARNING: This module has been moved to flext-ldif package.
    Use 'from flext_ldif import LDIFWriter, FlextLDIFWriter' instead.
    This module will be removed in v0.8.0.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning immediately when module is imported
warnings.warn(
    "flext_core.utils.ldif_writer is deprecated. "
    "Use 'from flext_ldif import LDIFWriter, FlextLDIFWriter' instead. "
    "This module will be removed in v0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from flext_ldif import FlextLDIFWriter
    from flext_ldif import LDIFHierarchicalSorter
    from flext_ldif import LDIFWriter
except ImportError as e:
    msg = (
        "flext-ldif package not found. "
        "Install it with: pip install flext-ldif, "
        "or use the new import: 'from flext_ldif import FlextLDIFWriter'"
    )
    raise ImportError(
        msg,
    ) from e

# For backwards compatibility
__all__ = [
    "FlextLDIFWriter",
    "LDIFHierarchicalSorter",
    "LDIFWriter",
]
