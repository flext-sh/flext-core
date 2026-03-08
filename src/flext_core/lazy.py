"""Public re-export of lazy import utilities.

This module provides a public interface to the lazy import utilities,
avoiding PLC2701 (private name import) lint errors in downstream projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

__all__ = ["cleanup_submodule_namespace", "lazy_getattr"]
