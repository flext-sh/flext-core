"""Dispatcher configuration types.

Use m.Config.DispatcherConfig at call sites (runtime alias m from project __init__).
No local alias; MRO protocol only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.settings import FlextModelsConfig

__all__ = ["FlextModelsConfig"]
