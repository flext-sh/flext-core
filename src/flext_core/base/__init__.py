"""Base classes for FLEXT data integration components.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides unified base classes for taps, targets, and other
data integration components to eliminate duplication across projects.
"""

from __future__ import annotations

from flext_core.base.config_base import BaseComponentConfig
from flext_core.base.repository_base import BaseComponentRepository
from flext_core.base.tap_base import BaseTap
from flext_core.base.tap_base import TapMetrics
from flext_core.base.tap_base import ValidationResult
from flext_core.base.target_base import BaseTarget

__all__ = [
    "BaseComponentConfig",
    "BaseComponentRepository",
    "BaseTap",
    "BaseTarget",
    "TapMetrics",
    "ValidationResult",
]
