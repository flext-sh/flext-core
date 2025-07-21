"""Configuration patterns and aliases for backward compatibility.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.config.base import BaseSettings

# Alias for backward compatibility with existing code
BaseFlextSettings = BaseSettings

__all__ = ["BaseFlextSettings", "BaseSettings"]
