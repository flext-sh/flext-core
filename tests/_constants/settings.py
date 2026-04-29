"""Constants mixin for settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConstants as c


class TestsFlextConstantsSettings:
    class Settings(c):
        """Configuration defaults for tests."""

        LogLevel = c.LogLevel
        Environment = c.Environment
