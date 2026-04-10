"""Constants mixin for loggings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import c


class TestsFlextCoreConstantsLoggings:
    class Logging(c):
        """Logging configuration for tests - real inheritance."""

        ContextOperation = c.ContextOperation
