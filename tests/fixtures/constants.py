"""Test constants organized by domain namespace.

DEPRECATED: This file is deprecated. Use tests.helpers.constants.TestConstants instead.

This file re-exports TestConstants from tests.helpers.constants for backward compatibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests.helpers.constants import TestConstants

# Backward compatibility - re-export TestConstants
__all__ = ["TestConstants"]
