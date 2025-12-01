"""Models for flext-core tests.

Provides TestModels, extending FlextTestModels with flext-core-specific models.
All generic test models come from flext_tests, only flext-core-specific additions here.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests.models import FlextTestModels


class TestModels(FlextTestModels):
    """Models for flext-core tests - extends FlextTestModels.

    Architecture: Extends FlextTestModels with flext-core-specific model definitions.
    All generic models from FlextTestModels are available through inheritance.

    Rules:
    - NEVER redeclare models from FlextTestModels
    - Only flext-core-specific models allowed
    - All generic models come from FlextTestModels
    """

    # Flext-core-specific model additions (if any)
    # All generic models (including Docker models) are inherited from FlextTestModels
    # Add flext-core-specific models here if needed


__all__ = ["TestModels"]
