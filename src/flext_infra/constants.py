"""Constants for flext-infra.

Defines configuration constants and enumerations for infrastructure services
including validation rules, check types, and workspace settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextInfraConstants:
    """Namespace for infrastructure constants.

    Provides configuration constants and enumerations for all infrastructure
    services including base.mk templates, check types, dependency rules,
    and workspace orchestration settings.

    Usage:
        >>> from flext_infra import c
        >>> # Access constants via c.ServiceName.CONSTANT_NAME
    """

    pass


c = FlextInfraConstants

__all__ = ["FlextInfraConstants", "c"]
