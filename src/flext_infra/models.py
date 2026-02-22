"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextInfraModels:
    """Namespace for infrastructure domain models.

    Provides data models and domain entities for all infrastructure services
    including base.mk configuration, check results, dependency information,
    and workspace state.

    Usage:
        >>> from flext_infra import m
        >>> # Access models via m.ServiceName.ModelName
    """

    pass


m = FlextInfraModels

__all__ = ["FlextInfraModels", "m"]
