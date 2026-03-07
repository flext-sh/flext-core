"""TOML utility helpers for flext-infra.

Provides type-safe TOML mapping narrowing used across TOML I/O operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping

from flext_core import t, u
from flext_infra.typings import FlextInfraTypes


class FlextInfraUtilitiesToml:
    """TOML utility helpers — type-safe mapping narrowing.

    Usage::

        from flext_infra import u

        result = u.Infra.Toml.as_toml_mapping(value)
    """

    @staticmethod
    def as_toml_mapping(
        value: t.ContainerValue,
    ) -> FlextInfraTypes.Infra.ContainerMap | None:
        """Check if value is a MutableMapping and return it typed, otherwise None."""
        if not isinstance(value, MutableMapping):
            return None

        converted: FlextInfraTypes.Infra.ContainerMap = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return None
            if not u.is_general_value_type(item):
                return None
            converted[key] = item
        return converted


__all__ = ["FlextInfraUtilitiesToml"]
