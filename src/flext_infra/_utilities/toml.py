"""TOML utility helpers for flext-infra.

Provides type-safe TOML mapping narrowing used across TOML I/O operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import u
from flext_infra import t


class FlextInfraUtilitiesToml:
    """TOML utility helpers — type-safe mapping narrowing.

    Usage::

        from flext_infra import u

        result = u.Infra.Toml.as_toml_mapping(value)
    """

    @staticmethod
    def as_toml_mapping(
        value: t.ContainerValue,
    ) -> t.Infra.ContainerDict | None:
        """Check if value is a MutableMapping and return it typed, otherwise None."""
        if not isinstance(value, dict):
            return None
        for item in value.values():
            if not u.is_general_value_type(item):
                return None
        result: t.Infra.ContainerDict = value
        return result


__all__ = ["FlextInfraUtilitiesToml"]
