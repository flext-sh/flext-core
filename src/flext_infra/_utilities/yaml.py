"""YAML loading and config normalization helpers for infrastructure.

Centralizes YAML-related helpers previously defined as module-level
functions in ``flext_infra.core.skill_validator``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from yaml import safe_load

from flext_infra.constants import FlextInfraConstants as c
from flext_infra.typings import t


class FlextInfraUtilitiesYaml:
    """YAML loading and validation helpers.

    Usage via namespace::

        from flext_infra import u

        data = u.Infra.safe_load_yaml(path)
    """

    @staticmethod
    def safe_load_yaml(path: Path) -> Mapping[str, t.ContainerValue]:
        """Load YAML file safely, returning empty mapping on missing/invalid.

        Args:
            path: Path to the YAML file to load.

        Returns:
            Parsed YAML data as a mapping, or empty dict on missing/invalid.

        Raises:
            TypeError: If the parsed content is not a mapping.

        """
        raw = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        parsed: t.ContainerValue | None = safe_load(raw)
        if parsed is None:
            return {}
        if not isinstance(parsed, Mapping):
            msg = f"rules.yml must be a mapping: {path}"
            raise TypeError(msg)
        normalized: dict[str, t.ContainerValue] = dict(parsed.items())
        return normalized

    @staticmethod
    def normalize_string_list(value: t.ContainerValue, field: str) -> list[str]:
        """Validate and normalize a list[str] config field.

        Args:
            value: The container value to normalize.
            field: Field name for error messages.

        Returns:
            Normalized list of strings.

        Raises:
            TypeError: If value is not a list of strings.

        """
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    msg = f"{field} must be list[str]"
                    raise TypeError(msg)
                out.append(item)
            return out
        msg = f"{field} must be list[str]"
        raise TypeError(msg)


__all__ = ["FlextInfraUtilitiesYaml"]
