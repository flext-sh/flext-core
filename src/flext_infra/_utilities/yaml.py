"""YAML loading and config normalization helpers for infrastructure.

Centralizes YAML-related helpers previously defined as module-level
functions in ``flext_infra.core.skill_validator``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError
from yaml import safe_load

from flext_infra.constants import FlextInfraConstants as c
from flext_infra.typings import t


class FlextInfraUtilitiesYaml:
    """YAML loading and validation helpers.

    Usage via namespace::

        from flext_infra import u

        data = u.Infra.safe_load_yaml(path)
    """

    _LIST_ADAPTER: TypeAdapter[list[JsonValue]] | None = None
    _MAPPING_ADAPTER: TypeAdapter[dict[str, t.Infra.InfraValue]] | None = None

    @staticmethod
    def _get_list_adapter() -> TypeAdapter[list[JsonValue]]:
        if FlextInfraUtilitiesYaml._LIST_ADAPTER is None:
            FlextInfraUtilitiesYaml._LIST_ADAPTER = TypeAdapter(list[JsonValue])
        return FlextInfraUtilitiesYaml._LIST_ADAPTER

    @staticmethod
    def _get_mapping_adapter() -> TypeAdapter[dict[str, t.Infra.InfraValue]]:
        if FlextInfraUtilitiesYaml._MAPPING_ADAPTER is None:
            FlextInfraUtilitiesYaml._MAPPING_ADAPTER = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            )
        return FlextInfraUtilitiesYaml._MAPPING_ADAPTER

    @staticmethod
    def safe_load_yaml(path: Path) -> Mapping[str, t.Infra.InfraValue]:
        """Load YAML file safely, returning empty mapping on missing/invalid.

        Args:
            path: Path to the YAML file to load.

        Returns:
            Parsed YAML data as a mapping, or empty dict on missing/invalid.

        Raises:
            TypeError: If the parsed content is not a mapping.

        """
        raw = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        parsed: t.Infra.InfraValue | None = safe_load(raw)
        if parsed is None:
            return {}
        if not isinstance(parsed, Mapping):
            msg = f"rules.yml must be a mapping: {path}"
            raise TypeError(msg)
        try:
            return FlextInfraUtilitiesYaml._get_mapping_adapter().validate_python(
                parsed
            )
        except ValidationError as exc:
            msg = f"rules.yml must be a mapping: {path}: {exc}"
            raise TypeError(msg) from exc

    @staticmethod
    def normalize_string_list(value: t.Infra.InfraValue, field: str) -> list[str]:
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
            try:
                typed_items = (
                    FlextInfraUtilitiesYaml._get_list_adapter().validate_python(value)
                )
            except ValidationError as exc:
                msg = f"{field} must be list[str]: {exc}"
                raise TypeError(msg) from exc
            out: list[str] = []
            for item in typed_items:
                if not isinstance(item, str):
                    msg = f"{field} must be list[str]"
                    raise TypeError(msg)
                out.append(item)
            return out
        msg = f"{field} must be list[str]"
        raise TypeError(msg)


__all__ = ["FlextInfraUtilitiesYaml"]
