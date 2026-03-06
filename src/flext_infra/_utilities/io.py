"""I/O helper functions for infrastructure file operations.

Centralizes convenience JSON/TOML I/O helpers previously defined as
module-level functions in ``flext_infra.json_io`` and ``flext_infra.toml_io``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path

from pydantic import BaseModel

from flext_core import r, t
from flext_infra.json_io import FlextInfraJsonService


class FlextInfraUtilitiesIo:
    """I/O convenience helpers for JSON/TOML file operations.

    Usage via namespace::

        from flext_infra import u

        result = u.Infra.Io.read_json(path)
    """

    @staticmethod
    def read_json(path: Path) -> r[Mapping[str, t.ContainerValue]]:
        """Read and parse a JSON file (convenience function).

        Args:
            path: Source file path.

        Returns:
            r with parsed JSON data. Returns empty mapping
            if the file does not exist.

        Example:
            >>> result = u.Infra.Io.read_json(Path("config.json"))
            >>> if result.is_success:
            ...     data = result.value

        """
        return FlextInfraJsonService().read(path)

    @staticmethod
    def write_json(
        path: Path,
        payload: BaseModel | t.ConfigurationMapping,
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> r[bool]:
        """Write a JSON payload to a file (convenience function).

        Creates parent directories as needed.

        Args:
            path: Destination file path.
            payload: Data to serialize as JSON.
            sort_keys: If True, sort dictionary keys alphabetically.
            ensure_ascii: If True, escape non-ASCII characters.

        Returns:
            r[bool] with True on success.

        Example:
            >>> result = u.Infra.Io.write_json(Path("output.json"), {"key": "value"})
            >>> assert result.is_success

        """
        return FlextInfraJsonService().write(
            path, payload, sort_keys=sort_keys, ensure_ascii=ensure_ascii
        )

    @staticmethod
    def as_toml_mapping(
        value: object,
    ) -> MutableMapping[str, object] | None:
        """Check if value is a MutableMapping and return it, otherwise None.

        Helper for TOML serialization that detects nested table structures.

        Args:
            value: Value to check for mapping nature.

        Returns:
            The value as MutableMapping if it is one, otherwise None.

        """
        if isinstance(value, MutableMapping):
            return value
        return None


__all__ = ["FlextInfraUtilitiesIo"]
