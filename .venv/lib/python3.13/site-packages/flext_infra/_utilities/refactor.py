"""Refactor helper utilities for infrastructure code analysis.

Centralizes rope-based helpers previously defined as module-level
functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_cli import r, u
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesRefactor:
    """Rope-based refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        methods = u.Infra.extract_public_methods_from_dir(package_dir)
    """

    @staticmethod
    def entry_list(value: t.Infra.InfraValue | None) -> t.SequenceOf[t.StrMapping]:
        """Normalize class-nesting settings entries to a strict list."""
        if value is None:
            return []
        try:
            entries: t.SequenceOf[t.StrMapping] = (
                t.Infra.STR_MAPPING_SEQ_ADAPTER.validate_python(value)
            )
        except c.ValidationError:
            msg = "class nesting entries must be a list"
            raise ValueError(msg) from None
        else:
            return entries

    @staticmethod
    def string_list(value: t.Infra.InfraValue | None) -> t.StrSequence:
        """Normalize policy fields that should contain string collections."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return list(t.Infra.STR_SEQ_ADAPTER.validate_python(value))
        except TypeError as exc:
            msg = "expected list value"
            raise TypeError(msg) from exc
        except c.ValidationError as exc:
            msg = "expected list value"
            raise TypeError(msg) from exc

    @staticmethod
    def normalize_module_path(path_value: str | Path) -> str:
        """Normalize module path."""
        path = Path(str(path_value).replace("\\", "/"))
        parts = path.parts
        if c.Infra.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.DEFAULT_SRC_DIR)
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    @staticmethod
    def write_impact_map(
        results: t.SequenceOf[m.Infra.Result], output_path: Path
    ) -> p.Result[bool]:
        """Write refactor impact map JSON to disk."""
        payload = {
            "files": [
                {
                    "path": str(item.file_path),
                    "success": item.success,
                    "modified": item.modified,
                    "error": item.error,
                    "changes": list(item.changes),
                }
                for item in results
            ]
        }
        normalized_payload: t.JsonValue = t.Cli.JSON_VALUE_ADAPTER.validate_python(
            payload
        )
        write_result = u.Cli.json_write(output_path, normalized_payload)
        if write_result.failure:
            return r[bool].fail(write_result.error or "impact map write failed")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraUtilitiesRefactor"]
