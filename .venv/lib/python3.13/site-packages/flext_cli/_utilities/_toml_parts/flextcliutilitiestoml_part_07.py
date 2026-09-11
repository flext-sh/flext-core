"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_cli import c, e, p, r, t
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_01 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart01,
)
from flext_cli._utilities._toml_parts.flextcliutilitiestoml_part_06 import (
    FlextCliUtilitiesToml as FlextCliUtilitiesTomlPart06,
)

if TYPE_CHECKING:
    from pathlib import Path


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    @staticmethod
    def toml_write_mapping(path: Path, mapping: t.JsonMapping) -> p.Result[bool]:
        """Write one validated plain mapping as TOML through the canonical writer."""
        try:
            document = FlextCliUtilitiesTomlPart01.toml_document_from_mapping(mapping)
        except c.EXC_TYPE_VALIDATION as exc:
            return e.fail_validation("TOML build", error=exc, result_type=r[bool])
        return FlextCliUtilitiesTomlPart06.toml_write_document(path, document)


__all__: list[str] = ["FlextCliUtilitiesToml"]
