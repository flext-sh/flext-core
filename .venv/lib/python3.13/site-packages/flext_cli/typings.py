"""CLI type facade."""

from __future__ import annotations

from flext_cli._typings.base import FlextCliTypesBase
from flext_cli._typings.domain import FlextCliTypesDomain
from flext_cli._typings.pipeline import FlextCliTypesPipeline
from flext_cli._typings.xlsx import FlextCliTypesXlsx
from flext_core import t


class FlextCliTypes(t):
    """CLI type definitions extending flext-core FlextTypes via inheritance."""

    class Cli(
        FlextCliTypesPipeline, FlextCliTypesDomain, FlextCliTypesBase, FlextCliTypesXlsx
    ):
        """CLI types namespace for cross-project access."""


t = FlextCliTypes

__all__: list[str] = ["FlextCliTypes", "t"]
