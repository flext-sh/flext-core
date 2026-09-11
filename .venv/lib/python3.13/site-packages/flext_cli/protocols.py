"""CLI protocol facade."""

from __future__ import annotations

from flext_cli._protocols.base import FlextCliProtocolsBase
from flext_cli._protocols.config import FlextCliProtocolsConfig
from flext_cli._protocols.domain import FlextCliProtocolsDomain
from flext_cli._protocols.framework import FlextCliProtocolsFramework
from flext_cli._protocols.pipeline import FlextCliProtocolsPipeline
from flext_cli._protocols.xlsx import FlextCliProtocolsXlsx
from flext_core import p as _core_p


class FlextCliProtocols(_core_p):
    """CLI protocol definitions extending FlextProtocols.

    CLI protocol refinements take precedence in MRO while ``Result`` and the
    other core protocol members remain inherited from ``FlextProtocols``.
    """

    class Cli(
        FlextCliProtocolsPipeline,
        FlextCliProtocolsDomain,
        FlextCliProtocolsFramework,
        FlextCliProtocolsBase,
        FlextCliProtocolsConfig,
        FlextCliProtocolsXlsx,
    ):
        """Unified CLI protocol namespace."""


# mro-j47u (codex): canonical facade rebinding must stay type-annotated — an
# unannotated alias makes Mypy treat the facade as the class itself, turning
# class-subscript annotations such as `p.Result[str]` into Any downstream.
p: type[FlextCliProtocols] = FlextCliProtocols  # canonical facade alias (annotated)

__all__: list[str] = ["FlextCliProtocols", "p"]
