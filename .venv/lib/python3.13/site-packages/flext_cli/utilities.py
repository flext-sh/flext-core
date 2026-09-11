"""FLEXT CLI utility facade."""

from __future__ import annotations

from flext_cli._utilities._cli_namespace import FlextCliUtilitiesCli
from flext_core import u


class FlextCliUtilities(u):
    """CLI utility facade composed from internal utility mixins."""

    # NOTE (multi-agent): mro-wkii.17.17 publishes the canonical class directly.
    Cli: type[FlextCliUtilitiesCli] = FlextCliUtilitiesCli


u = FlextCliUtilities

__all__: list[str] = ["FlextCliUtilities", "u"]
