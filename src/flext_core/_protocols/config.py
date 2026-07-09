"""FlextProtocolsConfig - declarative config loader protocol (ADR-005).

Behavioral contract for a config loader. flext-core ships a minimal stdlib
implementation (``u.config_load`` / ``u.config_merge`` / ``u.config_env_override``);
``flext-cli`` provides the advanced multi-format loader that also satisfies this
protocol. Callers depend on the protocol, not the concrete class (DIP).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .result import FlextProtocolsResult as pr

if TYPE_CHECKING:
    from pathlib import Path

    from flext_core import FlextTypes as t


class FlextProtocolsConfig:
    """Protocols for declarative config loading and env override."""

    @runtime_checkable
    class ConfigLoader(Protocol):
        """Structural contract for loading a config source into a mapping."""

        def config_load(self, path: Path) -> pr.Result[t.JsonMapping]:
            """Load and parse a config source into a validated mapping."""
            ...

        def config_merge(
            self,
            base: t.JsonMapping,
            override: t.JsonMapping,
        ) -> t.JsonMapping:
            """Deep-merge ``override`` onto ``base``, returning a new mapping."""
            ...


__all__: list[str] = ["FlextProtocolsConfig"]
