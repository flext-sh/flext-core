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

if TYPE_CHECKING:
    # NOTE (multi-agent, mro-wkii.17.26): the public t facade is still being
    # composed when this protocol module loads; it is needed only by annotations.
    from pathlib import Path

    from flext_core import FlextTypes as t

    from .result import FlextProtocolsResult as pr


class FlextProtocolsConfig:
    """Protocols for declarative config loading and env override."""

    @runtime_checkable
    class ConfigLoader(Protocol):
        """Structural contract for loading a config source into a mapping."""

        def config_load(self, path: Path) -> pr.Result[t.JsonMapping]:
            """Load and parse a config source into a validated mapping."""
            ...

        def config_merge(
            self, base: t.JsonMapping, override: t.JsonMapping
        ) -> t.JsonMapping:
            """Deep-merge ``override`` onto ``base``, returning a new mapping."""
            ...

    # mro-qc84 (fix-forward): protocol-of-model for a loaded config document
    # (m.ConfigDocument). Consumed at runtime by the flext-cli config loader
    # result contracts (r[p.ConfigDocument]).
    @runtime_checkable
    class ConfigDocument(Protocol):
        """A loaded, parsed config document with optional schema/source refs."""

        @property
        def data(self) -> t.JsonMapping: ...

        @property
        def source_path(self) -> str | None: ...

        @property
        def schema_ref(self) -> str | None: ...


__all__: list[str] = ["FlextProtocolsConfig"]
