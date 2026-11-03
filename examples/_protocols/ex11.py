# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex11
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from flext_core import FlextProtocols

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx11:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Payload(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str:
            """Payload text content."""
            ...

    @runtime_checkable
    class ServiceHandlerConfig(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        debug: bool
        trace: bool
        log_level: str
        logging_profile: Literal["console", "json"]
        timezone: str
        async_logging: bool

        @property
        def cache_dir(self) -> Path:
            """Consuming application's cache directory."""
            ...

        @property
        def work_dir(self) -> Path:
            """Consuming application's scratch/work directory."""
            ...

        @property
        def data_dir(self) -> Path:
            """Consuming application's durable data directory."""
            ...

        @property
        def state_dir(self) -> Path:
            """Consuming application's persistent state directory."""
            ...

        @property
        def config_dir(self) -> Path:
            """Consuming application's user configuration directory."""
            ...

        @property
        def runtime_dir(self) -> Path:
            """Consuming application's ephemeral runtime directory."""
            ...

        enabled: bool

    @runtime_checkable
    class ServiceHandlerLike(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        message_type: type[p.Value]

    @runtime_checkable
    class ProcessorProtocolGood(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing outcome status."""
            ...

    @runtime_checkable
    class ProcessorProtocolBad(FlextProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing failure status."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx11",
]
