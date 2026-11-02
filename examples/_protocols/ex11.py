# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
examples/_protocols/ex11
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import FlextProtocols, FlextSettings

if TYPE_CHECKING:
    from flext_core import p


class ExamplesFlextProtocolsEx11:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Payload(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str:
            """Payload text content."""
            ...

    @runtime_checkable
    class ServiceHandlerConfig(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def debug(self) -> bool:
            """Enable debug mode."""
            ...

        @property
        def trace(self) -> bool:
            """Enable detailed stacks and high-volume diagnostics."""
            ...

        log_level: str
        logging_profile: Literal[console, json]

        @property
        def timezone(self) -> str:
            """IANA timezone for datetime operations."""
            ...

        @property
        def async_logging(self) -> bool:
            """Enable asynchronous buffered logging."""
            ...

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

        @property
        def enabled(self) -> bool:
            """Whether the service is enabled."""
            ...

        @staticmethod
        def app_env_prefix(namespace: str) -> str:
            """Return the environment prefix owned by one application namespace."""

        @staticmethod
        def resolve_env_file(namespace: str | None = None) -> str:
            """Resolve the effective ``.env`` path honouring ``FLEXT_ENV_FILE``."""

        @classmethod
        def fetch_global(
            cls, *, overrides: Mapping[str, JsonValue] | None = None
        ) -> Self:
            """Return the shared per-class singleton (lazy, thread-safe)."""

        def clone(
            self,
            **overrides: BaseModel
            | JsonValue
            | Path
            | datetime
            | bytes
            | SecretStr
            | SecretBytes
            | Mapping[str, JsonValue]
            | tuple[JsonValue, ...]
            | FlextSettings._ModelValue
            | Sequence[JsonValue],
        ) -> Self:
            """Deep copy with optional field overrides + re-validation."""

        @classmethod
        def update_global(
            cls,
            **overrides: BaseModel
            | JsonValue
            | Path
            | datetime
            | bytes
            | SecretStr
            | SecretBytes
            | Mapping[str, JsonValue]
            | tuple[JsonValue, ...]
            | FlextSettings._ModelValue
            | Sequence[JsonValue],
        ) -> Self:
            """Validate an isolated candidate, then publish into the stable root."""

        @classmethod
        def reset_for_testing(cls) -> None:
            """Reload sources into a published root; leave an unused class lazy."""

        @staticmethod
        def platform_config_root() -> Path:
            """Structural member projected from the owning model.

            Return the OS-native config root shared with configuration loading.
            """

        @classmethod
        def apply_app_namespace(cls, namespace: str) -> None:
            """Set the outer application identity once for the current process."""

        @classmethod
        def reset_app_namespace(cls) -> None:
            """Clear the process application identity for isolated tests."""

    @runtime_checkable
    class ServiceHandlerLike(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message_type(self) -> type[p.Value]:
            """Message type handled by this handler."""
            ...

        @staticmethod
        def handle(message: p.Examples.Payload) -> p.Result[str]:
            """Structural member projected from the owning model."""

    @runtime_checkable
    class ProcessorProtocolGood(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing outcome status."""
            ...

    @runtime_checkable
    class ProcessorProtocolBad(FlextProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def status(self) -> str:
            """Processing failure status."""
            ...


__all__: list[str] = [
    "ExamplesFlextProtocolsEx11",
]
