"""Config-domain protocols part (composed into ``p.Cli`` via MRO).

Structural, field-level protocols for the validated config domains — never
model classes, never ``Any``/``object``. No runtime project imports; importable
by ``c``/``t``/``p``/``m``/``u`` without creating a cycle (foundation purity).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class FlextCliProtocolsConfig:
    """Config-domain protocol namespace (structural types; no project imports)."""

    @runtime_checkable
    class Cli(Protocol):
        """Structural surface of the validated ``Cli`` config domain."""

        @property
        def name(self) -> str: ...

        @property
        def version(self) -> str: ...


__all__: list[str] = ["FlextCliProtocolsConfig"]
