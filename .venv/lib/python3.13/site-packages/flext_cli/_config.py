"""FlextCliConfig — frozen, validated config singleton for flext-cli (§2.0b).

Every ``config/*.yaml`` file is auto-discovered and deep-merged at first
``fetch_global`` call (model-less, ``extra=allow`` at the FlextConfig base).
The flat YAML is then validated into the pure-Pydantic ``_models.config``
shapes and exposed as typed domain objects (``config.Cli.name`` /
``config.Cli.version``) — never a model-less dict subscript (§2.2/§2.4).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_cli._models.config import FlextCliConfigModels
from flext_core import FlextConfig

if TYPE_CHECKING:
    # NOTE (multi-agent): accessor typed by PROTOCOL (p), never the model
    # class; the protocol module enters under TYPE_CHECKING only (§2.5/§3.4).
    from flext_cli._protocols.config import FlextCliProtocolsConfig


class FlextCliConfig(FlextConfig):
    """Cli config auto-loaded from ``config/*.yaml`` and validated via models."""

    # NOTE (multi-agent): anchored to the package dir so the YAML SSOT loads
    # regardless of the caller's CWD (library code must not depend on CWD).
    CONFIG_DIR: ClassVar[str] = str(Path(__file__).resolve().parent / "config")

    @cached_property
    def Cli(self) -> FlextCliProtocolsConfig.Cli:
        """Validated ``Cli`` config domain (name/version identity metadata)."""
        root = FlextCliConfigModels.Root.model_validate(dict(self.model_extra or {}))
        return root.Cli


config: FlextCliConfig = FlextCliConfig.fetch_global()
"""Pre-instantiated frozen config singleton — ``from flext_cli import config``."""

__all__: list[str] = ["FlextCliConfig", "config"]
