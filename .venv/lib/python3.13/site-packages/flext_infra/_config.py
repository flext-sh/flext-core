"""Typed, frozen config singleton for flext-infra (ADR-005/U18).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_cli.config import FlextCliConfig
from flext_infra._models.config import FlextInfraConfigModels


class _FlextInfraConfig(FlextCliConfig):
    """Declarative flext-infra config loaded and validated once."""

    # NOTE (multi-agent, flext-wkii.9 + flext-wkii.17 / agent: codex): direct
    # config.Infra is the only codegen information surface; no accessor method.
    # NOTE (flext-sltx): CONFIG_DIR stays the relative default so
    # flext-core FlextConfig._config_dir() resolves the packaged flext_infra/config
    # in a wheel install AND the repo-root config/ in an editable source checkout.
    # An absolute parents[2] value broke every git-dep/wheel consumer (config poison).
    CONFIG_DIR: ClassVar[str] = "config"
    Infra: FlextInfraConfigModels.Infra


config: _FlextInfraConfig = _FlextInfraConfig()
"""Pre-instantiated frozen config singleton — ``from flext_infra import config``."""

__all__: list[str] = ["config"]
