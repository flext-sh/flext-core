"""Flext-cli config models (m facade; no project-specific imports).

Typed, frozen shapes for the ``config/*.yaml`` business-rule SSOT. This module
imports only the ``flext_core.m`` facade — the ``_config.py`` facade validates the
model-less YAML slices into these classes and exposes the ready objects under
``config.Cli``. Adding a new config domain = add a nested model here and a
validated field on ``Root`` (§2.0b reference: cosmos-main ``_models/config.py``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import m


class FlextCliConfigModels:
    """Namespace of typed flext-cli config models (m facade)."""

    class Cli(m.BaseModel):
        """CLI identity metadata from ``config/cli.yaml``."""

        model_config = m.ConfigDict(frozen=True, extra="forbid")

        name: str
        version: str

    class Root(m.BaseModel):
        """Root flext-cli runtime config validated from ``config/*.yaml``."""

        model_config = m.ConfigDict(frozen=True, extra="ignore")

        Cli: FlextCliConfigModels.Cli


__all__: list[str] = ["FlextCliConfigModels"]
