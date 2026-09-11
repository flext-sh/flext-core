"""FlextTestsConfig — frozen config singleton for flext-tests (ADR-005 §7).

Model-less: business rules live in ``config/*.yaml`` under the ``Tests:`` key and
are exposed through the open ``config.Tests`` namespace (``extra="allow"``), with
no per-domain model. Access is ``config.Tests.<domain>[<key>...]``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_cli import FlextCliConfig, m


class _TestsNamespace(m.BaseModel):
    """Open, frozen namespace exposing every ``config/*.yaml`` domain model-less."""

    model_config = m.ConfigDict(extra="allow", frozen=True)


class FlextTestsConfig(FlextCliConfig):
    """Tests config auto-loaded model-less from ``config/*.yaml``."""

    Tests: _TestsNamespace = _TestsNamespace()


config: FlextTestsConfig = FlextTestsConfig.fetch_global()
"""Pre-instantiated frozen config singleton — ``from flext_tests import config``."""

__all__: list[str] = ["FlextTestsConfig", "config"]
