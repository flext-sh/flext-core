"""Template-engine models (ADR-005) — generic folder render entry/report.

These models are policy-free: the engine consumes ``TemplateRenderEntry`` items and
emits a ``TemplateRenderReport``. FLEXT naming/layout policy lives in the caller
(e.g. flext-infra codegen), never in the engine.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from flext_core import m


class FlextCliModelsTemplate:
    """Generic template-engine models, flat in ``m.Cli.*``."""

    class TemplateRenderEntry(m.ArbitraryTypesModel):
        """One template file to render into a mirrored output path."""

        relpath_template: Annotated[
            Path,
            m.Field(
                description="Template path relative to templates_root (ends with .j2)"
            ),
        ]
        output_relpath: Annotated[
            Path,
            m.Field(
                description=(
                    "Output path relative to output_root; a trailing template "
                    "suffix is stripped by the engine. Path tokens are resolved by "
                    "the caller before invocation."
                )
            ),
        ]
        when: Annotated[
            bool,
            m.Field(default=True, description="Render only when True (else skipped)"),
        ] = True
        overwrite: Annotated[
            bool,
            m.Field(
                default=False, description="Overwrite an existing destination file"
            ),
        ] = False

    class TemplateRenderReport(m.ArbitraryTypesModel):
        """Outcome of a generic folder render."""

        # NOTE (multi-agent, mro-wkii.17 / agent: make_ssot_audit): callers
        # consume the declared failure field directly; models remain behavior-free.
        created: Annotated[
            tuple[Path, ...],
            m.Field(default_factory=tuple, description="Destination paths written"),
        ]
        skipped: Annotated[
            tuple[Path, ...],
            m.Field(
                default_factory=tuple,
                description="Destinations skipped (exists or disabled)",
            ),
        ]
        failed: Annotated[
            tuple[tuple[Path, str], ...],
            m.Field(
                default_factory=tuple,
                description="(destination, error) pairs that failed to render",
            ),
        ]


__all__: list[str] = ["FlextCliModelsTemplate"]
