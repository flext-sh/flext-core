"""Pydantic v2 data models for enforcement violation reporting.

Typed containers (``m.Enforcement.*``) that enforcement checks return
instead of raw string lists. This eliminates ``t.StrSequence`` conversions
at call sites and lets callers narrow by severity / layer / qualname
without reparsing formatted messages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from typing import Annotated, ClassVar

from pydantic import Field

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t


class FlextModelsEnforcement:
    """Namespace for enforcement violation data models.

    Exposed via MRO as ``m.Enforcement`` (the nested class below). Do not
    access ``FlextModelsEnforcement.Violation`` directly from user code —
    always go through ``m.Enforcement.Violation`` / ``m.Enforcement.Report``.
    """

    class Violation(mp.BaseModel):
        """Single enforcement violation located at ``qualname``."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        qualname: Annotated[
            str,
            Field(description="Qualified name of the violating class/attribute."),
        ]
        layer: Annotated[
            str,
            Field(description="Layer where the violation was detected."),
        ]
        severity: Annotated[
            str,
            Field(description='Severity label (e.g. "HARD rules", "best practices").'),
        ]
        message: Annotated[
            str,
            Field(description="Human-readable violation message."),
        ]

    class Report(mp.BaseModel):
        """Aggregated violation report returned by a check or runner."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True, extra="forbid"
        )

        violations: Annotated[
            Sequence[FlextModelsEnforcement.Violation],
            Field(
                default_factory=list,
                description="Violations detected by the check.",
            ),
        ] = Field(default_factory=tuple)

        @property
        def messages(self) -> t.StrSequence:
            """Return plain messages for backwards-compatible emission."""
            return [v.message for v in self.violations]

        @property
        def empty(self) -> bool:
            """True when no violations were recorded."""
            return not self.violations

        def __len__(self) -> int:
            """Expose violation count for ``len(report)``."""
            return len(self.violations)

        def __bool__(self) -> bool:
            """Truthy when violations exist."""
            return bool(self.violations)

        def __getitem__(self, index: int) -> str:
            """Return the nth message for ``report[i]`` access."""
            return self.messages[index]

        def __contains__(self, fragment: object) -> bool:
            """``"Any" in report`` — search message text."""
            if not isinstance(fragment, str):
                return False
            return any(fragment in m for m in self.messages)


__all__: list[str] = ["FlextModelsEnforcement"]
