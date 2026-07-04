"""Base models and report containers for enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t


class EnforcementModelBase(mp.BaseModel):
    """Frozen, extra-forbid base for internal enforcement models."""

    model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
        frozen=True,
        extra="forbid",
    )


class FlextModelsEnforcementBase:
    """Foundational enforcement models shared by catalog and predicates."""

    class Violation(EnforcementModelBase):
        """Single enforcement violation located at qualname."""

        qualname: str
        layer: str
        severity: str
        message: str
        rule_id: str = ""
        agents_md_anchor: str = ""
        file_path: str = ""
        line_number: int = 0

    class Report(EnforcementModelBase):
        """Aggregated violation report returned by a check or runner."""

        violations: t.SequenceOf[FlextModelsEnforcementBase.Violation] = ()

        @property
        def messages(self) -> t.StrSequence:
            """Plain messages for text emission."""
            return [violation.message for violation in self.violations]

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

        def __contains__(self, fragment: t.Scalar | None) -> bool:
            """Search message text with ``fragment in report``."""
            if not isinstance(fragment, str):
                return False
            return any(fragment in message for message in self.messages)

    @unique
    class EnforcementRuleSeverity(StrEnum):
        """Severity scale for catalog rules."""

        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    @unique
    class EnforcementSourceKind(StrEnum):
        """Addressable origin layer for a catalog rule."""

        FLEXT_INFRA_DETECTOR = "flext_infra_detector"
        FLEXT_TESTS_VALIDATOR = "flext_tests_validator"
        RUNTIME_WARNING = "runtime_warning"
        BEARTYPE = "beartype"
        CODE_SMELL = "code_smell"
        RUFF = "ruff"
        SKILL_POINTER = "skill_pointer"


__all__: list[str] = ["EnforcementModelBase", "FlextModelsEnforcementBase"]
