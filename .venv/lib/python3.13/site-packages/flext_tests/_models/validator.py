"""Models extraction for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Annotated, ClassVar

from flext_infra import m, p, u
from flext_tests import c, t


class FlextTestsValidatorModelsMixin:
    class EnforcementBuildContext(m.ArbitraryTypesModel):
        """Validated immutable inputs shared by enforcement item builders."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        infra_report: Annotated[
            p.AttributeProbe | None,
            u.Field(description="Optional namespace report for detector rules."),
        ] = None
        validator_targets: Annotated[
            tuple[Path, ...],
            u.Field(description="Validator targets collected for this session."),
        ] = ()
        workspace_root: Annotated[
            Path | None, u.Field(description="Resolved FLEXT workspace root.")
        ] = None

    class Violation(m.Value):
        """A detected architecture violation."""

        file_path: Annotated[
            Path, u.Field(description="Path to the offending source file.")
        ]
        line_number: Annotated[
            int, u.Field(description="1-based line number of the violation.")
        ]
        rule_id: Annotated[
            str, u.Field(description="Stable identifier for the rule that fired.")
        ]
        severity: Annotated[
            c.Tests.ValidatorSeverity,
            u.Field(description="Severity level assigned by the rule."),
        ]
        description: Annotated[
            str, u.Field(description="Human-readable violation description.")
        ]
        code_snippet: Annotated[
            str, u.Field(description="Source excerpt surrounding the violation.")
        ] = ""

        @u.field_validator("severity", mode="before")
        @classmethod
        def _coerce_severity(
            cls, value: c.Tests.ValidatorSeverity | str
        ) -> c.Tests.ValidatorSeverity:
            if isinstance(value, c.Tests.ValidatorSeverity):
                return value
            return c.Tests.ValidatorSeverity(value.upper())

    class ScanResult(m.Value):
        """Result of a validation scan."""

        validator_name: Annotated[
            str,
            u.Field(description="Identifier of the validator that produced the scan."),
        ]
        files_scanned: Annotated[
            int,
            u.Field(description="Count of source files inspected by the validator."),
        ]
        violations: Annotated[
            t.SequenceOf[FlextTestsValidatorModelsMixin.Violation],
            u.Field(description="All violations detected during the scan."),
        ]

        @u.computed_field
        @property
        def passed(self) -> bool:
            """True when the scan found no violations."""
            return not self.violations

    class EnforcementDispatcherConfig(m.Value):
        """Resolved runtime configuration for the pytest enforcement dispatcher."""

        active: Annotated[
            bool,
            u.Field(description="Whether the dispatcher is active for this session."),
        ]
        strict: Annotated[
            bool, u.Field(description="Promote runtime warnings to failures when true.")
        ]
        include: Annotated[
            frozenset[str],
            u.Field(description="Optional allow-list of enforcement rule IDs."),
        ] = frozenset()
        exclude: Annotated[
            frozenset[str],
            u.Field(description="Optional block-list of enforcement rule IDs."),
        ] = frozenset()
        workspace_root: Annotated[
            Path | None,
            u.Field(description="Resolved FLEXT workspace root for the session."),
        ] = None
        warning_counter: Annotated[
            MutableMapping[str, int],
            u.Field(
                description="Captured runtime warning counts keyed by dotted category."
            ),
        ] = u.Field(default_factory=dict)
