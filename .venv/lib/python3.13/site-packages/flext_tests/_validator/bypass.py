"""Bypass validation for FLEXT architecture.

Detects bypass patterns: noqa comments, pragma: no cover, exception swallowing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from typing import override

from flext_tests import c, m, t, u


class FlextValidatorBypass(u.Tests.ValidatorScannerMixin):
    """Bypass validation methods for FlextTestsValidator.

    Uses c.Tests.Validator for constants and m.Tests.Validator for models.
    """

    _VALIDATOR_KEY = c.Tests.VALIDATOR_BYPASS_KEY

    @classmethod
    def _check_exception_swallowing(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect exception swallowing patterns (bare except or except with pass)."""
        if u.Tests.approved("BYPASS-003", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_BARE_EXCEPT_RE.match(line) is not None:
                violations.append(
                    u.Tests.create_violation(
                        file_path,
                        line_number,
                        "BYPASS-003",
                        lines,
                        c.Tests.VALIDATOR_MSG_BYPASS_BARE_EXCEPT,
                    )
                )
                continue
            if c.Tests.VALIDATOR_EXCEPT_HEADER_RE.match(line) is None:
                continue
            if not u.Tests.except_block_only_pass(lines, line_number):
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "BYPASS-003",
                    lines,
                    c.Tests.VALIDATOR_MSG_BYPASS_TRIVIAL_BODY,
                )
            )
        return violations

    @classmethod
    def _check_noqa(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect # noqa comments."""
        if u.Tests.approved("BYPASS-001", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for i, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_NOQA_RE.search(line) and u.Tests.real_comment(
                line, c.Tests.VALIDATOR_NOQA_RE
            ):
                violation = u.Tests.create_violation(file_path, i, "BYPASS-001", lines)
                violations.append(violation)
        return violations

    @classmethod
    def _check_pragma_no_cover(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect # pragma: no cover comments."""
        if u.Tests.approved(
            "BYPASS-002",
            file_path,
            approved,
            c.Tests.VALIDATOR_APPROVED_PRAGMA_PATTERNS,
        ):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for i, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_PRAGMA_NO_COVER_RE.search(
                line
            ) and u.Tests.real_comment(line, c.Tests.VALIDATOR_PRAGMA_NO_COVER_RE):
                violation = u.Tests.create_violation(file_path, i, "BYPASS-002", lines)
                violations.append(violation)
        return violations

    @override
    @classmethod
    def _scan_file(
        cls, file_path: Path, approved: t.MappingKV[str, t.StrSequence]
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Scan a single file for bypass violations."""
        violations: MutableSequence[m.Tests.Violation] = []
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [
                u.Tests.create_violation(
                    file_path,
                    0,
                    "BYPASS-UNREADABLE",
                    (),
                    read.error or "could not read file",
                )
            ]
        lines = read.value.splitlines()
        violations.extend(cls._check_noqa(file_path, lines, approved))
        violations.extend(cls._check_pragma_no_cover(file_path, lines, approved))
        violations.extend(cls._check_exception_swallowing(file_path, lines, approved))
        return violations


__all__: list[str] = ["FlextValidatorBypass"]
