"""Test validation for FLEXT architecture.

Detects test violations: monkeypatch, mocks, @patch decorator usage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from typing import TYPE_CHECKING, override

from flext_tests import c, t, u

if TYPE_CHECKING:
    from flext_tests import m


class FlextValidatorTests(u.Tests.ValidatorScannerMixin):
    """Test validation methods for FlextTestsValidator.

    Uses c.Tests.Validator and u.Tests.ValidatorScannerMixin.
    """

    _VALIDATOR_KEY = c.Tests.VALIDATOR_TESTS_KEY

    @staticmethod
    def _function_signatures(lines: t.StrSequence) -> tuple[tuple[int, str, str], ...]:
        """Collect logical function signatures, including multiline parameter lists."""
        signatures: list[tuple[int, str, str]] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            match = c.Tests.VALIDATOR_FUNCTION_DEF_RE.match(line)
            if match is None:
                index += 1
                continue
            start_line = index + 1
            signature_lines = [line.strip()]
            balance = line.count("(") - line.count(")")
            while index + 1 < len(lines) and (balance > 0 or ":" not in line):
                index += 1
                line = lines[index]
                signature_lines.append(line.strip())
                balance += line.count("(") - line.count(")")
                if balance <= 0 and ":" in line:
                    break
            signatures.append((
                start_line,
                match.group("name"),
                " ".join(signature_lines),
            ))
            index += 1
        return tuple(signatures)

    @classmethod
    def _check_mock_usage(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect Mock and MagicMock usage."""
        if u.Tests.approved("TEST-002", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        mock_names = c.Tests.VALIDATOR_APPROVED_MOCK_NAMES
        for line_number, line in enumerate(lines, start=1):
            from_match = c.Tests.VALIDATOR_FROM_IMPORT_LINE_RE.match(line)
            if from_match is not None and u.Tests.code_match(
                line, c.Tests.VALIDATOR_FROM_IMPORT_LINE_RE
            ):
                module = from_match.group("module")
                if "mock" in module.lower():
                    for imported_name in u.Tests.split_import_targets(
                        from_match.group("names")
                    ):
                        if imported_name not in mock_names:
                            continue
                        violations.append(
                            u.Tests.create_violation(
                                file_path,
                                line_number,
                                "TEST-002",
                                lines,
                                f"import {imported_name}",
                            )
                        )
            for match in c.Tests.VALIDATOR_MOCK_CALL_RE.finditer(line):
                violations.append(
                    u.Tests.create_violation(
                        file_path,
                        line_number,
                        "TEST-002",
                        lines,
                        f"{match.group('name')}()",
                    )
                )
        return violations

    @classmethod
    def _check_monkeypatch(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect monkeypatch usage in function parameters and calls."""
        if u.Tests.approved("TEST-001", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, func_name, signature in cls._function_signatures(lines):
            if "monkeypatch" not in signature:
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "TEST-001",
                    lines,
                    c.Tests.VALIDATOR_MSG_TEST_MONKEYPATCH.format(func=func_name),
                )
            )
        for line_number, line in enumerate(lines, start=1):
            match = c.Tests.VALIDATOR_MONKEYPATCH_ACCESS_RE.search(line)
            if match is None or not u.Tests.code_match(
                line, c.Tests.VALIDATOR_MONKEYPATCH_ACCESS_RE
            ):
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "TEST-001",
                    lines,
                    f"monkeypatch.{match.group('attr')}",
                )
            )
        return violations

    @classmethod
    def _check_patch_decorator(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect @patch decorator usage."""
        if u.Tests.approved("TEST-003", file_path, approved):
            return []
        return [
            u.Tests.create_violation(file_path, line_number, "TEST-003", lines)
            for line_number, line in enumerate(lines, start=1)
            if c.Tests.VALIDATOR_PATCH_DECORATOR_RE.match(line) is not None
            and u.Tests.code_match(line, c.Tests.VALIDATOR_PATCH_DECORATOR_RE)
        ]

    @classmethod
    @override
    def _scan_file(
        cls, file_path: Path, approved: t.MappingKV[str, t.StrSequence]
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Scan a single file for test violations."""
        violations: MutableSequence[m.Tests.Violation] = []
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [
                u.Tests.create_violation(
                    file_path,
                    0,
                    "TEST-UNREADABLE",
                    (),
                    read.error or "could not read file",
                )
            ]
        lines = read.value.splitlines()
        violations.extend(cls._check_monkeypatch(file_path, lines, approved))
        violations.extend(cls._check_mock_usage(file_path, lines, approved))
        violations.extend(cls._check_patch_decorator(file_path, lines, approved))
        return violations


__all__: list[str] = ["FlextValidatorTests"]
