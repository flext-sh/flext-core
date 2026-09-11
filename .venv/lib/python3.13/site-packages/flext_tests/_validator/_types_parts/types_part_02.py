"""Type validation scanner for flext-tests."""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from typing import TYPE_CHECKING, override

from flext_tests import c, t, u
from flext_tests._validator._types_parts.types_part_01 import (
    FlextValidatorTypes as FlextValidatorTypesPart01,
)

if TYPE_CHECKING:
    from flext_tests import m


class FlextValidatorTypes(FlextValidatorTypesPart01):
    """Type validation scanner for FlextTestsValidator."""

    @classmethod
    def _check_bool_returning_is_helpers(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect is_* helpers still returning bool."""
        if u.Tests.approved("TYPE-007", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            match = c.Tests.VALIDATOR_BOOL_IS_HELPER_RE.match(line)
            if match is None:
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "TYPE-007",
                    lines,
                    c.Tests.VALIDATOR_MSG_TYPE_BOOL_IS_HELPER.format(
                        name=match.group("name")
                    ),
                )
            )
        return violations

    @classmethod
    def _check_any_types(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect wildcard type annotations."""
        if u.Tests.approved("TYPE-002", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_FUNCTION_DEF_RE.match(line) is not None:
                if c.Tests.VALIDATOR_ANY_RETURN_RE.search(line) and u.Tests.code_match(
                    line, c.Tests.VALIDATOR_ANY_RETURN_RE
                ):
                    violations.append(
                        u.Tests.create_violation(
                            file_path,
                            line_number,
                            "TYPE-002",
                            lines,
                            c.Tests.VALIDATOR_MSG_TYPE_ANY_RETURN,
                        )
                    )
                for match in c.Tests.VALIDATOR_ANY_ARG_RE.finditer(line):
                    violations.append(
                        u.Tests.create_violation(
                            file_path,
                            line_number,
                            "TYPE-002",
                            lines,
                            c.Tests.VALIDATOR_MSG_TYPE_ANY_ARG.format(
                                arg=match.group("arg")
                            ),
                        )
                    )
                continue
            if not c.Tests.VALIDATOR_ANY_VAR_RE.search(line):
                continue
            if not u.Tests.code_match(line, c.Tests.VALIDATOR_ANY_VAR_RE):
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path, line_number, "TYPE-002", lines, "in variable annotation"
                )
            )
        return violations

    @classmethod
    def _check_cast_usage(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect unapproved cast usage."""
        if u.Tests.approved(
            "TYPE-003", file_path, approved, c.Tests.VALIDATOR_APPROVED_CAST_PATTERNS
        ):
            return []
        return [
            u.Tests.create_violation(file_path, line_number, "TYPE-003", lines)
            for line_number, line in enumerate(lines, start=1)
            if c.Tests.VALIDATOR_CAST_USAGE_RE.search(line) is not None
            and u.Tests.code_match(line, c.Tests.VALIDATOR_CAST_USAGE_RE)
        ]

    @classmethod
    def _check_type_ignore(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect type: ignore comments in code."""
        if u.Tests.approved("TYPE-001", file_path, approved):
            return []
        return [
            u.Tests.create_violation(file_path, line_number, "TYPE-001", lines)
            for line_number, line in enumerate(lines, start=1)
            if c.Tests.VALIDATOR_TYPE_IGNORE_RE.search(line)
            and u.Tests.real_comment(line, c.Tests.VALIDATOR_TYPE_IGNORE_RE)
        ]

    @classmethod
    @override
    def _scan_file(
        cls, file_path: Path, approved: t.MappingKV[str, t.StrSequence]
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Scan a single file for type violations."""
        violations: MutableSequence[m.Tests.Violation] = []
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [
                u.Tests.create_violation(
                    file_path,
                    0,
                    "TYPE-UNREADABLE",
                    (),
                    read.error or "could not read file",
                )
            ]
        lines = read.value.splitlines()
        violations.extend(cls._check_type_ignore(file_path, lines, approved))
        violations.extend(cls._check_any_types(file_path, lines, approved))
        violations.extend(cls._check_cast_usage(file_path, lines, approved))
        violations.extend(
            cls._check_legacy_typing_factories(file_path, lines, approved)
        )
        violations.extend(
            cls._check_legacy_typing_annotations(file_path, lines, approved)
        )
        violations.extend(cls._check_object_annotations(file_path, lines, approved))
        violations.extend(
            cls._check_bool_returning_is_helpers(file_path, lines, approved)
        )
        return violations


__all__: list[str] = ["FlextValidatorTypes"]
