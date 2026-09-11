"""Type validation checks for flext-tests."""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from typing import TYPE_CHECKING

from flext_tests import c, t, u

if TYPE_CHECKING:
    from flext_tests import m


class FlextValidatorTypes(u.Tests.ValidatorScannerMixin):
    """Type validation checks for FlextTestsValidator."""

    _VALIDATOR_KEY = c.Tests.VALIDATOR_TYPES_KEY

    @staticmethod
    def _match_names(line: str, pattern: t.Infra.RegexPattern) -> t.StrSequence:
        """Collect distinct named regex matches from one line."""
        return tuple(sorted({match.group("name") for match in pattern.finditer(line)}))

    @classmethod
    def _check_legacy_typing_factories(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect pre-PEP 695 typing factories and Generic base syntax."""
        if u.Tests.approved("TYPE-004", file_path, approved):
            return []

        def emit(line_number: int, name: str) -> m.Tests.Violation:
            violation: m.Tests.Violation = u.Tests.create_violation(
                file_path,
                line_number,
                "TYPE-004",
                lines,
                c.Tests.VALIDATOR_MSG_TYPE_LEGACY_FACTORY.format(name=name),
            )
            return violation

        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            names: set[str] = set()
            if c.Tests.VALIDATOR_LEGACY_FACTORY_RE.search(line) and u.Tests.code_match(
                line, c.Tests.VALIDATOR_LEGACY_FACTORY_RE
            ):
                names.update(
                    cls._match_names(line, c.Tests.VALIDATOR_LEGACY_FACTORY_RE)
                )
            if c.Tests.VALIDATOR_TYPE_ALIAS_RE.search(line) and u.Tests.code_match(
                line, c.Tests.VALIDATOR_TYPE_ALIAS_RE
            ):
                names.add("TypeAlias")
            if c.Tests.VALIDATOR_GENERIC_BASE_RE.search(line) and u.Tests.code_match(
                line, c.Tests.VALIDATOR_GENERIC_BASE_RE
            ):
                names.add("Generic")
            violations.extend(emit(line_number, name) for name in sorted(names))
        return violations

    @classmethod
    def _check_legacy_typing_annotations(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect legacy annotation constructs."""
        if u.Tests.approved("TYPE-005", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            if not c.Tests.VALIDATOR_LEGACY_ANNOTATION_RE.search(line):
                continue
            if not u.Tests.code_match(line, c.Tests.VALIDATOR_LEGACY_ANNOTATION_RE):
                continue
            for name in cls._match_names(line, c.Tests.VALIDATOR_LEGACY_ANNOTATION_RE):
                violations.append(
                    u.Tests.create_violation(
                        file_path,
                        line_number,
                        "TYPE-005",
                        lines,
                        c.Tests.VALIDATOR_MSG_TYPE_LEGACY_ANNOTATION.format(name=name),
                    )
                )
        return violations

    @classmethod
    def _check_object_annotations(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect forbidden object annotations in governed code."""
        if u.Tests.approved("TYPE-006", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_FUNCTION_DEF_RE.match(line) is not None:
                if c.Tests.VALIDATOR_OBJECT_RETURN_RE.search(
                    line
                ) and u.Tests.code_match(line, c.Tests.VALIDATOR_OBJECT_RETURN_RE):
                    violations.append(
                        u.Tests.create_violation(
                            file_path,
                            line_number,
                            "TYPE-006",
                            lines,
                            c.Tests.VALIDATOR_MSG_TYPE_OBJECT_ANNOTATION.format(
                                location="return type"
                            ),
                        )
                    )
                for match in c.Tests.VALIDATOR_OBJECT_ARG_RE.finditer(line):
                    violations.append(
                        u.Tests.create_violation(
                            file_path,
                            line_number,
                            "TYPE-006",
                            lines,
                            c.Tests.VALIDATOR_MSG_TYPE_OBJECT_ANNOTATION.format(
                                location=f"argument '{match.group('arg')}'"
                            ),
                        )
                    )
                continue
            if not c.Tests.VALIDATOR_OBJECT_VAR_RE.search(line):
                continue
            if not u.Tests.code_match(line, c.Tests.VALIDATOR_OBJECT_VAR_RE):
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "TYPE-006",
                    lines,
                    c.Tests.VALIDATOR_MSG_TYPE_OBJECT_ANNOTATION.format(
                        location="variable annotation"
                    ),
                )
            )
        return violations


__all__: list[str] = ["FlextValidatorTypes"]
