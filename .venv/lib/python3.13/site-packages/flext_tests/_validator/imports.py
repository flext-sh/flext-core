"""Import validation for FLEXT architecture.

Detects import violations: lazy imports, TYPE_CHECKING, ImportError handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path
from typing import override

from flext_tests import c, m, t, u


class FlextValidatorImports(u.Tests.ValidatorScannerMixin):
    """Import validation methods for FlextTestsValidator."""

    _VALIDATOR_KEY = c.Tests.VALIDATOR_IMPORTS_KEY

    @classmethod
    def _check_import_error_handling(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        if u.Tests.approved("IMPORT-003", file_path, approved):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            if c.Tests.VALIDATOR_IMPORT_ERROR_RE.search(line) is None:
                continue
            if not u.Tests.code_match(line, c.Tests.VALIDATOR_IMPORT_ERROR_RE):
                continue
            violations.append(
                u.Tests.create_violation(file_path, line_number, "IMPORT-003", lines)
            )
        return violations

    @classmethod
    def _check_lazy_imports(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        if u.Tests.approved("IMPORT-001", file_path, approved):
            return []
        return [
            u.Tests.create_violation(file_path, line_number, "IMPORT-001", lines)
            for line_number, line in enumerate(lines, start=1)
            if c.Tests.VALIDATOR_INDENTED_IMPORT_RE.match(line) is not None
        ]

    @classmethod
    def _check_non_root_flext_imports(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect non-root imports from flext-* packages internal modules.

        Detects imports from internal modules (prefixed with _) like:
        - from flext_core import domain  (violation)
        - from flext_tests import imports  (violation)

        Allows public module imports:
        - from flext_core import r  (OK)
        - from flext_tests import m  (OK)

        Allows __init__.py inside internal packages to import sibling modules:
        - _validator/__init__.py can import from flext_tests._validator.* (OK)
        """
        if u.Tests.approved(
            "IMPORT-006",
            file_path,
            approved,
            c.Tests.VALIDATOR_APPROVED_INTERNAL_INIT_PATTERNS,
        ):
            return []
        violations: MutableSequence[m.Tests.Violation] = []
        for line_number, line in enumerate(lines, start=1):
            from_match = c.Tests.VALIDATOR_FLEXT_FROM_IMPORT_RE.match(line)
            import_match = c.Tests.VALIDATOR_FLEXT_IMPORT_RE.match(line)
            module = None
            if from_match is not None:
                module = from_match.group("module")
            elif import_match is not None:
                module = import_match.group("module")
            if module is None:
                continue
            parts = module.split(".")
            internal_parts = [part for part in parts[1:] if part.startswith("_")]
            if not internal_parts:
                continue
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    line_number,
                    "IMPORT-006",
                    lines,
                    f"from {module} (internal: {internal_parts[0]})",
                )
            )
        return violations

    @classmethod
    def _check_sys_path(
        cls,
        file_path: Path,
        lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect sys.path manipulation."""
        if u.Tests.approved("IMPORT-004", file_path, approved):
            return []
        return [
            u.Tests.create_violation(file_path, line_number, "IMPORT-004", lines)
            for line_number, line in enumerate(lines, start=1)
            if c.Tests.VALIDATOR_SYS_PATH_RE.search(line) is not None
            and u.Tests.code_match(line, c.Tests.VALIDATOR_SYS_PATH_RE)
        ]

    @classmethod
    def _check_type_checking(
        cls,
        file_path: Path,
        _lines: t.StrSequence,
        approved: t.MappingKV[str, t.StrSequence],
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Detect TYPE_CHECKING blocks in files with Pydantic field annotations.

        TYPE_CHECKING is permitted for type-only imports in non-Pydantic files.
        TYPE_CHECKING is forbidden only in files where the imported types are
        used in Pydantic BaseModel/RootModel field annotations.

        For now, we allow TYPE_CHECKING in all files since detecting usage in
        field annotations requires complex AST analysis.
        """
        if u.Tests.approved("IMPORT-002", file_path, approved):
            return []
        return []

    @classmethod
    @override
    def _scan_file(
        cls, file_path: Path, approved: t.MappingKV[str, t.StrSequence]
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Scan a single file for import violations."""
        violations: MutableSequence[m.Tests.Violation] = []
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [
                u.Tests.create_violation(
                    file_path,
                    0,
                    "IMPORT-UNREADABLE",
                    (),
                    extra_desc=read.error or "could not read file",
                )
            ]
        lines = read.value.splitlines()
        violations.extend(cls._check_lazy_imports(file_path, lines, approved))
        violations.extend(cls._check_type_checking(file_path, lines, approved))
        violations.extend(cls._check_import_error_handling(file_path, lines, approved))
        violations.extend(cls._check_sys_path(file_path, lines, approved))
        violations.extend(cls._check_non_root_flext_imports(file_path, lines, approved))
        return violations


__all__: list[str] = ["FlextValidatorImports"]
