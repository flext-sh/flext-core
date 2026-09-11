"""Markdown code-block validator for flext-tests."""

from __future__ import annotations

from collections.abc import MutableSequence
from pathlib import Path

from flext_tests import c, m, p, t, u


class FlextValidatorMarkdown:
    """Validate Python code blocks inside markdown docs."""

    @classmethod
    def markdown(
        cls,
        paths: t.SequenceOf[Path],
        *,
        approved_exceptions: t.MappingKV[str, t.StrSequence] | None = None,
    ) -> p.Result[m.Tests.ScanResult]:
        """Validate Python code blocks in markdown files."""
        return u.Tests.validator_run_scan(
            files=list(paths),
            approved_exceptions=approved_exceptions,
            validator_name=c.Tests.VALIDATOR_MARKDOWN_KEY,
            scan_file=cls._scan_file,
        )

    @classmethod
    def _scan_file(
        cls, file_path: Path, approved: t.MappingKV[str, t.StrSequence]
    ) -> t.SequenceOf[m.Tests.Violation]:
        """Scan a single markdown file for Python code block violations."""
        violations: MutableSequence[m.Tests.Violation] = []

        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return [
                u.Tests.create_violation(
                    file_path,
                    0,
                    "MD-UNREADABLE",
                    (),
                    read.error or "could not read file",
                )
            ]

        content = read.value
        lines = content.splitlines()

        for match in c.Tests.VALIDATOR_MD_PYTHON_BLOCK_RE.finditer(content):
            info = match.group("info")
            code = match.group("code")
            block_start = content[: match.start()].count("\n") + 1
            is_notest = c.Tests.VALIDATOR_MD_NOTEST_MARKER in info

            if not is_notest:
                cls._check_syntax(file_path, code, lines, block_start, violations)

            cls._check_forbidden_imports(
                file_path, code, lines, block_start, approved, violations
            )
            cls._check_object_annotations(
                file_path, code, lines, block_start, approved, violations
            )
            cls._check_any_annotations(
                file_path, code, lines, block_start, approved, violations
            )
            cls._check_future_annotations(
                file_path, code, lines, block_start, approved, violations
            )

        return violations

    @classmethod
    def _check_syntax(
        cls,
        file_path: Path,
        code: str,
        lines: t.StrSequence,
        block_start: int,
        violations: MutableSequence[m.Tests.Violation],
    ) -> None:
        """Check that a Python code block is syntactically valid."""
        try:
            compile(code, str(file_path), "exec")
        except SyntaxError:
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    block_start,
                    "MD-001",
                    lines,
                    c.Tests.VALIDATOR_MSG_MD_SYNTAX.format(msg="invalid Python"),
                )
            )

    @classmethod
    def _check_forbidden_imports(
        cls,
        file_path: Path,
        code: str,
        lines: t.StrSequence,
        block_start: int,
        approved: t.MappingKV[str, t.StrSequence],
        violations: MutableSequence[m.Tests.Violation],
    ) -> None:
        """Check for forbidden typing imports via line scanning."""
        if u.Tests.approved("MD-002", file_path, approved):
            return
        for line_offset, code_line in enumerate(code.splitlines()):
            stripped = code_line.strip()
            if not stripped.startswith(c.Tests.VALIDATOR_MD_TYPING_IMPORT_PREFIX):
                continue
            for name in c.Tests.VALIDATOR_MD_FORBIDDEN_TYPING_NAMES:
                if name in stripped:
                    violations.append(
                        u.Tests.create_violation(
                            file_path,
                            block_start + line_offset + 1,
                            "MD-002",
                            lines,
                            c.Tests.VALIDATOR_MSG_MD_FORBIDDEN_IMPORT.format(
                                import_name=f"from typing import {name}"
                            ),
                        )
                    )

    @classmethod
    def _check_object_annotations(
        cls,
        file_path: Path,
        code: str,
        lines: t.StrSequence,
        block_start: int,
        approved: t.MappingKV[str, t.StrSequence],
        violations: MutableSequence[m.Tests.Violation],
    ) -> None:
        """Check for 'object' used as type annotation."""
        if u.Tests.approved("MD-004", file_path, approved):
            return
        for line_offset, code_line in enumerate(code.splitlines()):
            if c.Tests.VALIDATOR_MD_OBJECT_ANNOTATION_RE.search(code_line):
                violations.append(
                    u.Tests.create_violation(
                        file_path,
                        block_start + line_offset + 1,
                        "MD-004",
                        lines,
                        c.Tests.VALIDATOR_MSG_MD_FORBIDDEN_ANNOTATION.format(
                            annotation="object"
                        ),
                    )
                )

    @classmethod
    def _check_any_annotations(
        cls,
        file_path: Path,
        code: str,
        lines: t.StrSequence,
        block_start: int,
        approved: t.MappingKV[str, t.StrSequence],
        violations: MutableSequence[m.Tests.Violation],
    ) -> None:
        """Check for 'Any' used as type annotation."""
        if u.Tests.approved("MD-005", file_path, approved):
            return
        for line_offset, code_line in enumerate(code.splitlines()):
            if c.Tests.VALIDATOR_MD_ANY_ANNOTATION_RE.search(code_line):
                violations.append(
                    u.Tests.create_violation(
                        file_path,
                        block_start + line_offset + 1,
                        "MD-005",
                        lines,
                        c.Tests.VALIDATOR_MSG_MD_FORBIDDEN_ANNOTATION.format(
                            annotation="Any"
                        ),
                    )
                )

    @classmethod
    def _check_future_annotations(
        cls,
        file_path: Path,
        code: str,
        lines: t.StrSequence,
        block_start: int,
        approved: t.MappingKV[str, t.StrSequence],
        violations: MutableSequence[m.Tests.Violation],
    ) -> None:
        """Check for missing future annotations import."""
        if u.Tests.approved("MD-003", file_path, approved):
            return
        has_future = c.Tests.VALIDATOR_MD_FUTURE_ANNOTATIONS_MARKER in code
        has_annotations = any(
            ":" in line and not line.strip().startswith("#")
            for line in code.splitlines()
            if "def " in line or "class " in line or "->" in line
        )
        if not has_future and has_annotations:
            violations.append(
                u.Tests.create_violation(
                    file_path,
                    block_start,
                    "MD-003",
                    lines,
                    c.Tests.VALIDATOR_MSG_MD_MISSING_FUTURE,
                )
            )

    @classmethod
    def collect_markdown_files(cls, project_root: Path) -> t.SequenceOf[Path]:
        """Collect all .md files that may contain Python code blocks."""
        md_files: MutableSequence[Path] = []
        for search_dir in (
            project_root,
            project_root / ".claude" / "skills",
            project_root / "docs",
        ):
            if search_dir.is_dir():
                md_files.extend(
                    u.Infra.iter_matching_files(search_dir, includes=["*.md"])
                )
        return md_files


__all__: list[str] = ["FlextValidatorMarkdown"]
