"""Pytest diagnostics extraction service.

Extracts robust pytest diagnostics from JUnit XML and log outputs,
producing structured failure/error/warning/skip/slow-test reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra.base import s
from flext_infra.validate._pytest_diag_xml import FlextInfraPytestDiagXmlMixin

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraPytestDiagExtractor(FlextInfraPytestDiagXmlMixin, s[bool]):
    """Extracts pytest diagnostics from JUnit XML and log files.

    Parses JUnit XML for structured failure/error/skip/timing data
    and uses regex-based log parsing when XML is unavailable.
    """

    junit: Annotated[Path, m.Field(description="JUnit XML path")]
    log_path: Annotated[Path, m.Field(description="Pytest log path")] = m.Field(
        alias="log"
    )
    failed: Annotated[
        Path | None, m.Field(description="Path to write failed cases")
    ] = None
    errors: Annotated[
        Path | None, m.Field(description="Path to write error traces")
    ] = None
    warnings: Annotated[Path | None, m.Field(description="Path to write warnings")] = (
        None
    )
    slowest: Annotated[
        Path | None, m.Field(description="Path to write slowest entries")
    ] = None
    skips: Annotated[
        Path | None, m.Field(description="Path to write skipped cases")
    ] = None

    @staticmethod
    def _extract_slow_from_log(lines: t.StrSequence, diag: m.Infra.DiagResult) -> None:
        """Extract slow test durations from log when XML unavailable."""
        capture_slow = False
        for line in lines:
            if c.Infra.PYTEST_SLOWEST_HEADER_RE.match(line):
                capture_slow = True
                continue
            if capture_slow and c.Infra.PYTEST_SECTION_DIVIDER_RE.match(line):
                break
            if capture_slow and line.strip():
                diag.slow_entries.append(line)

    @staticmethod
    def _extract_warnings(lines: t.StrSequence, diag: m.Infra.DiagResult) -> None:
        """Extract warning lines from pytest log."""
        capture_warn = False
        for line in lines:
            if c.Infra.PYTEST_WARNINGS_HEADER_RE.match(line):
                capture_warn = True
                continue
            if capture_warn and c.Infra.PYTEST_DOCS_FOOTER_RE.match(line):
                break
            if capture_warn and c.Infra.PYTEST_WARNING_LINE_RE.search(line):
                diag.warning_lines.append(line)
        if not diag.warning_lines:
            diag.warning_lines = [
                line for line in lines if c.Infra.PYTEST_WARNING_LINE_RE.search(line)
            ]

    @staticmethod
    def _parse_log_into_diag(lines: t.StrSequence, diag: m.Infra.DiagResult) -> None:
        """Parse pytest log output for failures/skips when XML unavailable."""
        diag.failed_cases = [
            line for line in lines if c.Infra.PYTEST_FAILED_LINE_RE.search(line)
        ]
        diag.error_cases = [
            line for line in lines if c.Infra.PYTEST_ERROR_LINE_RE.search(line)
        ]
        diag.skip_cases = [
            line for line in lines if c.Infra.PYTEST_SKIPPED_LINE_RE.search(line)
        ]
        capture = False
        block: t.MutableSequenceOf[str] = []
        for line in lines:
            if c.Infra.PYTEST_FAILURES_OR_ERRORS_RE.match(line):
                capture = True
            if capture:
                block.append(line)
                if c.Infra.PYTEST_BLOCK_END_RE.match(line):
                    break
        diag.error_traces = block

    def extract(
        self, junit_path: Path, log_path: Path
    ) -> p.Result[m.Infra.PytestDiagnostics]:
        """Extract diagnostics from JUnit XML and pytest log.

        Args:
            junit_path: Path to JUnit XML result file.
            log_path: Path to raw pytest log output.

        Returns:
            r with diagnostics dict containing counts and entries.

        """
        try:
            return self._extract_diagnostics(junit_path, log_path)
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[m.Infra.PytestDiagnostics].fail_op(
                "pytest diagnostics extraction", exc
            )

    @staticmethod
    def _read_log_text(log_path: Path) -> p.Result[str]:
        """Read pytest log text when present."""
        if not log_path.exists():
            return r[str].ok("")
        log_read = u.Cli.files_read_text(log_path)
        if log_read.failure:
            return r[str].fail(
                log_read.error or f"Failed to read pytest log: {log_path}"
            )
        return r[str].ok(log_read.value)

    @staticmethod
    def _diagnostics_model(diag: m.Infra.DiagResult) -> m.Infra.PytestDiagnostics:
        """Convert mutable extraction state to the canonical diagnostics model."""
        return m.Infra.PytestDiagnostics(
            failed_count=len(diag.failed_cases),
            error_count=len(diag.error_cases),
            warning_count=len(diag.warning_lines),
            skipped_count=len(diag.skip_cases),
            failed_cases=diag.failed_cases,
            error_traces=diag.error_traces,
            warning_lines=diag.warning_lines,
            skip_cases=diag.skip_cases,
            slow_entries=diag.slow_entries,
        )

    def _extract_diagnostics(
        self, junit_path: Path, log_path: Path
    ) -> p.Result[m.Infra.PytestDiagnostics]:
        """Extract pytest diagnostics after input normalization."""
        log_text_result = self._read_log_text(log_path)
        if log_text_result.failure:
            return r[m.Infra.PytestDiagnostics].fail(
                log_text_result.error or f"Failed to read pytest log: {log_path}"
            )
        lines = log_text_result.value.splitlines()
        diag = m.Infra.DiagResult()
        xml_parsed = self._parse_xml(junit_path, diag)
        if not xml_parsed:
            self._parse_log_into_diag(lines, diag)
        self._extract_warnings(lines, diag)
        if not diag.slow_entries:
            self._extract_slow_from_log(lines, diag)
        return r[m.Infra.PytestDiagnostics].ok(self._diagnostics_model(diag))

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the pytest diagnostics CLI flow."""
        result = self.extract(self.junit, self.log_path)
        if result.failure:
            return r[bool].fail(result.error or "extraction failed")
        for output_path, attr_name, separator in [
            (self.failed, "failed_cases", "\n\n"),
            (self.errors, "error_traces", "\n\n"),
            (self.warnings, "warning_lines", "\n"),
            (self.slowest, "slow_entries", "\n"),
            (self.skips, "skip_cases", "\n"),
        ]:
            if output_path is None:
                continue
            items = [
                value
                for value in getattr(result.value, attr_name, [])
                if isinstance(value, str)
            ]
            u.write_file(
                output_path,
                separator.join(items) + "\n",
                encoding=c.Cli.ENCODING_DEFAULT,
            )
        sys.stdout.write(
            f"failed_count={result.value.failed_count}\n"
            f"error_count={result.value.error_count}\n"
            f"warning_count={result.value.warning_count}\n"
            f"skipped_count={result.value.skipped_count}\n"
        )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraPytestDiagExtractor"]
