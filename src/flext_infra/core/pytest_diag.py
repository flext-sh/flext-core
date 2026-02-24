"""Pytest diagnostics extraction service.

Extracts robust pytest diagnostics from JUnit XML and log outputs,
producing structured failure/error/warning/skip/slow-test reports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: S405

from flext_core.result import FlextResult, r
from flext_core.typings import t

from flext_infra.constants import ic


class _DiagResult:
    """Internal container for extracted diagnostics."""

    __slots__ = (
        "error_traces",
        "failed_cases",
        "skip_cases",
        "slow_entries",
        "warning_lines",
    )

    def __init__(self) -> None:
        self.failed_cases: list[str] = []
        self.error_traces: list[str] = []
        self.skip_cases: list[str] = []
        self.warning_lines: list[str] = []
        self.slow_entries: list[str] = []


class PytestDiagExtractor:
    """Extracts pytest diagnostics from JUnit XML and log files.

    Parses JUnit XML for structured failure/error/skip/timing data
    and uses regex-based log parsing when XML is unavailable.
    """

    _ENCODING = ic.Encoding.DEFAULT

    def extract(
        self,
        junit_path: Path,
        log_path: Path,
    ) -> FlextResult[Mapping[str, t.ConfigMapValue]]:
        """Extract diagnostics from JUnit XML and pytest log.

        Args:
            junit_path: Path to JUnit XML result file.
            log_path: Path to raw pytest log output.

        Returns:
            FlextResult with diagnostics dict containing counts and entries.

        """
        try:
            log_text = (
                log_path.read_text(encoding=self._ENCODING, errors="replace")
                if log_path.exists()
                else ""
            )
            lines = log_text.splitlines()
            diag = _DiagResult()

            xml_parsed = self._parse_xml(junit_path, diag)
            if not xml_parsed:
                self._parse_log_into_diag(lines, diag)

            self._extract_warnings(lines, diag)
            if not diag.slow_entries:
                self._extract_slow_from_log(lines, diag)

            result: MutableMapping[str, t.ConfigMapValue] = {
                "failed_count": len(diag.failed_cases),
                "error_count": len(diag.error_traces),
                "warning_count": len(diag.warning_lines),
                "skipped_count": len(diag.skip_cases),
                "failed_cases": diag.failed_cases,
                "error_traces": diag.error_traces,
                "warning_lines": diag.warning_lines,
                "skip_cases": diag.skip_cases,
                "slow_entries": diag.slow_entries,
            }
            return r[Mapping[str, t.ConfigMapValue]].ok(result)
        except (OSError, TypeError, ValueError) as exc:
            return r[Mapping[str, t.ConfigMapValue]].fail(
                f"pytest diagnostics extraction failed: {exc}",
            )

    @staticmethod
    def _parse_xml(junit_path: Path, diag: _DiagResult) -> bool:
        """Parse JUnit XML and populate diagnostics. Returns True on success."""
        if not junit_path.exists():
            return False
        try:
            root = ET.parse(junit_path).getroot()  # noqa: S314
        except ET.ParseError:
            return False

        slow_rows: list[tuple[float, str]] = []
        for case in root.iter("testcase"):
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            label = f"{classname}::{name}" if classname else name
            try:
                secs = float(case.attrib.get("time", "0") or 0.0)
            except ValueError:
                secs = 0.0
            slow_rows.append((secs, label))

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                diag.failed_cases.append(label)
                msg = (failure.attrib.get("message") or "").strip()
                trace = (failure.text or "").strip()
                chunk = [f"=== FAILURE: {label} ==="]
                if msg:
                    chunk.append(msg)
                if trace:
                    chunk.append(trace)
                diag.error_traces.append("\n".join(chunk))

            if error is not None:
                msg = (error.attrib.get("message") or "").strip()
                trace = (error.text or "").strip()
                chunk = [f"=== ERROR: {label} ==="]
                if msg:
                    chunk.append(msg)
                if trace:
                    chunk.append(trace)
                diag.error_traces.append("\n".join(chunk))

            if skipped is not None:
                reason = (skipped.attrib.get("message") or skipped.text or "").strip()
                diag.skip_cases.append(
                    f"{label} | {reason}" if reason else label,
                )

        if slow_rows:
            diag.slow_entries = [
                f"{secs:.6f}s | {label}"
                for secs, label in sorted(slow_rows, reverse=True)
            ]
        return True

    @staticmethod
    def _parse_log_into_diag(lines: list[str], diag: _DiagResult) -> None:
        """Parse pytest log output for failures/skips when XML unavailable."""
        diag.failed_cases = [
            line for line in lines if re.search(r"(^FAILED |::.* FAILED( |$))", line)
        ]
        diag.skip_cases = [
            line for line in lines if re.search(r"(^SKIPPED |::.* SKIPPED( |$))", line)
        ]

        capture = False
        block: list[str] = []
        for line in lines:
            if re.match(r"^=+ (FAILURES|ERRORS) =+", line):
                capture = True
            if capture:
                block.append(line)
                if re.match(
                    r"^=+ (short test summary info|warnings summary|.+ in [0-9.]+s) =+",
                    line,
                ):
                    break
        diag.error_traces = block

    @staticmethod
    def _extract_warnings(lines: list[str], diag: _DiagResult) -> None:
        """Extract warning lines from pytest log."""
        capture_warn = False
        for line in lines:
            if re.match(r"^=+ warnings summary =+", line):
                capture_warn = True
            if capture_warn:
                diag.warning_lines.append(line)
                if re.match(r"^-- Docs: https://docs.pytest.org/", line):
                    break
        if not diag.warning_lines:
            diag.warning_lines = [
                line
                for line in lines
                if re.search(
                    r"CoverageWarning|PytestCollectionWarning|DeprecationWarning"
                    r"|UserWarning|RuntimeWarning",
                    line,
                )
            ]

    @staticmethod
    def _extract_slow_from_log(lines: list[str], diag: _DiagResult) -> None:
        """Extract slow test durations from log when XML unavailable."""
        capture_slow = False
        for line in lines:
            if re.match(r"^=+ slowest durations =+", line):
                capture_slow = True
                continue
            if capture_slow and re.match(r"^=+", line):
                break
            if capture_slow and line.strip():
                diag.slow_entries.append(line)


__all__ = ["PytestDiagExtractor"]
