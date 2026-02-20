#!/usr/bin/env python3
"""Extract robust pytest diagnostics artifacts from junit/log outputs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def _write_or_placeholder(path: Path, entries: list[str], placeholder: str) -> None:
    values = entries if entries else [placeholder]
    _ = path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _count_effective(path: Path, placeholder: str) -> int:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    return sum(1 for line in lines if line and line != placeholder)


def main() -> int:
    if len(sys.argv) != 8:
        raise SystemExit(
            "usage: pytest_diag_extract.py <junit> <log> <failed> <errors> <warnings> <slowest> <skips>"
        )

    junit_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    failed_path = Path(sys.argv[3])
    errors_path = Path(sys.argv[4])
    warnings_path = Path(sys.argv[5])
    slowest_path = Path(sys.argv[6])
    skips_path = Path(sys.argv[7])

    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )
    lines = log_text.splitlines()

    failed_cases: list[str] = []
    error_traces: list[str] = []
    skip_cases: list[str] = []
    warning_lines: list[str] = []
    slow_rows: list[tuple[float, str]] = []

    xml_parsed = False
    if junit_path.exists():
        try:
            root = ET.parse(junit_path).getroot()
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
                    failed_cases.append(label)
                    msg = (failure.attrib.get("message") or "").strip()
                    trace = (failure.text or "").strip()
                    chunk = [f"=== FAILURE: {label} ==="]
                    if msg:
                        chunk.append(msg)
                    if trace:
                        chunk.append(trace)
                    error_traces.append("\n".join(chunk))

                if error is not None:
                    msg = (error.attrib.get("message") or "").strip()
                    trace = (error.text or "").strip()
                    chunk = [f"=== ERROR: {label} ==="]
                    if msg:
                        chunk.append(msg)
                    if trace:
                        chunk.append(trace)
                    error_traces.append("\n".join(chunk))

                if skipped is not None:
                    reason = (
                        skipped.attrib.get("message") or skipped.text or ""
                    ).strip()
                    skip_cases.append(f"{label} | {reason}" if reason else label)
            xml_parsed = True
        except ET.ParseError:
            xml_parsed = False

    if not xml_parsed:
        failed_cases = [
            line for line in lines if re.search(r"(^FAILED |::.* FAILED( |$))", line)
        ]
        skip_cases = [
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
        error_traces = block

    capture_warn = False
    for line in lines:
        if re.match(r"^=+ warnings summary =+", line):
            capture_warn = True
        if capture_warn:
            warning_lines.append(line)
            if re.match(r"^-- Docs: https://docs.pytest.org/", line):
                break
    if not warning_lines:
        warning_lines = [
            line
            for line in lines
            if re.search(
                r"CoverageWarning|PytestCollectionWarning|DeprecationWarning|UserWarning|RuntimeWarning",
                line,
            )
        ]

    if slow_rows:
        slow_entries = [
            f"{secs:.6f}s | {label}" for secs, label in sorted(slow_rows, reverse=True)
        ]
    else:
        capture_slow = False
        slow_entries: list[str] = []
        for line in lines:
            if re.match(r"^=+ slowest durations =+", line):
                capture_slow = True
                continue
            if capture_slow and re.match(r"^=+", line):
                break
            if capture_slow and line.strip():
                slow_entries.append(line)

    _write_or_placeholder(failed_path, failed_cases, "No failing tests.")
    _write_or_placeholder(errors_path, error_traces, "No error traces captured.")
    _write_or_placeholder(warnings_path, warning_lines, "No warnings captured.")
    _write_or_placeholder(skips_path, skip_cases, "No skipped tests.")
    _write_or_placeholder(slowest_path, slow_entries, "No slow-duration section found.")

    failed_count = _count_effective(failed_path, "No failing tests.")
    error_headers = sum(
        1
        for line in errors_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if line.startswith("=== FAILURE:") or line.startswith("=== ERROR:")
    )
    error_count = (
        error_headers
        if error_headers
        else _count_effective(errors_path, "No error traces captured.")
    )
    warning_count = _count_effective(warnings_path, "No warnings captured.")
    skipped_count = _count_effective(skips_path, "No skipped tests.")

    print(f"failed_count={failed_count}")
    print(f"error_count={error_count}")
    print(f"warning_count={warning_count}")
    print(f"skipped_count={skipped_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
