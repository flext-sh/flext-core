"""Text pattern scanning service.

Scans text files with regex patterns and reports violation counts,
supporting both present-match and absent-match violation modes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.base import s

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraTextPatternScanner(s[bool]):
    """Scans files for regex pattern matches and reports violations.

    Supports include/exclude glob filtering and configurable match
    modes (present = matches are violations, absent = no matches is a violation).
    """

    pattern: Annotated[str, m.Field(description="Regex pattern")]
    include: Annotated[
        t.StrSequence,
        m.Field(
            default_factory=tuple, description="Glob patterns included in the scan."
        ),
    ] = m.Field(default_factory=tuple)
    exclude: Annotated[
        t.StrSequence,
        m.Field(
            default_factory=tuple, description="Glob patterns excluded from the scan."
        ),
    ] = m.Field(default_factory=tuple)
    match: Annotated[
        c.Infra.MatchMode, m.Field(description="Violation mode (present or absent)")
    ] = c.Infra.MatchMode.PRESENT

    @staticmethod
    def _count_matches(
        files: t.SequenceOf[Path], regex: t.Infra.RegexPattern
    ) -> p.Result[int]:
        """Count regex matches across files; surface any unreadable file as failure."""
        total = 0
        for file_path in files:
            read = u.Cli.files_read_text(file_path)
            if read.failure:
                return r[int].fail(read.error or f"unreadable file: {file_path}")
            total += sum(1 for _ in regex.finditer(read.value))
        return r[int].ok(total)

    def scan(
        self,
        scan_root: Path,
        pattern: str,
        *,
        includes: t.StrSequence,
        excludes: t.StrSequence | None = None,
        match_mode: c.Infra.MatchMode = c.Infra.MatchMode.PRESENT,
    ) -> p.Result[t.ConfigurationMapping]:
        """Scan files under scan_root for regex matches.

        Args:
            scan_root: Directory to scan.
            pattern: Regex pattern to search for.
            includes: Glob patterns for files to include.
            excludes: Glob patterns for files to exclude.
            match_mode: ``"present"`` (matches = violations) or
                ``"absent"`` (no matches = violation).

        Returns:
            r with violation count and match details.

        """
        error = self._validate_scan_inputs(scan_root, includes)
        if error is not None:
            return r[t.ScalarMapping].fail(error)
        try:
            return self._scan_validated(
                scan_root, pattern, includes, excludes or (), match_mode
            )
        except c.Infra.REGEX_ERROR as exc:
            return r[t.ScalarMapping].fail(f"invalid regex pattern: {exc}")
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[t.ScalarMapping].fail_op("text pattern scan", exc)

    @staticmethod
    def _violation_count(matches: int, match_mode: c.Infra.MatchMode) -> int:
        """Return violation count for the selected match mode."""
        if match_mode == c.Infra.MatchMode.PRESENT:
            return matches
        return 0 if matches > 0 else 1

    def _scan_validated(
        self,
        scan_root: Path,
        pattern: str,
        includes: t.StrSequence,
        excludes: t.StrSequence,
        match_mode: c.Infra.MatchMode,
    ) -> p.Result[t.ConfigurationMapping]:
        """Scan a validated root with a compiled regex."""
        regex = c.Infra.compile_multiline(pattern)
        files = u.Infra.iter_matching_files(
            scan_root, includes=includes, excludes=excludes
        )
        matches_result = self._count_matches(files, regex)
        if matches_result.failure:
            return r[t.ScalarMapping].fail(
                matches_result.error or "text pattern scan read failed"
            )
        matches = matches_result.value
        result: t.MutableConfigurationMapping = {
            "violation_count": self._violation_count(matches, match_mode),
            "match_count": matches,
            "files_scanned": len(files),
        }
        return r[t.ScalarMapping].ok(result)

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the text-pattern scan CLI flow."""
        result = self.scan(
            self.repository_root,
            self.pattern,
            includes=self.include,
            excludes=self.exclude,
            match_mode=self.match,
        )
        if result.failure:
            return r[bool].fail(result.error or "scan failed")
        count = result.value.get("violation_count", 0)
        if isinstance(count, int) and count > 0:
            return r[bool].fail(f"Scan found {count} violation(s)")
        return r[bool].ok(True)

    @staticmethod
    def _validate_scan_inputs(scan_root: Path, includes: t.StrSequence) -> str | None:
        """Return an error message if scan inputs are invalid, else None."""
        if not scan_root.exists() or not scan_root.is_dir():
            return f"scan_root directory does not exist: {scan_root}"
        if not includes:
            return "at least one include glob required"
        return None


__all__: list[str] = ["FlextInfraTextPatternScanner"]
