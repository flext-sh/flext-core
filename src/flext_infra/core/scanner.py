"""Text pattern scanning service.

Scans text files with regex patterns and reports violation counts,
supporting both present-match and absent-match violation modes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_core.result import FlextResult, r
from flext_core.typings import t
from flext_infra.constants import c


class TextPatternScanner:
    """Scans files for regex pattern matches and reports violations.

    Supports include/exclude glob filtering and configurable match
    modes (present = matches are violations, absent = no matches is a violation).
    """

    _ENCODING = c.Encoding.DEFAULT

    def scan(
        self,
        root: Path,
        pattern: str,
        *,
        includes: list[str],
        excludes: list[str] | None = None,
        match_mode: str = "present",
    ) -> FlextResult[Mapping[str, t.ScalarValue]]:
        """Scan files under root for regex matches.

        Args:
            root: Directory to scan.
            pattern: Regex pattern to search for.
            includes: Glob patterns for files to include.
            excludes: Glob patterns for files to exclude.
            match_mode: ``"present"`` (matches = violations) or
                ``"absent"`` (no matches = violation).

        Returns:
            FlextResult with violation count and match details.

        """
        try:
            if not root.exists() or not root.is_dir():
                return r[Mapping[str, t.ScalarValue]].fail(
                    f"root directory does not exist: {root}",
                )
            if not includes:
                return r[Mapping[str, t.ScalarValue]].fail(
                    "at least one include glob required"
                )
            if match_mode not in {"present", "absent"}:
                return r[Mapping[str, t.ScalarValue]].fail(
                    f"invalid match_mode: {match_mode}",
                )

            regex = re.compile(pattern, flags=re.MULTILINE)
            files = self._collect_files(root, includes, excludes or [])
            matches = self._count_matches(files, regex)

            violation_count = (
                matches if match_mode == "present" else (0 if matches > 0 else 1)
            )

            result: MutableMapping[str, t.ScalarValue] = {
                "violation_count": violation_count,
                "match_count": matches,
                "files_scanned": len(files),
            }
            return r[Mapping[str, t.ScalarValue]].ok(result)
        except re.error as exc:
            return r[Mapping[str, t.ScalarValue]].fail(f"invalid regex pattern: {exc}")
        except (OSError, ValueError, TypeError) as exc:
            return r[Mapping[str, t.ScalarValue]].fail(
                f"text pattern scan failed: {exc}",
            )

    @staticmethod
    def _collect_files(
        root: Path,
        includes: list[str],
        excludes: list[str],
    ) -> list[Path]:
        """Collect files matching include/exclude globs."""
        selected: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in includes):
                if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                    continue
                selected.append(path)
        return selected

    @staticmethod
    def _count_matches(files: list[Path], regex: re.Pattern[str]) -> int:
        """Count regex matches across files."""
        total = 0
        for file_path in files:
            try:
                text = file_path.read_text(
                    encoding=c.Encoding.DEFAULT,
                    errors="ignore",
                )
            except OSError:
                continue
            total += sum(1 for _ in regex.finditer(text))
        return total


__all__ = ["TextPatternScanner"]
