"""Violation analysis helpers for flext_infra.refactor."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


class FlextInfraRefactorViolationAnalyzer:
    """Scan files and aggregate massive pattern violations."""

    _PATTERNS: dict[str, re.Pattern[str]] = {
        "container_invariance": re.compile(
            r"\bdict\s*\[\s*str\s*,\s*t\.(?:Container|ContainerValue)\s*\]"
        ),
        "redundant_cast": re.compile(r"\bcast\s*\(\s*[\"'][^\"']+[\"']\s*,"),
        "direct_submodule_import": re.compile(
            r"\bfrom\s+flext_core\.[\w\.]+\s+import\b"
        ),
        "legacy_typing_mapping": re.compile(
            r"\bfrom\s+typing\s+import\s+.*\bMapping\b"
        ),
        "runtime_alias_violation": re.compile(
            r"\bfrom\s+flext_core\s+import\s+(?!.*\b(?:c|m|r|t|u|p|d|e|h|s|x)\b).*"
        ),
    }

    @classmethod
    def analyze_files(cls, files: list[Path]) -> dict[str, object]:
        """Return aggregate and per-file violation counts."""
        totals: Counter[str] = Counter()
        per_file: dict[str, dict[str, int]] = {}

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                continue

            file_counts: dict[str, int] = {}
            for name, pattern in cls._PATTERNS.items():
                count = len(pattern.findall(content))
                if count <= 0:
                    continue
                totals[name] += count
                file_counts[name] = count

            if file_counts:
                per_file[str(file_path)] = file_counts

        hottest_files = sorted(
            (
                {
                    "file": file_name,
                    "total": sum(counts.values()),
                    "counts": counts,
                }
                for file_name, counts in per_file.items()
            ),
            key=lambda item: int(item["total"]),
            reverse=True,
        )

        return {
            "totals": dict(totals),
            "files": per_file,
            "top_files": hottest_files[:25],
            "files_scanned": len(files),
        }


__all__ = ["FlextInfraRefactorViolationAnalyzer"]
