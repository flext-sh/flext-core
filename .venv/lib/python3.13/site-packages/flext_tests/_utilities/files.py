"""Test-domain file helpers (format detection only).

Format detection delegates to ``u.Cli`` canonical helpers; no I/O lives here.
"""

from __future__ import annotations

from pathlib import Path

from flext_cli import u
from flext_tests import c, m, t


class FlextTestsFilesUtilitiesMixin:
    """Test-only format-detection facade; delegates to ``u.Cli`` canonical helpers."""

    @staticmethod
    def detect_format(
        content: str
        | bytes
        | m.ConfigMap
        | m.Dict
        | t.MappingKV[str, t.Tests.TestobjectSerializable]
        | t.SequenceOf[t.StrSequence],
        name: str,
        fmt: str,
    ) -> str:
        """Detect format by content shape + name; honors explicit ``fmt``."""
        detected_format: str = u.Cli.files_detect_format_from_content(
            content, name, fmt
        )
        return detected_format

    @staticmethod
    def detect_format_from_path(path: Path, fmt: str) -> str:
        """Detect format from path extension; honors explicit ``fmt``."""
        detected_format: str = u.Cli.files_detect_format_from_path(path, fmt)
        return detected_format

    # NOTE (multi-agent): format_size moved here from c.Tests constants facet
    # (declaration purity - constants must not hold formatting behavior; mro-i6nq.11).
    @staticmethod
    def format_size(size: int) -> str:
        """Format a byte size in human-readable units (e.g. "1.2 KB")."""
        value = size
        for unit in c.Tests.SIZE_UNITS:
            if value < c.Tests.SIZE_THRESHOLD:
                return f"{value:.1f} {unit}" if unit != "B" else f"{value} {unit}"
            value //= c.Tests.SIZE_THRESHOLD
        return f"{value:.1f} PB"
