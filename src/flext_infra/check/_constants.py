"""Centralized constants for the check subpackage."""

from __future__ import annotations

import re
from typing import Final


class FlextInfraCheckConstants:
    """Check infrastructure constants."""

    REQUIRED_EXCLUDES: Final[list[str]] = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]
    RUFF_FORMAT_FILE_RE: Final[re.Pattern[str]] = re.compile(
        r"^\s*-->\s*(.+?):\d+:\d+\s*$"
    )
    MARKDOWN_RE: Final[re.Pattern[str]] = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+)(?::(?P<col>\d+))?\s+error\s+"
        r"(?P<code>MD\d+)(?:/[^\s]+)?\s+(?P<msg>.*)$",
    )
    GO_VET_RE: Final[re.Pattern[str]] = re.compile(
        r"^(?P<file>[^:\n]+\.go):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$",
    )
    MAX_DISPLAY_ISSUES: Final[int] = 50


__all__ = ["FlextInfraCheckConstants"]
