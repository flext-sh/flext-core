"""FlextConstantsRegex - regex pattern constants (SSOT).

Owns every workspace-wide compiled ``re.Pattern``. Consumer modules
import the pre-compiled ``*_RE`` constants directly; ``import re``
outside this module (or another constants module) is forbidden by
AGENTS.md §3.1 ``regex-from-constants`` rule.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import ClassVar, Final


class FlextConstantsRegex:
    """SSOT for regex pattern string constants used across the workspace."""

    PATTERN_IDENTIFIER_WITH_UNDERSCORE: Final[str] = "^[a-zA-Z_][a-zA-Z0-9_]*$"
    "Pattern for identifiers that can start with underscore (context keys)."
    PATTERN_ISO8601_TIMESTAMP: Final[str] = (
        "^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}[Z+\\-][0-9:]*)?$"
    )
    "Pattern for ISO 8601 timestamps (optional, allows empty string)."
    PATTERN_HOSTNAME_OR_IP: Final[str] = (
        "^[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$|^localhost$|^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$"
    )
    "Pattern for hostnames (domain names) or IPv4 addresses."
    PATTERN_LDAP_DN: Final[str] = (
        "^[a-zA-Z][\\w\\-]*\\s*=\\s*[^,]+(?:\\s*,\\s*[a-zA-Z][\\w\\-]*\\s*=\\s*[^,]+)*$"
    )
    "Pattern for LDAP Distinguished Names (DN)."
    PATTERN_IDENTIFIER_LOWERCASE: Final[str] = (
        "^[a-z0-9](?:[a-z0-9\\-_.]{0,62}[a-z0-9])?$"
    )
    "Pattern for lowercase identifiers with optional hyphens, underscores, and dots (max 64 chars)."
    PATTERN_CAMEL_TO_SNAKE: Final[str] = r"([a-z0-9])([A-Z])"
    "Boundary used to insert underscores when converting camelCase → snake_case."
    PATTERN_FORBIDDEN_FACADE_IMPORT: Final[str] = (
        r"^\s*from\s+(tests|examples|scripts)\.([\w.]+)\s+import\s+([\w,\s]+?)\s*$"
    )
    "Matches `from <forbidden>.<module> import …` lines (multiline source scan)."

    # === Pre-compiled regex authorities (consumers MUST use these) ===
    PATTERN_IDENTIFIER_WITH_UNDERSCORE_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_IDENTIFIER_WITH_UNDERSCORE
    )
    PATTERN_ISO8601_TIMESTAMP_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_ISO8601_TIMESTAMP
    )
    PATTERN_HOSTNAME_OR_IP_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_HOSTNAME_OR_IP
    )
    PATTERN_LDAP_DN_RE: ClassVar[re.Pattern[str]] = re.compile(PATTERN_LDAP_DN)
    PATTERN_IDENTIFIER_LOWERCASE_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_IDENTIFIER_LOWERCASE
    )
    CAMEL_TO_SNAKE_RE: ClassVar[re.Pattern[str]] = re.compile(PATTERN_CAMEL_TO_SNAKE)
    FORBIDDEN_FACADE_IMPORT_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_FORBIDDEN_FACADE_IMPORT, flags=re.MULTILINE
    )

    @staticmethod
    def compile_pattern(
        pattern: str,
        *,
        ignorecase: bool = False,
        multiline: bool = False,
        dotall: bool = False,
    ) -> re.Pattern[str]:
        """Compile a runtime-supplied regex pattern.

        Sole sanctioned ``re.compile`` entry-point for non-constant
        patterns workspace-wide. Consumer modules MUST call this
        instead of importing ``re`` directly.
        """
        flags = 0
        if ignorecase:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        if dotall:
            flags |= re.DOTALL
        return re.compile(pattern, flags=flags)
