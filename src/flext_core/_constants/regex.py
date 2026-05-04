"""FlextConstantsRegex - regex pattern constants (SSOT).

Owns every workspace-wide fixed compiled ``re.Pattern``. Consumer modules
import the pre-compiled ``*_RE`` constants directly; runtime-supplied regex
construction routes through the single ``compile_pattern`` entry-point here.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import ClassVar, Final


class FlextConstantsRegex:
    """SSOT for regex pattern string constants used across the workspace."""

    PATTERN_ENFORCE_RULE_ID: Final[str] = r"^ENFORCE-\d{3}$"
    "Pattern for canonical enforcement rule identifiers."
    PATTERN_SEMVER: Final[str] = r"^\d+\.\d+\.\d+"
    "Pattern for semantic-version strings with major.minor.patch prefix."
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
    PATTERN_EXAMPLE_RESULT_LINE: Final[str] = r"^[^\[][^\n]+: .+$"
    "Matches one normalized PASS/FAIL/GENERATED summary line emitted by examples."

    # === Pre-compiled regex authorities (consumers MUST use these) ===
    PATTERN_ENFORCE_RULE_ID_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_ENFORCE_RULE_ID
    )
    PATTERN_SEMVER_RE: ClassVar[re.Pattern[str]] = re.compile(PATTERN_SEMVER)
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
    PATTERN_EXAMPLE_RESULT_LINE_RE: ClassVar[re.Pattern[str]] = re.compile(
        PATTERN_EXAMPLE_RESULT_LINE
    )
