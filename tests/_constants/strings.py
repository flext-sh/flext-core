"""Constants mixin for strings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class TestsFlextConstantsStrings:
    class Strings:
        """Flext-core-specific test strings organized by complexity."""

        EMPTY: Final[str] = ""
        SINGLE_CHAR: Final[str] = "a"
        BASIC_WORD: Final[str] = "hello"
        BASIC_LIST: Final[str] = "a,b,c"
        NUMERIC_LIST: Final[str] = "1,2,3"
        WITH_SPACES: Final[str] = "a, b, c"
        EXCESSIVE_SPACES: Final[str] = "  a  ,  b  ,  c  "
        LEADING_SPACES: Final[str] = "  hello"
        TRAILING_SPACES: Final[str] = "hello  "
        LEADING_TRAILING: Final[str] = ",a,b,c,"
        WITH_EMPTY: Final[str] = "a,,c"
        ONLY_DELIMITERS: Final[str] = ",,,"
        UNICODE_CHARS: Final[str] = "héllo,wörld"
        VALID_EMAIL: Final[str] = "test@example.com"
        INVALID_EMAIL: Final[str] = "invalid-email"
        USER_ID_VALID: Final[str] = "123"
        USER_ID_INVALID: Final[str] = "invalid"
        USER_ID_EMPTY: Final[str] = ""

    class Delimiters:
        """Flext-core-specific delimiter characters for string parsing."""

        COMMA: Final[str] = ","
        SEMICOLON: Final[str] = ";"
        PIPE: Final[str] = "|"
        COLON: Final[str] = ":"
        TAB: Final[str] = "\t"
        NEWLINE: Final[str] = "\n"

    class EscapeChars:
        """Flext-core-specific escape characters for string parsing."""

        BACKSLASH: Final[str] = "\\"
        HASH: Final[str] = "#"
        AT: Final[str] = "@"
        QUOTE: Final[str] = '"'
        SINGLE_QUOTE: Final[str] = "'"

    class Replacements:
        """Flext-core-specific replacement strings for string processing."""

        SPACE: Final[str] = " "
        UNDERSCORE: Final[str] = "_"
        DASH: Final[str] = "-"
        EQUALS: Final[str] = "="
        COMMA: Final[str] = ","
        EMPTY: Final[str] = ""

    class Patterns:
        """Flext-core-specific regex patterns for string processing."""

        WHITESPACE: Final[str] = "\\s+"
        DASH: Final[str] = "-+"
        EQUALS_SPACE: Final[str] = "\\s+="
        COMMA_SPACE: Final[str] = ",\\s+"
        EMAIL: Final[str] = "^[^@]+@[^@]+\\.[^@]+$"
        ALPHA_ONLY: Final[str] = "^[a-zA-Z]+$"
        NUMERIC_ONLY: Final[str] = "^\\d+$"
        SEMVER: Final[str] = "^\\d+\\.\\d+\\.\\d+"
