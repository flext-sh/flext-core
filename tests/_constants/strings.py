"""Constants mixin for strings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import Final


class TestsFlextConstantsStrings:
    # String test values
    STR_EMPTY: Final[str] = ""
    STR_SINGLE_CHAR: Final[str] = "a"
    STR_BASIC_WORD: Final[str] = "hello"
    STR_BASIC_LIST: Final[str] = "a,b,c"
    STR_NUMERIC_LIST: Final[str] = "1,2,3"
    STR_WITH_SPACES: Final[str] = "a, b, c"
    STR_EXCESSIVE_SPACES: Final[str] = "  a  ,  b  ,  c  "
    STR_LEADING_SPACES: Final[str] = "  hello"
    STR_TRAILING_SPACES: Final[str] = "hello  "
    STR_LEADING_TRAILING: Final[str] = ",a,b,c,"
    STR_WITH_EMPTY: Final[str] = "a,,c"
    STR_ONLY_DELIMITERS: Final[str] = ",,,"
    STR_UNICODE_CHARS: Final[str] = "héllo,wörld"
    STR_VALID_EMAIL: Final[str] = "test@example.com"
    STR_INVALID_EMAIL: Final[str] = "invalid-email"
    STR_USER_ID_VALID: Final[str] = "123"
    STR_USER_ID_INVALID: Final[str] = "invalid"
    STR_USER_ID_EMPTY: Final[str] = ""

    # Delimiter characters
    DELIM_COMMA: Final[str] = ","
    DELIM_SEMICOLON: Final[str] = ";"
    DELIM_PIPE: Final[str] = "|"
    DELIM_COLON: Final[str] = ":"
    DELIM_TAB: Final[str] = "\t"
    DELIM_NEWLINE: Final[str] = "\n"

    # Escape characters
    ESCAPE_BACKSLASH: Final[str] = "\\"
    ESCAPE_HASH: Final[str] = "#"
    ESCAPE_AT: Final[str] = "@"
    ESCAPE_QUOTE: Final[str] = '"'
    ESCAPE_SINGLE_QUOTE: Final[str] = "'"

    # Replacement strings
    REPLACE_SPACE: Final[str] = " "
    REPLACE_UNDERSCORE: Final[str] = "_"
    REPLACE_DASH: Final[str] = "-"
    REPLACE_EQUALS: Final[str] = "="
    REPLACE_COMMA: Final[str] = ","
    REPLACE_EMPTY: Final[str] = ""

    # Compiled regex patterns
    PATTERN_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
    PATTERN_DASH: Final[re.Pattern[str]] = re.compile(r"-+")
    PATTERN_EQUALS_SPACE: Final[re.Pattern[str]] = re.compile(r"\s+=")
    PATTERN_COMMA_SPACE: Final[re.Pattern[str]] = re.compile(r",\s+")
    PATTERN_EMAIL: Final[re.Pattern[str]] = re.compile(r"^[^@]+@[^@]+\.[^@]+$")
    PATTERN_ALPHA_ONLY: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z]+$")
    PATTERN_NUMERIC_ONLY: Final[re.Pattern[str]] = re.compile(r"^\d+$")
    PATTERN_SEMVER: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+")
