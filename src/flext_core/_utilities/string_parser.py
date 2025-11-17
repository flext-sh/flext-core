"""Utilities module - FlextUtilitiesStringParser.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from flext_core.result import FlextResult

_logger = logging.getLogger(__name__)


@dataclass
class ParseOptions:
    """Parameter object for parse_delimited configuration."""

    strip: bool = True
    remove_empty: bool = True
    validator: Callable[[str], bool] | None = None


class FlextUtilitiesStringParser:
    r"""String parsing utilities for delimited and structured text processing.

    Provides generic string parsing methods that can be reused across projects
    for handling delimited strings, escaped characters, and whitespace normalization.

    **Usage Examples**:
    >>> # Parse comma-delimited string
    >>> result = FlextUtilitiesStringParser.parse_delimited("a, b, c", ",")
    >>> if result.is_success:
    ...     values = result.unwrap()  # ["a", "b", "c"]

    >>> # Parse with escape character handling
    >>> result = FlextUtilitiesStringParser.split_on_char_with_escape(
    ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\,dc=com", ",", "\\"
    ... )

    >>> # Normalize whitespace
    >>> result = FlextUtilitiesStringParser.normalize_whitespace("  hello   world  ")
    >>> cleaned = result.unwrap()  # "hello world"
    """

    @staticmethod
    def parse_delimited(
        text: str,
        delimiter: str,
        *,
        options: ParseOptions | None = None,
        strip: bool = True,
        remove_empty: bool = True,
        validator: Callable[[str], bool] | None = None,
    ) -> FlextResult[list[str]]:
        """Parse delimited string into list of components.

        **Generic replacement for**: DN.split(), CSV parsing, config parsing

        Args:
            text: String to parse
            delimiter: Delimiter character/string
            options: ParseOptions object (preferred). If provided, overrides other params
            strip: Strip whitespace from each component (legacy, use options)
            remove_empty: Remove empty components after stripping (legacy, use options)
            validator: Optional validation function for each component (legacy, use options)

        Returns:
            FlextResult with list of parsed components or error

        Example:
            >>> # NEW - Using ParseOptions
            >>> opts = ParseOptions(strip=True, remove_empty=True)
            >>> result = FlextUtilitiesStringParser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com", ",", options=opts
            ... )
            >>> components = result.unwrap()
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD", "ou=users", "dc=example", "dc=com"]

            >>> # OLD - Backward compatible
            >>> result = FlextUtilitiesStringParser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com", ","
            ... )
            >>> components = result.unwrap()

        """
        # Use options if provided, otherwise use individual params for backward compatibility
        if options is not None:
            strip = options.strip
            remove_empty = options.remove_empty
            validator = options.validator

        if not text:
            return FlextResult[list[str]].ok([])

        try:
            components = text.split(delimiter)

            if strip:
                components = [c.strip() for c in components]

            if remove_empty:
                components = [c for c in components if c]

            if validator:
                for comp in components:
                    if not validator(comp):
                        return FlextResult[list[str]].fail(f"Invalid component: {comp}")

            return FlextResult[list[str]].ok(components)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[list[str]].fail(f"Failed to parse delimited string: {e}")

    @staticmethod
    def split_on_char_with_escape(
        text: str,
        split_char: str,
        escape_char: str = "\\",
    ) -> FlextResult[list[str]]:
        r"""Split string on character, respecting escape sequences.

        **Generic replacement for**: DN parsing with escapes, CSV with quotes

        Args:
            text: String to split
            split_char: Character to split on
            escape_char: Escape character (default: backslash)

        Returns:
            FlextResult with list of split components or error

        Example:
            >>> # Parse DN with escaped commas
            >>> result = FlextUtilitiesStringParser.split_on_char_with_escape(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\,user,ou=users", ","
            ... )
            >>> parts = result.unwrap()
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD\\,user", "ou=users"]

        """
        if not text:
            return FlextResult[list[str]].ok([])

        try:
            components: list[str] = []
            current: list[str] = []
            i = 0

            while i < len(text):
                if text[i] == escape_char and i + 1 < len(text):
                    # Add escape sequence as-is
                    current.extend((text[i], text[i + 1]))
                    i += 2
                elif text[i] == split_char:
                    # Found unescaped delimiter
                    components.append("".join(current))
                    current = []
                    i += 1
                else:
                    current.append(text[i])
                    i += 1

            # Add final component
            if current:
                components.append("".join(current))

            return FlextResult[list[str]].ok(components)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[list[str]].fail(f"Failed to split with escape: {e}")

    @staticmethod
    def normalize_whitespace(
        text: str,
        pattern: str = r"\s+",
        replacement: str = " ",
    ) -> FlextResult[str]:
        r"""Normalize whitespace in text using regex pattern.

        **Generic replacement for**: Multiple spaces → single space normalization

        Args:
            text: Text to normalize
            pattern: Regex pattern to match (default: one or more whitespace)
            replacement: Replacement string (default: single space)

        Returns:
            FlextResult with normalized text or error

        Example:
            >>> result = FlextUtilitiesStringParser.normalize_whitespace(
            ...     "hello    world\\t\\nfoo"
            ... )
            >>> normalized = result.unwrap()  # "hello world foo"

        """
        if not text:
            return FlextResult[str].ok(text)

        try:
            normalized = re.sub(pattern, replacement, text).strip()
            return FlextResult[str].ok(normalized)

        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
            re.error,
        ) as e:
            return FlextResult[str].fail(f"Failed to normalize whitespace: {e}")

    @staticmethod
    def apply_regex_pipeline(
        text: str,
        patterns: list[tuple[str, str]],
    ) -> FlextResult[str]:
        r"""Apply sequence of regex substitutions to text.

        **Generic replacement for**: Multiple regex.sub() calls

        Args:
            text: Text to transform
            patterns: List of (pattern, replacement) tuples

        Returns:
            FlextResult with transformed text or error

        Example:
            >>> patterns = [
            ...     (r"\\s+=", "="),  # Remove spaces before =
            ...     (r",\\s+", ","),  # Remove spaces after ,
            ...     (r"\\s+", " "),  # Normalize whitespace
            ... ]
            >>> result = FlextUtilitiesStringParser.apply_regex_pipeline(
            ...     "cn = REDACTED_LDAP_BIND_PASSWORD , ou = users", patterns
            ... )
            >>> cleaned = result.unwrap()  # "cn=REDACTED_LDAP_BIND_PASSWORD,ou=users"

        """
        if not text:
            return FlextResult[str].ok(text)

        try:
            result_text = text
            for pattern, replacement in patterns:
                result_text = re.sub(pattern, replacement, result_text)

            return FlextResult[str].ok(result_text.strip())

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[str].fail(f"Failed to apply regex pipeline: {e}")

    @staticmethod
    def get_object_key(obj: object) -> str:
        """Get comparable string key from object (generic helper).

        This generic helper consolidates object-to-key conversion logic from
        dispatcher.py (_normalize_command_key) and provides flexible key extraction
        strategies for objects.

        Extraction Strategy:
            1. Try __name__ attribute (for types, classes, functions)
            2. Try str(obj)
            3. Use type name as final strategy

        Args:
            obj: Object to extract key from (type, class, instance, etc.)

        Returns:
            String key for object (comparable, hashable)

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> # Class/Type
            >>> FlextUtilities.StringParser.get_object_key(int)
            'int'
            >>> # Function
            >>> def my_func():
            ...     pass
            >>> FlextUtilities.StringParser.get_object_key(my_func)
            'my_func'
            >>> # Instance
            >>> obj = object()
            >>> key = FlextUtilities.StringParser.get_object_key(obj)
            >>> isinstance(key, str)
            True

        """
        # Strategy 1: Try __name__ attribute (classes, types, functions)
        name_attr = getattr(obj, "__name__", None)
        if name_attr is not None:
            return str(name_attr)

        # Strategy 2: Use str(obj)
        try:
            return str(obj)
        except (TypeError, ValueError, AttributeError):
            pass

        # Strategy 3: Use type name
        return type(obj).__name__


__all__ = ["FlextUtilitiesStringParser"]
