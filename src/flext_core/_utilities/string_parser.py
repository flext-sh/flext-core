"""Utilities module - FlextUtilitiesStringParser.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult


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

    **Architecture**: Detailed structured logging following FlextLdifParser pattern
    - debug: Processing steps and intermediate results
    - warning: Non-fatal issues with context
    - error: Fatal errors with detailed context and abort
    - info: Major operations and performance metrics

    **Usage Examples**:
    >>> # Parse comma-delimited string
    >>> parser = FlextUtilitiesStringParser()
    >>> result = parser.parse_delimited("a, b, c", ",")
    >>> if result.is_success:
    ...     values = result.unwrap()  # ["a", "b", "c"]

    >>> # Parse with escape character handling
    >>> result = parser.split_on_char_with_escape("cn=admin\\,dc=com", ",", "\\")

    >>> # Normalize whitespace
    >>> result = parser.normalize_whitespace("  hello   world  ")
    >>> cleaned = result.unwrap()  # "hello world"
    """

    def __init__(self) -> None:
        """Initialize string parser with logging."""
        self.logger = FlextLogger.create_module_logger(__name__)

    def _safe_text_length(self, text: object) -> str | int:
        """Safely get text length for logging."""
        try:
            if isinstance(text, (str, bytes)):
                return len(text)
            return "unknown"
        except (TypeError, AttributeError):
            return "unknown"

    def _process_components(
        self,
        components: list[str],
        *,
        strip: bool,
        remove_empty: bool,
        validator: Callable[[str], bool] | None,
    ) -> FlextResult[list[str]]:
        """Process components with strip, remove_empty, and validator."""
        if strip:
            self.logger.debug(
                "Stripping whitespace from components",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            components = [c.strip() for c in components]

        if remove_empty:
            self.logger.debug(
                "Removing empty components",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            components = [c for c in components if c]

        if validator:
            self.logger.debug(
                "Validating components with custom validator",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            for comp in components:
                if not validator(comp):
                    self.logger.warning(
                        "Component validation failed",
                        operation="parse_delimited",
                        invalid_component=comp,
                        validator_type=type(validator).__name__,
                        source="flext-core/src/flext_core/_utilities/string_parser.py",
                    )
                    return FlextResult[list[str]].fail(f"Invalid component: {comp}")

        return FlextResult[list[str]].ok(components)

    def parse_delimited(
        self,
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
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.parse_delimited(
            ...     "cn=admin, ou=users, dc=example, dc=com", ",", options=opts
            ... )
            >>> components = result.unwrap()
            >>> # ["cn=admin", "ou=users", "dc=example", "dc=com"]

            >>> # OLD - Backward compatible
            >>> result = parser.parse_delimited(
            ...     "cn=admin, ou=users, dc=example, dc=com", ","
            ... )
            >>> components = result.unwrap()

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        self.logger.debug(
            "Starting delimited string parsing",
            operation="parse_delimited",
            text_length=text_len,
            delimiter=delimiter,
            has_options=options is not None,
            strip=strip,
            remove_empty=remove_empty,
            has_validator=validator is not None,
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        # Use options if provided, otherwise use individual params for backward compatibility
        if options is not None:
            strip = options.strip
            remove_empty = options.remove_empty
            validator = options.validator

        if not text:
            self.logger.debug(
                "Empty text provided, returning empty list",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].ok([])

        try:
            self.logger.debug(
                "Splitting text by delimiter",
                operation="parse_delimited",
                delimiter=delimiter,
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            components = text.split(delimiter)

            self.logger.debug(
                "Initial split completed",
                operation="parse_delimited",
                raw_components_count=len(components),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            # Process components (strip, remove_empty, validate)
            result = self._process_components(
                components,
                strip=strip,
                remove_empty=remove_empty,
                validator=validator,
            )

            if result.is_failure:
                return result

            components = result.unwrap()

            self.logger.debug(
                "Delimited parsing completed successfully",
                operation="parse_delimited",
                final_components_count=len(components),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            return FlextResult[list[str]].ok(components)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Safely get text length (may fail for non-string objects in tests)
            try:
                text_len = self._safe_text_length(text)
            except (TypeError, AttributeError):
                text_len = -1  # Unknown length

            self.logger.exception(
                "FATAL ERROR during delimited parsing - PARSING ABORTED",
                operation="parse_delimited",
                error=str(e),
                error_type=type(e).__name__,
                text_length=text_len,
                delimiter=delimiter,
                consequence="Cannot parse delimited string - invalid input or internal error",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].fail(f"Failed to parse delimited string: {e}")

    def split_on_char_with_escape(
        self,
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
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.split_on_char_with_escape(
            ...     "cn=admin\\,user,ou=users", ","
            ... )
            >>> parts = result.unwrap()
            >>> # ["cn=admin\\,user", "ou=users"]

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        self.logger.debug(
            "Starting escape-aware string splitting",
            operation="split_on_char_with_escape",
            text_length=text_len,
            split_char=split_char,
            escape_char=escape_char,
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning empty list",
                operation="split_on_char_with_escape",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].ok([])

        try:
            self.logger.debug(
                "Processing text with escape character handling",
                operation="split_on_char_with_escape",
                text_length=self._safe_text_length(text),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            components: list[str] = []
            current: list[str] = []
            i = 0
            escape_count = 0

            while i < len(text):
                if text[i] == escape_char and i + 1 < len(text):
                    # Add escape sequence as-is
                    self.logger.debug(
                        "Found escape sequence",
                        operation="split_on_char_with_escape",
                        position=i,
                        escaped_char=text[i + 1],
                        source="flext-core/src/flext_core/_utilities/string_parser.py",
                    )
                    current.extend((text[i], text[i + 1]))
                    escape_count += 1
                    i += 2
                elif text[i] == split_char:
                    # Found unescaped delimiter
                    self.logger.debug(
                        "Found unescaped delimiter",
                        operation="split_on_char_with_escape",
                        position=i,
                        current_component_length=len(current),
                        source="flext-core/src/flext_core/_utilities/string_parser.py",
                    )
                    components.append("".join(current))
                    current = []
                    i += 1
                else:
                    current.append(text[i])
                    i += 1

            # Add final component
            if current:
                self.logger.debug(
                    "Adding final component",
                    operation="split_on_char_with_escape",
                    final_component_length=len(current),
                    source="flext-core/src/flext_core/_utilities/string_parser.py",
                )
                components.append("".join(current))

            self.logger.debug(
                "Escape-aware splitting completed successfully",
                operation="split_on_char_with_escape",
                components_count=len(components),
                escape_sequences_found=escape_count,
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            return FlextResult[list[str]].ok(components)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Safely get text length (may fail for non-string objects in tests)
            try:
                text_len = self._safe_text_length(text)
            except (TypeError, AttributeError):
                text_len = -1  # Unknown length

            self.logger.exception(
                "FATAL ERROR during escape-aware splitting - SPLITTING ABORTED",
                operation="split_on_char_with_escape",
                error=str(e),
                error_type=type(e).__name__,
                text_length=text_len,
                split_char=split_char,
                escape_char=escape_char,
                consequence="Cannot split string with escape handling - invalid input or internal error",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].fail(f"Failed to split with escape: {e}")

    def normalize_whitespace(
        self,
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
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.normalize_whitespace("hello    world\\t\\nfoo")
            >>> normalized = result.unwrap()  # "hello world foo"

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        self.logger.debug(
            "Starting whitespace normalization",
            operation="normalize_whitespace",
            text_length=text_len,
            pattern=pattern,
            replacement=replacement,
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning unchanged",
                operation="normalize_whitespace",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[str].ok(text)

        try:
            self.logger.debug(
                "Applying regex pattern for whitespace normalization",
                operation="normalize_whitespace",
                pattern=pattern,
                replacement=replacement,
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            normalized = re.sub(pattern, replacement, text).strip()

            self.logger.debug(
                "Whitespace normalization completed",
                operation="normalize_whitespace",
                original_length=len(text),
                normalized_length=len(normalized),
                replacements_made=len(text) - len(normalized),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            return FlextResult[str].ok(normalized)

        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
            re.error,
        ) as e:
            self.logger.exception(
                "FATAL ERROR during whitespace normalization - NORMALIZATION ABORTED",
                operation="normalize_whitespace",
                error=str(e),
                error_type=type(e).__name__,
                pattern=pattern,
                replacement=replacement,
                consequence="Cannot normalize whitespace - invalid pattern or internal error",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[str].fail(f"Failed to normalize whitespace: {e}")

    def apply_regex_pipeline(
        self,
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
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.apply_regex_pipeline(
            ...     "cn = admin , ou = users", patterns
            ... )
            >>> cleaned = result.unwrap()  # "cn=admin,ou=users"

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        self.logger.debug(
            "Starting regex pipeline application",
            operation="apply_regex_pipeline",
            text_length=text_len,
            patterns_count=len(patterns),
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning unchanged",
                operation="apply_regex_pipeline",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[str].ok(text)

        if not patterns:
            self.logger.warning(
                "No patterns provided for regex pipeline",
                operation="apply_regex_pipeline",
                text_length=self._safe_text_length(text),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[str].ok(text)

        try:
            self.logger.debug(
                "Applying regex patterns sequentially",
                operation="apply_regex_pipeline",
                patterns_count=len(patterns),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            result_text = text
            applied_patterns = 0

            for i, (pattern, replacement) in enumerate(patterns):
                self.logger.debug(
                    "Applying regex pattern",
                    operation="apply_regex_pipeline",
                    pattern_index=i + 1,
                    total_patterns=len(patterns),
                    pattern=pattern,
                    replacement=replacement,
                    source="flext-core/src/flext_core/_utilities/string_parser.py",
                )

                before_length = len(result_text)
                result_text = re.sub(pattern, replacement, result_text)
                after_length = len(result_text)
                replacements = before_length - after_length

                self.logger.debug(
                    "Pattern applied",
                    operation="apply_regex_pipeline",
                    pattern_index=i + 1,
                    replacements_made=replacements,
                    source="flext-core/src/flext_core/_utilities/string_parser.py",
                )

                applied_patterns += 1

            final_result = result_text.strip()

            self.logger.debug(
                "Regex pipeline completed successfully",
                operation="apply_regex_pipeline",
                patterns_applied=applied_patterns,
                original_length=len(text),
                final_length=len(final_result),
                total_replacements=len(text) - len(final_result),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            return FlextResult[str].ok(final_result)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Safely get text length (may fail for non-string objects in tests)
            try:
                text_len = self._safe_text_length(text)
            except (TypeError, AttributeError):
                text_len = -1  # Unknown length

            self.logger.exception(
                "FATAL ERROR during regex pipeline application - PIPELINE ABORTED",
                operation="apply_regex_pipeline",
                error=str(e),
                error_type=type(e).__name__,
                patterns_count=len(patterns),
                text_length=text_len,
                consequence="Cannot apply regex transformations - invalid pattern or internal error",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[str].fail(f"Failed to apply regex pipeline: {e}")

    def get_object_key(self, obj: object) -> str:
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
            >>> parser = FlextUtilities.StringParser()
            >>> # Class/Type
            >>> parser.get_object_key(int)
            'int'
            >>> # Function
            >>> def my_func():
            ...     pass
            >>> parser.get_object_key(my_func)
            'my_func'
            >>> # Instance
            >>> obj = object()
            >>> key = parser.get_object_key(obj)
            >>> isinstance(key, str)
            True

        """
        self.logger.debug(
            "Starting object key extraction",
            operation="get_object_key",
            obj_type=type(obj).__name__,
            has_name_attr=hasattr(obj, "__name__"),
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        # Strategy 1: Try __name__ attribute (classes, types, functions)
        name_attr = getattr(obj, "__name__", None)
        if name_attr is not None:
            self.logger.debug(
                "Using __name__ attribute for key",
                operation="get_object_key",
                key=name_attr,
                strategy="name_attribute",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return str(name_attr)

        # Strategy 2: Use str(obj)
        try:
            str_key = str(obj)
            self.logger.debug(
                "Using str() representation for key",
                operation="get_object_key",
                key=str_key,
                strategy="str_conversion",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return str_key
        except (TypeError, ValueError, AttributeError) as e:
            self.logger.warning(
                "str() conversion failed, falling back to type name",
                operation="get_object_key",
                error=str(e),
                error_type=type(e).__name__,
                strategy="fallback",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

        # Strategy 3: Use type name
        type_name = type(obj).__name__
        self.logger.debug(
            "Using type name for key",
            operation="get_object_key",
            key=type_name,
            strategy="type_name",
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )
        return type_name


__all__ = ["FlextUtilitiesStringParser"]
