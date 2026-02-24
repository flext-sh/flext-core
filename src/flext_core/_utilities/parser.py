"""String parsing helpers for deterministic CQRS utility flows.

These helpers centralize delimiter handling, whitespace normalization, and
escaped character parsing so dispatcher handlers and services receive
predictable ``FlextResult`` outcomes instead of ad-hoc string handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import StrEnum
from types import NoneType
from typing import overload

import structlog
from pydantic import BaseModel

from flext_core._models.collections import FlextModelsCollections
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core.constants import c
from flext_core.models import m
from flext_core.result import r
from flext_core.typings import t


class FlextUtilitiesParser:
    r"""Parse delimited and structured strings with predictable results.

    The parser consolidates delimiter handling, escape-aware splits, and
    normalization routines behind ``FlextResult`` so callers can compose
    parsing logic in dispatcher pipelines without manual error handling.

    Examples:
        >>> parser = FlextUtilitiesParser()
        >>> parser.parse_delimited("a, b, c", ",").value
        ['a', 'b', 'c']
        >>> parser.split_on_char_with_escape(
        ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\,dc=com", ",", "\\"
        ... ).value
        ['cn=REDACTED_LDAP_BIND_PASSWORD', 'dc=com']

    """

    # Use centralized constants from FlextConstants
    PATTERN_TUPLE_MIN_LENGTH: int = c.Processing.PATTERN_TUPLE_MIN_LENGTH
    PATTERN_TUPLE_MAX_LENGTH: int = c.Processing.PATTERN_TUPLE_MAX_LENGTH

    # Magic value constants to reduce complexity
    TUPLE_LENGTH_2: int = 2
    TUPLE_LENGTH_3: int = 3

    def __init__(self) -> None:
        """Initialize string parser with logging."""
        super().__init__()
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def _safe_text_length(text: t.ConfigMapValue) -> str | int:
        """Safely get text length for logging."""
        try:
            if text.__class__ in {str, bytes}:
                len_method = getattr(text, "__len__", None)
                if callable(len_method):
                    return len_method()
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
    ) -> r[list[str]]:
        """Process components with strip, remove_empty, and validator."""
        if strip:
            self.logger.debug(
                "Stripping whitespace from components",
                operation="parse_delimited",
            )
            components = [c.strip() for c in components]

        if remove_empty:
            self.logger.debug(
                "Removing empty components",
                operation="parse_delimited",
            )
            # NOTE: Cannot use u.filter() here due to circular import
            components = [c for c in components if c.strip()]

        if validator:
            self.logger.debug(
                "Validating components with custom validator",
                operation="parse_delimited",
            )
            # Filter out invalid components instead of failing
            valid_components: list[str] = []
            for comp in components:
                if validator(comp):
                    valid_components.append(comp)
                else:
                    self.logger.debug(
                        "Component filtered out by validator",
                        operation="parse_delimited",
                        invalid_component=comp,
                        validator_type=validator.__class__.__name__,
                    )
            components = valid_components

        return r[list[str]].ok(components)

    def parse_delimited(
        self,
        text: str,
        delimiter: str,
        *,
        options: FlextModelsCollections.ParseOptions | None = None,
    ) -> r[list[str]]:
        """Parse delimited string into list of components.

        **Generic replacement for**: DN.split(), CSV parsing, config parsing

        Args:
            text: String to parse
            delimiter: Delimiter character/string
            options: ParseOptions object with parsing configuration

        Returns:
            FlextResult with list of parsed components or error

        Example:
            >>> from flext_core._models.collections import FlextModelsCollections
            >>> opts = FlextModelsCollections.ParseOptions(
            ...     strip=True, remove_empty=True
            ... )
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com",
            ...     ",",
            ...     options=opts,
            ... )
            >>> components = result.value
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD", "ou=users", "dc=example", "dc=com"]

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        # Use provided ParseOptions or create default
        parse_opts = (
            options if options is not None else FlextModelsCollections.ParseOptions()
        )

        strip = parse_opts.strip
        remove_empty = parse_opts.remove_empty
        validator = parse_opts.validator

        self.logger.debug(
            "Starting delimited string parsing",
            operation="parse_delimited",
            text_length=text_len,
            delimiter=delimiter,
            has_options=options is not None,
            strip=strip,
            remove_empty=remove_empty,
            has_validator=validator is not None,
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning empty list",
                operation="parse_delimited",
            )
            return r[list[str]].ok([])

        # Validate delimiter
        if not delimiter or len(delimiter) != 1:
            return r[list[str]].fail(
                f"Delimiter must be exactly one character, got '{delimiter}'",
            )

        # Reject whitespace/control characters as delimiters
        if delimiter.isspace() or not delimiter.isprintable():
            return r[list[str]].fail(
                f"Delimiter cannot be a whitespace or control character: '{delimiter}'",
            )

        try:
            self.logger.debug(
                "Splitting text by delimiter",
                operation="parse_delimited",
                delimiter=delimiter,
            )
            components = text.split(delimiter)

            self.logger.debug(
                "Initial split completed",
                operation="parse_delimited",
                raw_components_count=len(components),
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

            # Use .value directly - FlextResult never returns None on success
            components = result.value

            self.logger.debug(
                "Delimited parsing completed successfully",
                operation="parse_delimited",
                final_components_count=len(components),
            )

            return r[list[str]].ok(components)

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
                error_type=e.__class__.__name__,
                text_length=text_len,
                delimiter=delimiter,
                consequence="Cannot parse delimited string - invalid input or internal error",
            )
            return r[list[str]].fail(f"Failed to parse delimited string: {e}")

    @staticmethod
    def _validate_split_inputs(
        split_char: str,
        escape_char: str,
    ) -> r[bool]:
        """Validate inputs for split operation.

        Args:
            split_char: Character to split on
            escape_char: Escape character

        Returns:
            r[bool]: True if valid, failure with error message

        """
        if not split_char:
            return r[bool].fail("Split character cannot be empty")
        if not escape_char:
            return r[bool].fail("Escape character cannot be empty")
        if split_char == escape_char:
            return r[bool].fail(
                "Split character and escape character cannot be the same",
            )
        return r[bool].ok(value=True)

    def _get_safe_text_length(self, text: str) -> int:
        """Get text length safely, handling non-string objects in tests.

        Args:
            text: Text to measure

        Returns:
            Text length or -1 if measurement fails

        """
        try:
            length = self._safe_text_length(text)
            if length.__class__ is int:
                return length
            return int(length) if length != "unknown" else -1
        except (TypeError, AttributeError, ValueError):
            return -1  # Unknown length

    def _execute_escape_splitting(
        self,
        text: str,
        split_char: str,
        escape_char: str,
    ) -> r[list[str]]:
        """Execute escape-aware splitting with logging and error handling.

        Args:
            text: String to split
            split_char: Character to split on
            escape_char: Escape character

        Returns:
            FlextResult with list of split components or error

        """
        # Safely get text length for logging
        text_len = self._get_safe_text_length(text)

        self.logger.debug(
            "Starting escape-aware string splitting",
            operation="split_on_char_with_escape",
            text_length=text_len,
            split_char=split_char,
            escape_char=escape_char,
        )

        try:
            self.logger.debug(
                "Processing text with escape character handling",
                operation="split_on_char_with_escape",
                text_length=text_len,
            )

            # Process the text and extract components
            split_result = self._process_escape_splitting(text, split_char, escape_char)
            if split_result.is_failure:
                return r[list[str]].fail(
                    split_result.error or "Unknown error in escape splitting",
                )

            # Use .value directly - FlextResult never returns None on success
            components, escape_count = split_result.value

            self.logger.debug(
                "Escape-aware splitting completed successfully",
                operation="split_on_char_with_escape",
                components_count=len(components),
                escape_sequences_found=escape_count,
            )

            return r[list[str]].ok(components)

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            text_len = self._get_safe_text_length(text)
            self.logger.exception(
                "FATAL ERROR during escape-aware splitting - SPLITTING ABORTED",
                operation="split_on_char_with_escape",
                error=str(e),
                error_type=e.__class__.__name__,
                text_length=text_len,
                split_char=split_char,
                escape_char=escape_char,
                consequence="Cannot split string with escape handling - invalid input or internal error",
            )
            return r[list[str]].fail(f"Failed to split with escape: {e}")

    def split_on_char_with_escape(
        self,
        text: str,
        split_char: str,
        escape_char: str = "\\",
    ) -> r[list[str]]:
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
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.split_on_char_with_escape(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\,user,ou=users", ","
            ... )
            >>> parts = result.value
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD\\,user", "ou=users"]

        """
        validation_result = self._validate_split_inputs(split_char, escape_char)
        if validation_result.is_failure:
            return r[list[str]].fail(
                validation_result.error or "Validation failed",
            )

        # Handle empty text early (only if it's actually a string and empty)
        try:
            if not text:
                self.logger.debug(
                    "Empty text provided, returning list with empty string",
                    operation="split_on_char_with_escape",
                )
                return r[list[str]].ok([""])
        except (TypeError, AttributeError):
            # If text doesn't support truthiness check, continue to processing
            # where exception will be caught
            pass

        # Execute splitting with error handling
        return self._execute_escape_splitting(text, split_char, escape_char)

    def normalize_whitespace(
        self,
        text: str,
        pattern: str = r"\s+",
        replacement: str = " ",
    ) -> r[str]:
        r"""Normalize whitespace in text using regex pattern.

            **Generic replacement for**: Multiple spaces to single space normalization

        Args:
            text: Text to normalize
            pattern: Regex pattern to match (default: one or more whitespace)
            replacement: Replacement string (default: single space)

        Returns:
            FlextResult with normalized text or error

        Example:
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.normalize_whitespace("hello    world\\t\\nfoo")
            >>> normalized = result.value  # "hello world foo"

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
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning unchanged",
                operation="normalize_whitespace",
            )
            return r[str].ok(text)

        try:
            self.logger.debug(
                "Applying regex pattern for whitespace normalization",
                operation="normalize_whitespace",
                pattern=pattern,
                replacement=replacement,
            )

            normalized = re.sub(pattern, replacement, text).strip()

            self.logger.debug(
                "Whitespace normalization completed",
                operation="normalize_whitespace",
                original_length=len(text),
                normalized_length=len(normalized),
                replacements_made=len(text) - len(normalized),
            )

            return r[str].ok(normalized)

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
                error_type=e.__class__.__name__,
                pattern=pattern,
                replacement=replacement,
                consequence="Cannot normalize whitespace - invalid pattern or internal error",
            )
            return r[str].fail(f"Failed to normalize whitespace: {e}")

    def apply_regex_pipeline(
        self,
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[str]:
        r"""Apply sequence of regex substitutions to text.

        **Generic replacement for**: Multiple regex.sub() calls

        Args:
            text: Text to transform
            patterns: List of (pattern, replacement) or (pattern, replacement, flags) tuples

        Returns:
            FlextResult with transformed text or error

        Example:
            >>> patterns = [
            ...     (r"\\s+=", "="),  # Remove spaces before =
            ...     (r",\\s+", ","),  # Remove spaces after ,
            ...     (r"\\s+", " "),  # Normalize whitespace
            ... ]
            >>> parser = FlextUtilitiesParser()
            >>> result = parser.apply_regex_pipeline(
            ...     "cn = REDACTED_LDAP_BIND_PASSWORD , ou = users", patterns
            ... )
            >>> cleaned = result.value  # "cn=REDACTED_LDAP_BIND_PASSWORD,ou=users"

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
        )

        # Handle edge cases
        edge_result = self._handle_pipeline_edge_cases(text, patterns)
        if edge_result is not None:
            return edge_result

        # Ensure text is not None before processing
        if text is None:
            return r[str].fail("Text cannot be None for regex pipeline")

        try:
            self.logger.debug(
                "Applying regex patterns sequentially",
                operation="apply_regex_pipeline",
                patterns_count=len(patterns),
            )

            # Process all patterns - text is guaranteed to be str here
            process_result = self._process_all_patterns(text, patterns)
            if process_result.is_failure:
                return r[str].fail(
                    process_result.error or "Unknown error in pattern processing",
                )

            # Use .value directly - FlextResult never returns None on success
            result_text, applied_patterns = process_result.value

            final_result = result_text.strip()

            self.logger.debug(
                "Regex pipeline completed successfully",
                operation="apply_regex_pipeline",
                patterns_applied=applied_patterns,
                original_length=len(text),
                final_length=len(final_result),
                total_replacements=len(text) - len(final_result),
            )

            return r[str].ok(final_result)

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
                error_type=e.__class__.__name__,
                patterns_count=len(patterns),
                text_length=text_len,
                consequence="Cannot apply regex transformations - invalid pattern or internal error",
            )
            return r[str].fail(f"Failed to apply regex pipeline: {e}")

    @staticmethod
    def _extract_key_from_mapping(obj: t.ConfigMapValue) -> str | None:
        """Extract key from mapping object (Strategy 2).

        Args:
            obj: Mapping object to extract key from.

        Returns:
            String key if found, None otherwise.

        """
        items_method = getattr(obj, "items", None)
        if not callable(items_method):
            return None
        mapping_data: Mapping[str, t.ConfigMapValue] = dict(items_method())
        for key in ("name", "id"):
            if key in mapping_data:
                value = mapping_data[key]
                if value.__class__ is str:
                    return value
        return None

    @staticmethod
    def _extract_key_from_attributes(
        obj: t.ConfigMapValue,
    ) -> str | None:
        """Extract key from object attributes (Strategy 3).

        Args:
            obj: Object to extract key from.

        Returns:
            String key if found, None otherwise.

        """
        for attr in ("name", "id"):
            attr_value = getattr(obj, attr, None)
            if FlextUtilitiesGuards.is_type(attr_value, str):
                return attr_value
        return None

    @staticmethod
    def _extract_key_from_str_conversion(
        obj: t.ConfigMapValue,
    ) -> str | None:
        """Extract key from string conversion (Strategy 5).

        Args:
            obj: Object to convert to string.

        Returns:
            String key if valid, None otherwise.

        """
        try:
            str_repr = str(obj)
            if str_repr and str_repr != f"<{obj.__class__.__name__} object>":
                return str_repr
        except (TypeError, ValueError):
            pass
        return None

    def get_object_key(self, obj: t.ConfigMapValue) -> str:
        """Get comparable string key from object (generic helper).

        This generic helper consolidates object-to-key conversion logic from
        dispatcher.py (_normalize_command_key) and provides flexible key extraction
        strategies for objects.

        Extraction Strategy (in order):
            1. Try __name__ attribute (for types, classes, functions)
            2. Try dict 'name' or 'id' key values (for dict-like objects)
            3. Try 'name' or 'id' attribute on instances
            4. Try object class name
            5. Try str conversion
            6. Use type name as final fallback

        Args:
            obj: Object to extract key from (type, class, instance, etc.)

        Returns:
            String key for object (comparable, hashable)

        Example:
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
        from flext_core.protocols import p
            >>> parser = u.Parser()
            >>> # Class/Type
            >>> parser.get_object_key(int)
            'int'
            >>> # Function
            >>> parser.get_object_key(len)
            'len'
            >>> # Dict with name key
            >>> parser.get_object_key({"name": "MyObj"})
            'MyObj'
            >>> # Instance
            >>> obj = object()
            >>> key = parser.get_object_key(obj)
            >>> key.__class__ is str
            True

        """
        self.logger.debug(
            "Starting object key extraction",
            operation="get_object_key",
            obj_type=obj.__class__.__name__,
            has_name_attr=hasattr(obj, "__name__"),
        )

        if isinstance(obj, str):
            key = obj
        elif isinstance(dunder_name := getattr(obj, "__name__", None), str):
            key = dunder_name
        elif Mapping in obj.__class__.__mro__:
            # After isinstance, obj is Mapping - use directly
            mapping_key = self._extract_key_from_mapping(obj)
            key = mapping_key if mapping_key is not None else obj.__class__.__name__
        # Strategy 3: Try 'name' or 'id' attribute on instances
        elif (attr_key := self._extract_key_from_attributes(obj)) is not None:
            key = attr_key
        # Strategy 4: Try object class name
        elif hasattr(obj, "__class__"):
            key = obj.__class__.__name__
        # Strategy 5: Try str conversion
        elif (str_key := self._extract_key_from_str_conversion(obj)) is not None:
            key = str_key
        # Final fallback: type name
        else:
            key = obj.__class__.__name__

        return key

    @overload
    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str],
    ) -> r[tuple[str, str, int]]: ...

    @overload
    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str, int],
    ) -> r[tuple[str, str, int]]: ...

    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str] | tuple[str, str, int],
    ) -> r[tuple[str, str, int]]:
        """Extract pattern, replacement, and flags from tuple."""
        # Convert to list for safe indexing (mypy can't narrow union tuple types)
        elements: list[str | int] = list(pattern_tuple)
        tuple_len = len(elements)

        # Initialize variables for type checking before branches
        pattern_val: str = ""
        replacement_val: str = ""
        flags_val: int = 0

        if tuple_len == self.PATTERN_TUPLE_MIN_LENGTH:
            if elements[0].__class__ is not str or elements[1].__class__ is not str:
                return r[tuple[str, str, int]].fail(
                    "validation error: pattern and replacement must be strings",
                )
            pattern_val = elements[0]
            replacement_val = elements[1]
            flags_val = 0
        elif tuple_len == self.PATTERN_TUPLE_MAX_LENGTH:
            if elements[0].__class__ is not str or elements[1].__class__ is not str:
                return r[tuple[str, str, int]].fail(
                    "validation error: pattern and replacement must be strings",
                )
            pattern_val = elements[0]
            replacement_val = elements[1]
            third_elem = elements[2]
            if third_elem.__class__ is not int:
                return r[tuple[str, str, int]].fail(
                    "validation error: regex flags must be an integer",
                )
            flags_val = third_elem
        else:
            return r[tuple[str, str, int]].fail(
                f"Invalid pattern tuple length {tuple_len}, expected 2 or 3",
            )

        return r[tuple[str, str, int]].ok((pattern_val, replacement_val, flags_val))

    def _apply_single_pattern(
        self,
        params: FlextModelsCollections.PatternApplicationParams,
    ) -> r[str]:
        """Apply a single regex pattern to text."""
        self.logger.debug(
            "Applying regex pattern",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            total_patterns=params.total_patterns,
            pattern=params.pattern,
            replacement=params.replacement,
            flags=params.flags,
        )

        before_length = len(params.text)
        try:
            result_text = re.sub(
                params.pattern,
                params.replacement,
                params.text,
                flags=params.flags,
            )
        except (re.PatternError, ValueError) as e:
            return r[str].fail(
                f"Invalid regex pattern '{params.pattern}': {e}",
            )

        after_length = len(result_text)
        replacements = before_length - after_length

        self.logger.debug(
            "Pattern applied",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            replacements_made=replacements,
        )

        return r[str].ok(result_text)

    def _handle_pipeline_edge_cases(
        self,
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[str] | None:
        """Handle edge cases for regex pipeline application.

        Returns:
            FlextResult if edge case handled, None to continue processing

        """
        if text is None:
            self.logger.debug(
                "None text provided, returning failure",
                operation="apply_regex_pipeline",
            )
            return r[str].fail("Text cannot be None")

        if not text:
            self.logger.debug(
                "Empty text provided, returning unchanged",
                operation="apply_regex_pipeline",
            )
            return r[str].ok(text)

        if not patterns:
            self.logger.warning(
                "No patterns provided for regex pipeline",
                operation="apply_regex_pipeline",
                text_length=self._safe_text_length(text),
            )
            return r[str].ok(text)

        # No edge case, continue processing
        return None

    def _process_escape_splitting(
        self,
        text: str,
        split_char: str,
        escape_char: str,
    ) -> r[tuple[list[str], int]]:
        """Process text with escape character handling and return components."""
        components: list[str] = []
        current: list[str] = []
        i = 0
        escape_count = 0

        while i < len(text):
            if text[i] == escape_char and i + 1 < len(text):
                # Add escaped character (remove escape character)
                self.logger.debug(
                    "Found escape sequence",
                    operation="split_on_char_with_escape",
                    position=i,
                    escaped_char=text[i + 1],
                )
                current.append(text[i + 1])  # Add only the escaped character
                escape_count += 1
                i += 2
            elif text[i] == split_char:
                # Found unescaped delimiter
                self.logger.debug(
                    "Found unescaped delimiter",
                    operation="split_on_char_with_escape",
                    position=i,
                    current_component_length=len(current),
                )
                components.append("".join(current))
                current = []
                i += 1
            else:
                current.append(text[i])
                i += 1

        # Add final component
        # Always add final component, even if empty (for trailing delimiters)
        self.logger.debug(
            "Adding final component",
            operation="split_on_char_with_escape",
            final_component_length=len(current),
        )
        components.append("".join(current))

        return r[tuple[list[str], int]].ok((components, escape_count))

    def _process_all_patterns(
        self,
        text: str,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> r[tuple[str, int]]:
        """Process all regex patterns and return final text and count."""
        result_text = text
        applied_patterns = 0

        for i, pattern_tuple in enumerate(patterns):
            # Extract pattern components from tuple
            # Type narrowing: convert union to explicit tuple for overload matching
            tuple_len = len(pattern_tuple)

            if tuple_len == self.TUPLE_LENGTH_2:
                # Explicitly create 2-element tuple for first overload
                pattern_result = self._extract_pattern_components((
                    pattern_tuple[0],
                    pattern_tuple[1],
                ))
            elif tuple_len == self.TUPLE_LENGTH_3:
                # Convert to list for dynamic indexing - mypy can't narrow tuple union types
                items: list[str | int] = list(pattern_tuple)
                a: str = str(items[0])
                b: str = str(items[1])
                # Third element is int for 3-element tuple variant
                c_raw = items[2] if len(items) > self.TUPLE_LENGTH_2 else 0
                c: int = c_raw if c_raw.__class__ is int else int(str(c_raw))
                # Call method with typed values
                pattern_result = self._extract_pattern_components((a, b, c))
            else:
                msg = f"Pattern tuple must have 2 or 3 elements, got {tuple_len}"
                return r[tuple[str, int]].fail(msg)
            if pattern_result.is_failure:
                return r[tuple[str, int]].fail(
                    pattern_result.error
                    or "Unknown error extracting pattern components",
                )

            # Use .value directly - FlextResult never returns None on success
            pattern, replacement, flags = pattern_result.value

            # Apply the pattern (u.from_kwargs at call sites)
            params_result = FlextUtilitiesModel.from_kwargs(
                FlextModelsCollections.PatternApplicationParams,
                text=result_text,
                pattern=pattern,
                replacement=replacement,
                flags=flags,
                pattern_index=i,
                total_patterns=len(patterns),
            )
            if params_result.is_failure:
                return r[tuple[str, int]].fail(
                    params_result.error or "Unknown error creating params",
                )

            # Use .value directly - FlextResult never returns None on success
            apply_result = self._apply_single_pattern(params_result.value)
            if apply_result.is_failure:
                return r[tuple[str, int]].fail(
                    apply_result.error or "Unknown error applying pattern",
                )

            # Use .value directly - FlextResult never returns None on success
            result_text = apply_result.value
            applied_patterns += 1

        return r[tuple[str, int]].ok((result_text, applied_patterns))

    # =========================================================================
    # PARSE METHODS - Universal type parsing
    # These methods avoid circular imports by using inline helper implementations
    # =========================================================================

    @staticmethod
    def _parse_get_attr(
        obj: t.ConfigMapValue, attr: str, default: t.ConfigMapValue = None
    ) -> t.ConfigMapValue:
        """Get attribute safely (avoids circular import with u.get)."""
        return getattr(obj, attr, default)

    @staticmethod
    def _parse_find_first[T](
        items: list[T],
        predicate: Callable[[T], bool],
    ) -> T | None:
        """Find first item matching predicate (avoids circular import)."""
        for item in items:
            # Type narrowing: item is T
            item_typed: T = item
            if predicate(item_typed):
                return item_typed
        return None

    @staticmethod
    def _parse_normalize_compare(a: t.ConfigMapValue, b: t.ConfigMapValue) -> bool:
        """Case-insensitive string comparison (avoids circular import)."""
        if a.__class__ is not str or b.__class__ is not str:
            return False
        return a.lower() == b.lower()

    @staticmethod
    def _parse_normalize_str(value: t.ConfigMapValue, *, case: str = "lower") -> str:
        """Normalize string value (avoids circular import with u.normalize)."""
        if value.__class__ is not str:
            return str(value)
        value_str: str = value
        if case == "lower":
            return value_str.lower()
        if case == "upper":
            return value_str.upper()
        return value_str

    @staticmethod
    def _parse_result_error[T](result: r[T], default: str = "") -> str:
        """Extract error from result (avoids circular import with u.err)."""
        if result.is_failure:
            return result.error or default
        return default

    @staticmethod
    def _parse_with_default[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        error_msg: str,
    ) -> r[T]:
        """Return default or error for parse failures."""
        if default is not None:
            return r[T].ok(default)
        if default_factory is not None:
            return r[T].ok(default_factory())
        return r[T].fail(error_msg)

    @staticmethod
    def _parse_enum[T: StrEnum](
        value: str,
        target: type[T],
        *,
        case_insensitive: bool,
    ) -> r[T] | None:
        """Parse StrEnum with optional case-insensitivity. Returns None if not enum."""
        if StrEnum not in target.__mro__:
            return None
        members_proxy = getattr(target, "__members__", {})
        # Convert to dict for easier iteration
        members: Mapping[str, T] = dict(members_proxy)

        # Case-insensitive matching using members lookup
        if case_insensitive:
            for member_name, member_value in members.items():
                # Check name or value match (case-insensitive)
                name_matches = FlextUtilitiesParser._parse_normalize_compare(
                    member_name,
                    value,
                )
                value_attr = getattr(member_value, "value", None)
                value_matches = (
                    value_attr is not None
                    and FlextUtilitiesParser._parse_normalize_compare(
                        value_attr,
                        value,
                    )
                )
                if name_matches or value_matches:
                    return r[T].ok(member_value)

        # Case-sensitive: direct lookup in __members__
        if value in members:
            return r[T].ok(members[value])

        # Try parsing value as enum value (not name)
        for member_instance in members.values():
            member_val = getattr(member_instance, "value", None)
            if member_val == value:
                return r[T].ok(member_instance)

        return r[T].fail(
            f"Cannot parse '{value}' as {getattr(target, '__name__', 'Unknown')}",
        )

    @staticmethod
    def _parse_model[T: BaseModel](
        value: t.ConfigMapValue,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
    ) -> r[T] | None:
        """Parse Pydantic BaseModel. Returns None if not model."""
        if BaseModel not in target.__mro__:
            return None
        if Mapping not in value.__class__.__mro__:
            return r[T].fail(
                f"{field_prefix}Expected dict for model, got {value.__class__.__name__}",
            )
        value_dict_data: t.Dict = t.Dict({})
        items_method = getattr(value, "items", None)
        if not callable(items_method):
            return r[T].fail(
                f"{field_prefix}Expected dict for model, got {value.__class__.__name__}",
            )

        for k, v in items_method():
            key = str(k)
            if v.__class__ in {str, int, float, bool, datetime, NoneType}:
                value_dict_data.root[key] = v
                continue
            if v.__class__ in {list, tuple}:
                value_dict_data.root[key] = str(v)
                continue
            value_dict_data.root[key] = str(v)
        scalar_data: t.Dict = t.Dict({})
        for dict_key, dict_value in value_dict_data.root.items():
            if dict_value is None or dict_value.__class__ in {
                str,
                int,
                float,
                bool,
                datetime,
            }:
                scalar_data.root[dict_key] = dict_value
            else:
                scalar_data.root[dict_key] = str(dict_value)

        value_dict: Mapping[str, t.ScalarValue] = scalar_data.root
        result = FlextUtilitiesModel.from_dict(target, value_dict, strict=strict)
        if result.is_success:
            return r[T].ok(result.value)
        return r[T].fail(
            FlextUtilitiesParser._parse_result_error(result, "Model parse failed"),
        )

    @staticmethod
    def _coerce_to_int(value: t.ConfigMapValue) -> r[int] | None:
        """Coerce value to int. Returns None if not coercible."""
        if value.__class__ in {str, float}:
            try:
                return r[int].ok(int(float(str(value))))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _coerce_to_float(value: t.ConfigMapValue) -> r[float] | None:
        """Coerce value to float. Returns None if not coercible."""
        if value.__class__ in {str, int}:
            try:
                return r[float].ok(float(str(value)))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _coerce_to_bool(value: t.ConfigMapValue) -> r[bool] | None:
        """Coerce value to bool. Returns None if not coercible."""
        if FlextUtilitiesGuards.is_type(value, str):
            normalized_val = FlextUtilitiesParser._parse_normalize_str(
                value,
                case="lower",
            )
            if normalized_val in {"true", "1", "yes", "on"}:
                return r[bool].ok(value=True)
            if normalized_val in {"false", "0", "no", "off"}:
                return r[bool].ok(False)
            return None
        return r[bool].ok(bool(value))

    @staticmethod
    def _coerce_to_str(value: t.ConfigMapValue) -> r[str]:
        """Coerce value to string - returns FlextResult[str]."""
        return r[str].ok(str(value))

    @staticmethod
    def _coerce_primitive(
        value: t.ConfigMapValue,
        target: type[int | float | str | bool],
    ) -> r[int] | r[float] | r[str] | r[bool] | None:
        """Coerce primitive types. Returns None if no coercion applied."""
        if target is int:
            return FlextUtilitiesParser._coerce_to_int(value)
        if target is float:
            return FlextUtilitiesParser._coerce_to_float(value)
        if target is str:
            return FlextUtilitiesParser._coerce_to_str(value)
        if target is bool:
            return FlextUtilitiesParser._coerce_to_bool(value)
        return None

    @staticmethod
    def _parse_try_enum[T](
        value: t.ConfigMapValue,
        target: type[T],
        *,
        case_insensitive: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T] | None:
        """Helper: Try enum parsing, return None if not enum."""
        # Early return if not a StrEnum subclass
        if not issubclass(target, StrEnum):
            return None
        # Get members - returns MappingProxyType[str, T]
        members_proxy = getattr(target, "__members__", {})
        # Convert to dict for easier iteration
        members: Mapping[str, T] = dict(members_proxy)
        value_str = str(value)

        # Case-insensitive matching
        if case_insensitive:
            value_lower = value_str.lower()
            for member_name, member_value in members.items():
                if member_name.lower() == value_lower:
                    return r[T].ok(member_value)
                # Also check by value
                member_val = getattr(member_value, "value", None)
                if member_val is not None and str(member_val).lower() == value_lower:
                    return r[T].ok(member_value)
        else:
            # Case-sensitive lookup by name
            if value_str in members:
                return r[T].ok(members[value_str])
            # Try matching by value
            for member_value in members.values():
                member_val = getattr(member_value, "value", None)
                if member_val == value_str:
                    return r[T].ok(member_value)

        # No match found - return default or error
        error_msg = (
            f"Cannot parse '{value_str}' as {getattr(target, '__name__', 'Unknown')}"
        )
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            f"{field_prefix}{error_msg}",
        )

    @staticmethod
    def _parse_try_model[T: BaseModel](
        value: t.ConfigMapValue,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
    ) -> r[T] | None:
        """Helper: Try model parsing, return None if not model."""
        model_result = FlextUtilitiesParser._parse_model(
            value,
            target,
            field_prefix,
            strict=strict,
        )
        if model_result is None:
            return None
        if model_result.is_success:
            return model_result
        return FlextUtilitiesParser._parse_with_default(
            default,
            default_factory,
            FlextUtilitiesParser._parse_result_error(model_result, ""),
        )

    @staticmethod
    def _is_primitive_type(target: type) -> bool:
        """Check if target is a primitive type."""
        return target in {int, float, str, bool}

    @staticmethod
    def _try_coerce_to_primitive(
        value: t.ConfigMapValue,
        target: type[int | float | str | bool],
    ) -> r[int] | r[float] | r[str] | r[bool] | None:
        """Try to coerce value to primitive type."""
        if target is int:
            return FlextUtilitiesParser._coerce_to_int(value)
        if target is float:
            return FlextUtilitiesParser._coerce_to_float(value)
        if target is str:
            return FlextUtilitiesParser._coerce_to_str(value)
        if target is bool:
            return FlextUtilitiesParser._coerce_to_bool(value)
        return None

    @staticmethod
    def _parse_try_primitive(
        value: t.ConfigMapValue,
        target: type,
        default: float | str | bool | None,
        default_factory: Callable[[], int | float | str | bool] | None,
        field_prefix: str,
    ) -> r[int] | r[float] | r[str] | r[bool] | None:
        """Helper: Try primitive coercion."""
        # Only coerce primitive types
        if not FlextUtilitiesParser._is_primitive_type(target):
            return None
        try:
            # Dispatch to concrete primitive coercion - each branch returns directly
            if target is int:
                int_result = FlextUtilitiesParser._coerce_to_int(value)
                if int_result is not None:
                    return int_result
            elif target is float:
                float_result = FlextUtilitiesParser._coerce_to_float(value)
                if float_result is not None:
                    return float_result
            elif target is str:
                return FlextUtilitiesParser._coerce_to_str(value)
            elif target is bool:
                bool_result = FlextUtilitiesParser._coerce_to_bool(value)
                if bool_result is not None:
                    return bool_result
        except (ValueError, TypeError) as e:
            target_name = FlextUtilitiesParser._parse_get_attr(
                target,
                "__name__",
                "type",
            )
            # For error case, return failure wrapped in appropriate type
            if target is int and default.__class__ is int:
                return r[int].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is float and default.__class__ is float:
                return r[float].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is str and default.__class__ is str:
                return r[str].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
            if target is bool and default.__class__ is bool:
                return r[bool].fail(
                    f"{field_prefix}Cannot coerce {value.__class__.__name__} to {target_name}: {e}",
                )
        return None

    @staticmethod
    def _parse_try_direct[T](
        value: t.ConfigMapValue,
        target: type[T],
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Helper: Try direct type call."""
        # Guard: t.ConfigMapValue type doesn't accept constructor arguments
        if target is object:
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot construct 'object' type directly",
            )
        try:
            constructor = target
            return r[T].ok(constructor(value))
        except Exception as e:
            target_name = FlextUtilitiesParser._parse_get_attr(
                target,
                "__name__",
                "type",
            )
            return FlextUtilitiesParser._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot parse {value.__class__.__name__} to {target_name}: {e}",
            )

    @staticmethod
    def parse[T](
        value: t.ConfigMapValue,
        target: type[T],
        *,
        strict: bool = False,
        coerce: bool = True,
        case_insensitive: bool = False,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        field_name: str | None = None,
    ) -> r[T]:
        """Universal type parser supporting enums, models, and primitives.

        Parsing order: enum → model → primitive coercion → direct type call.

        Args:
            value: The value to parse.
            target: Target type to parse into.
            strict: If True, disable type coercion (exact match only).
            coerce: If True (default), allow type coercion.
            case_insensitive: For enums, match case-insensitively.
            default: Default value to return on parse failure.
            default_factory: Callable to create default on failure.
            field_name: Field name for error messages.

        Returns:
            r[T]: Ok(parsed_value) or Fail with error message.

        Examples:
            >>> result = FlextUtilitiesParser.parse("ACTIVE", Status)
            >>> result = FlextUtilitiesParser.parse("42", int)  # Ok(42)
            >>> result = FlextUtilitiesParser.parse("invalid", int, default=c.ZERO)

        """
        field_prefix = f"{field_name}: " if field_name else ""

        if value is None:
            if default is not None:
                return r[T].ok(default)
            if default_factory is not None:
                return r[T].ok(default_factory())
            return r[T].fail(field_prefix or "Value is None")

        if target.__class__ is type and target in value.__class__.__mro__:
            return r[T].ok(value)

        enum_result = FlextUtilitiesParser._parse_try_enum(
            value,
            target,
            case_insensitive=case_insensitive,
            default=default,
            default_factory=default_factory,
            field_prefix=field_prefix,
        )
        if enum_result is not None:
            return enum_result

        model_result = FlextUtilitiesParser._parse_try_model(
            value,
            target,
            field_prefix,
            strict=strict,
            default=default,
            default_factory=default_factory,
        )
        if model_result is not None:
            return model_result

        primitive_default: int | float | str | bool | None
        if default is None or default.__class__ in {int, float, str, bool}:
            primitive_default = default
        else:
            primitive_default = None

        primitive_default_factory: Callable[[], int | float | str | bool] | None = None
        if default_factory is not None:
            primitive_default_factory = default_factory

        primitive_result = FlextUtilitiesParser._parse_try_primitive(
            value,
            target,
            default=primitive_default,
            default_factory=primitive_default_factory,
            field_prefix=field_prefix,
        )
        if primitive_result is not None:
            return primitive_result

        # Let _parse_try_direct handle all cases including primitives
        return FlextUtilitiesParser._parse_try_direct(
            value,
            target,
            default,
            default_factory,
            field_prefix,
        )

    # =========================================================================
    # CONVERT METHODS - Type conversion with safe fallback
    # =========================================================================

    # Note: bool must come before int because bool is subclass of int in Python
    @overload
    @staticmethod
    def convert(
        value: t.ConfigMapValue,
        target_type: type[bool],
        default: bool,
    ) -> bool: ...

    @overload
    @staticmethod
    def convert(
        value: t.ConfigMapValue,
        target_type: type[int],
        default: int,
    ) -> int: ...

    @overload
    @staticmethod
    def convert(
        value: t.ConfigMapValue,
        target_type: type[float],
        default: float,
    ) -> float: ...

    @overload
    @staticmethod
    def convert(
        value: t.ConfigMapValue,
        target_type: type[str],
        default: str,
    ) -> str: ...

    # Note: Generic [T] overload removed - not needed as we have specific overloads for all primitives

    @staticmethod
    def convert(
        value: t.ConfigMapValue,
        target_type: type[int | float | str | bool | t.ConfigMapValue],
        default: float | str | bool | t.ConfigMapValue,
    ) -> int | float | str | bool | t.ConfigMapValue:
        """Unified type conversion with safe fallback.

        Automatically handles common type conversions (int, str, float, bool) with
        safe fallback to default value on conversion failure.

        Args:
            value: Value to convert
            target_type: Target type (int, str, float, bool)
            default: Default value to return on conversion failure

        Returns:
            Converted value or default

        Example:
            # Convert to int
            result = FlextUtilitiesParser.convert("123", int, 0)
            # → 123

            # Convert to int (invalid)
            result = FlextUtilitiesParser.convert("invalid", int, 0)
            # → 0

            # Convert to float
            result = FlextUtilitiesParser.convert("3.14", float, 0.0)
            # → 3.14

        """
        if value.__class__ is target_type:
            return value

        if target_type is int and default.__class__ is int:
            return FlextUtilitiesParser._convert_to_int(value, default=default)
        if target_type is float and default.__class__ is float:
            return FlextUtilitiesParser._convert_to_float(value, default=default)
        if target_type is str and default.__class__ is str:
            return FlextUtilitiesParser._convert_to_str(value, default=default)
        if target_type is bool and default.__class__ is bool:
            return FlextUtilitiesParser._convert_to_bool(value, default=default)
        # Fallback for other types
        return default

    @staticmethod
    def _convert_to_int(value: t.ConfigMapValue, *, default: int) -> int:
        """Convert value to int with fallback."""
        if value.__class__ is int:
            return value
        if value.__class__ is str:
            try:
                return int(value)
            except ValueError:
                return default
        if value.__class__ is float:
            return int(value)
        return default

    @staticmethod
    def _convert_to_float(value: t.ConfigMapValue, *, default: float) -> float:
        """Convert value to float with fallback."""
        if value.__class__ is float:
            return value
        if value.__class__ in {int, str}:
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def _convert_to_str(value: t.ConfigMapValue, *, default: str) -> str:
        """Convert value to str with fallback."""
        if value.__class__ is str:
            return value
        if value is None:
            return default
        try:
            return str(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _convert_to_bool(value: t.ConfigMapValue, *, default: bool) -> bool:
        """Convert value to bool with fallback."""
        if value.__class__ is bool:
            return value
        if value.__class__ is str:
            normalized = FlextUtilitiesParser._parse_normalize_str(value, case="lower")
            return normalized in {"true", "1", "yes", "on"}
        if value.__class__ in {int, float}:
            return bool(value)
        return default

    @staticmethod
    def _convert_fallback[T](
        value: t.ConfigMapValue,
        target_type: type[T],
        default: T,
    ) -> T:
        """Fallback: try direct type constructor."""
        # Guard: t.ConfigMapValue type doesn't accept constructor arguments
        return default

    # =========================================================================
    # CONV_* METHODS - Convenience conversion wrappers
    # =========================================================================

    @staticmethod
    def conv_str(value: t.ConfigMapValue, *, default: str = "") -> str:
        """Convert to string (builder: conv().str()).

        Mnemonic: conv = convert, str = string

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            str: Converted string

        """
        if value is None:
            return default
        if value.__class__ is str:
            return value
        try:
            return str(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def conv_str_list(
        value: t.ConfigMapValue,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert to str_list (builder: conv().str_list()).

        Mnemonic: conv = convert, str_list = list[str]

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted list

        """
        if default is None:
            default = []
        if value is None:
            return default
        if value.__class__ is list:
            return [str(item) for item in value]
        if value.__class__ is str:
            return [value] if value else default
        if value.__class__ in {tuple, set, frozenset}:
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def conv_int(value: t.ConfigMapValue, *, default: int = 0) -> int:
        """Convert to int (builder: conv().int()).

        Mnemonic: conv = convert, int = integer

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            int: Converted integer

        """
        return FlextUtilitiesParser.convert(value, int, default)

    @staticmethod
    def conv_str_list_truthy(
        value: t.ConfigMapValue | None,
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Convert to str_list and filter truthy.

        Mnemonic: conv_str_list_truthy = convert + filter truthy

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted and filtered list

        """
        result = FlextUtilitiesParser.conv_str_list(value, default=default)
        return [item for item in result if item]

    @staticmethod
    def conv_str_list_safe(value: t.ConfigMapValue | None) -> list[str]:
        """Safe str_list conversion.

        Mnemonic: conv_str_list_safe = convert + safe mode

        Args:
            value: Value to convert (can be None)

        Returns:
            list[str]: Converted list or []

        """
        if value is None:
            return []
        return FlextUtilitiesParser.conv_str_list(value, default=[])

    # =========================================================================
    # NORM_* METHODS - String normalization utilities
    # =========================================================================

    @staticmethod
    def norm_str(
        value: t.ConfigMapValue,
        *,
        case: str | None = None,
        default: str = "",
    ) -> str:
        """Normalize string (builder: norm().str()).

        Mnemonic: norm = normalize, str = string

        Args:
            value: Value to normalize
            case: Case normalization ("lower", "upper", "title")
            default: Default if None

        Returns:
            str: Normalized string

        """
        str_value = FlextUtilitiesParser.conv_str(value, default=default)
        if case:
            return FlextUtilitiesParser._parse_normalize_str(str_value, case=case)
        return str_value

    @staticmethod
    def norm_list(
        items: list[str] | t.ConfigMap,
        *,
        case: str | None = None,
        filter_truthy: bool = False,
        to_set: bool = False,
    ) -> list[str] | set[str] | t.ConfigMap:
        """Normalize list/dict (builder: norm().list()).

        Mnemonic: norm = normalize, list = list[str]

        Args:
            items: Items to normalize
            case: Case normalization
            filter_truthy: Filter truthy first
            to_set: Return set instead of list

        Returns:
            Normalized list/set/dict

        """
        if items.__class__ is dict:
            dict_items: Mapping[str, str] = items
            if filter_truthy:
                dict_items = {k: v for k, v in dict_items.items() if v}
            return t.ConfigMap({
                k: FlextUtilitiesParser.norm_str(v, case=case)
                for k, v in dict_items.items()
            })

        # items is list[str] here
        list_items = items
        if filter_truthy:
            list_items = [item for item in list_items if item]

        normalized = [
            FlextUtilitiesParser.norm_str(item, case=case) for item in list_items
        ]
        if to_set:
            return set(normalized)
        return normalized

    @staticmethod
    def norm_join(items: list[str], *, case: str | None = None, sep: str = " ") -> str:
        """Normalize and join (builder: norm().join()).

        Mnemonic: norm = normalize, join = string join

        Args:
            items: Items to normalize and join
            case: Case normalization
            sep: Separator

        Returns:
            str: Normalized and joined string

        """
        if case:
            normalized = [FlextUtilitiesParser.norm_str(v, case=case) for v in items]
        else:
            normalized = items
        return sep.join(normalized)

    @staticmethod
    def norm_in(
        value: str,
        items: list[str] | m.ConfigMap,
        *,
        case: str | None = None,
    ) -> bool:
        """Normalized membership check (builder: norm().in_()).

        Mnemonic: norm = normalize, in_ = membership check

        Args:
            value: Value to check
            items: Items to check against
            case: Case normalization

        Returns:
            bool: True if normalized value in normalized items

        """
        items_to_check: list[str]
        if Mapping in items.__class__.__mro__ and hasattr(items, "root"):
            items_to_check = [str(k) for k in items.root]
        elif Mapping in items.__class__.__mro__:
            items_to_check = [str(k) for k in items]
        elif items.__class__ is list:
            items_to_check = items
        else:
            items_to_check = []

        normalized_value = FlextUtilitiesParser.norm_str(value, case=case or "lower")
        normalized_result = FlextUtilitiesParser.norm_list(
            items_to_check,
            case=case or "lower",
        )
        if normalized_result.__class__ in {list, set}:
            return normalized_value in normalized_result
        return normalized_value in normalized_result.values()


__all__ = [
    "FlextUtilitiesParser",
]
