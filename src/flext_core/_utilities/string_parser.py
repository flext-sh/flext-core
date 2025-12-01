"""Utilities module - FlextUtilitiesStringParser.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import cast

import structlog

from flext_core._models.collections import FlextModelsCollections
from flext_core.typings import FlextTypes


class FlextUtilitiesStringParser:
    r"""String parsing utilities for delimited and structured text processing.

    Constants:
        PATTERN_TUPLE_MIN_LENGTH: Minimum length for pattern tuple (pattern, replacement)
        PATTERN_TUPLE_MAX_LENGTH: Maximum length for pattern tuple (pattern, replacement, flags)

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
    >>> result = parser.split_on_char_with_escape("cn=REDACTED_LDAP_BIND_PASSWORD\\,dc=com", ",", "\\")

    >>> # Normalize whitespace
    >>> result = parser.normalize_whitespace("  hello   world  ")
    >>> cleaned = result.unwrap()  # "hello world"
    """

    PATTERN_TUPLE_MIN_LENGTH: int = 2
    PATTERN_TUPLE_MAX_LENGTH: int = 3

    # Magic value constants to reduce complexity
    TUPLE_LENGTH_2: int = 2
    TUPLE_LENGTH_3: int = 3

    def __init__(self) -> None:
        """Initialize string parser with logging."""
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def _safe_text_length(text: FlextTypes.GeneralValueType) -> str | int:
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
            components = [c for c in components if c.strip()]

        if validator:
            self.logger.debug(
                "Validating components with custom validator",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            # Filter out invalid components instead of failing
            valid_components = []
            for comp in components:
                if validator(comp):
                    valid_components.append(comp)
                else:
                    self.logger.debug(
                        "Component filtered out by validator",
                        operation="parse_delimited",
                        invalid_component=comp,
                        validator_type=type(validator).__name__,
                        source="flext-core/src/flext_core/_utilities/string_parser.py",
                    )
            components = valid_components

        return FlextResult[list[str]].ok(components)

    def parse_delimited(
        self,
        text: str,
        delimiter: str,
        *,
        options: FlextModelsCollections.ParseOptions | None = None,
        **legacy_options: bool | Callable[[str], bool] | None,
    ) -> FlextResult[list[str]]:
        """Parse delimited string into list of components.

        **Generic replacement for**: DN.split(), CSV parsing, config parsing

        Args:
            text: String to parse
            delimiter: Delimiter character/string
            options: ParseOptions object (preferred). If provided, overrides legacy_options
            **legacy_options: Legacy keyword arguments for backward compatibility:
                - strip: Strip whitespace from each component (default: True)
                - remove_empty: Remove empty components after stripping (default: True)
                - validator: Optional validation function for each component (default: None)

        Returns:
            FlextResult with list of parsed components or error

        Example:
            >>> # NEW - Using ParseOptions
            >>> from flext_core._models.collections import FlextModelsCollections
            >>> opts = FlextModelsCollections.ParseOptions(
            ...     strip=True, remove_empty=True
            ... )
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com", ",", options=opts
            ... )
            >>> components = result.unwrap()
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD", "ou=users", "dc=example", "dc=com"]

            >>> # OLD - Backward compatible
            >>> result = parser.parse_delimited(
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD, ou=users, dc=example, dc=com", ","
            ... )
            >>> components = result.unwrap()

        """
        # Safely get text length for logging
        try:
            text_len = self._safe_text_length(text)
        except (TypeError, AttributeError):
            text_len = -1  # Unknown length

        # Normalize options: use provided ParseOptions or create from legacy_options
        if options is not None:
            parse_opts = options
        else:
            # Extract legacy options with defaults
            strip_val = legacy_options.get("strip", True)
            remove_empty_val = legacy_options.get("remove_empty", True)
            validator_val = legacy_options.get("validator")
            # Filter out bool values - validator must be Callable or None
            validator_typed: Callable[[str], bool] | None = (
                validator_val
                if callable(validator_val) and not isinstance(validator_val, bool)
                else None
            )
            parse_opts = FlextModelsCollections.ParseOptions(
                strip=bool(strip_val),
                remove_empty=bool(remove_empty_val),
                validator=validator_typed,
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
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        if not text:
            self.logger.debug(
                "Empty text provided, returning empty list",
                operation="parse_delimited",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].ok([])

        # Validate delimiter
        if not delimiter or len(delimiter) != 1:
            return FlextResult[list[str]].fail(
                f"Delimiter must be exactly one character, got '{delimiter}'",
            )

        # Reject whitespace/control characters as delimiters
        if delimiter.isspace() or not delimiter.isprintable():
            return FlextResult[list[str]].fail(
                f"Delimiter cannot be a whitespace or control character: '{delimiter}'",
            )

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
            ...     "cn=REDACTED_LDAP_BIND_PASSWORD\\,user,ou=users", ","
            ... )
            >>> parts = result.unwrap()
            >>> # ["cn=REDACTED_LDAP_BIND_PASSWORD\\,user", "ou=users"]

        """
        # Validate inputs
        if not split_char:
            return FlextResult[list[str]].fail("Split character cannot be empty")

        if not escape_char:
            return FlextResult[list[str]].fail("Escape character cannot be empty")

        if split_char == escape_char:
            return FlextResult[list[str]].fail(
                "Split character and escape character cannot be the same",
            )

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
                "Empty text provided, returning list with empty string",
                operation="split_on_char_with_escape",
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )
            return FlextResult[list[str]].ok([""])

        try:
            self.logger.debug(
                "Processing text with escape character handling",
                operation="split_on_char_with_escape",
                text_length=self._safe_text_length(text),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            # Process the text and extract components
            split_result = self._process_escape_splitting(text, split_char, escape_char)
            if split_result.is_failure:
                return FlextResult[list[str]].fail(
                    split_result.error or "Unknown error in escape splitting",
                )

            components, escape_count = split_result.unwrap()

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

            **Generic replacement for**: Multiple spaces to single space normalization

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
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> FlextResult[str]:
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
            >>> parser = FlextUtilitiesStringParser()
            >>> result = parser.apply_regex_pipeline(
            ...     "cn = REDACTED_LDAP_BIND_PASSWORD , ou = users", patterns
            ... )
            >>> cleaned = result.unwrap()  # "cn=REDACTED_LDAP_BIND_PASSWORD,ou=users"

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

        # Handle edge cases
        edge_result = self._handle_pipeline_edge_cases(text, patterns)
        if edge_result is not None:
            return edge_result

        # Ensure text is not None before processing
        if text is None:
            return FlextResult[str].fail("Text cannot be None for regex pipeline")

        try:
            self.logger.debug(
                "Applying regex patterns sequentially",
                operation="apply_regex_pipeline",
                patterns_count=len(patterns),
                source="flext-core/src/flext_core/_utilities/string_parser.py",
            )

            # Process all patterns - text is guaranteed to be str here
            process_result = self._process_all_patterns(text, patterns)
            if process_result.is_failure:
                return FlextResult[str].fail(
                    process_result.error or "Unknown error in pattern processing",
                )

            result_text, applied_patterns = process_result.unwrap()

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

    def get_object_key(self, obj: FlextTypes.GeneralValueType) -> str:
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
            >>> parser.get_object_key(len)
            'len'
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

        # Try strategies in order
        # Strategy 1: Try __name__ attribute (for types, classes, functions)
        # Use getattr for type-safe access (returns None if attribute doesn't exist)
        name = getattr(obj, "__name__", None)
        if isinstance(name, str):
            return name

        # Strategy 2: Try dict keys (for dict-like objects)
        if isinstance(obj, Mapping):
            keys = list(obj.keys())
            if keys:
                return str(keys[0])

        # Strategy 3: Try object attributes
        if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
            return obj.__class__.__name__

        # Strategy 4: Try str conversion
        try:
            str_repr = str(obj)
            if str_repr and str_repr != f"<{type(obj).__name__} object>":
                return str_repr
        except (TypeError, ValueError):
            pass

        # Final fallback: type name
        return type(obj).__name__

    def _extract_pattern_components(
        self,
        pattern_tuple: tuple[str, str] | tuple[str, str, int],
    ) -> FlextResult[tuple[str, str, int]]:
        """Extract pattern, replacement, and flags from tuple."""
        tuple_len = len(pattern_tuple)

        if tuple_len == self.PATTERN_TUPLE_MIN_LENGTH:
            # Type narrowing: tuple[str, str]
            pattern: str = pattern_tuple[0]
            replacement: str = pattern_tuple[1]
            flags: int = 0
        elif tuple_len == self.PATTERN_TUPLE_MAX_LENGTH:
            # Type narrowing: tuple[str, str, int] - cast since we checked length
            full_tuple = cast("tuple[str, str, int]", pattern_tuple)
            pattern = str(full_tuple[0])
            replacement = str(full_tuple[1])
            flags_value = full_tuple[2]
            flags = flags_value if isinstance(flags_value, int) else 0
        else:
            return FlextResult[tuple[str, str, int]].fail(
                f"Invalid pattern tuple length {tuple_len}, expected 2 or 3",
            )

        return FlextResult[tuple[str, str, int]].ok((pattern, replacement, flags))

    def _apply_single_pattern(
        self,
        params: FlextModelsCollections.PatternApplicationParams,
    ) -> FlextResult[str]:
        """Apply a single regex pattern to text."""
        self.logger.debug(
            "Applying regex pattern",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            total_patterns=params.total_patterns,
            pattern=params.pattern,
            replacement=params.replacement,
            flags=params.flags,
            source="flext-core/src/flext_core/_utilities/string_parser.py",
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
            return FlextResult[str].fail(
                f"Invalid regex pattern '{params.pattern}': {e}",
            )

        after_length = len(result_text)
        replacements = before_length - after_length

        self.logger.debug(
            "Pattern applied",
            operation="apply_regex_pipeline",
            pattern_index=params.pattern_index + 1,
            replacements_made=replacements,
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )

        return FlextResult[str].ok(result_text)

    def _handle_pipeline_edge_cases(
        self,
        text: str | None,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> FlextResult[str] | None:
        """Handle edge cases for regex pipeline application.

        Returns:
            FlextResult if edge case handled, None to continue processing

        """
        if text is None:
            # None text is invalid - return failure
            return FlextResult[str].fail("Text cannot be None")

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

        # No edge case, continue processing
        return None

    def _process_escape_splitting(
        self,
        text: str,
        split_char: str,
        escape_char: str,
    ) -> FlextResult[tuple[list[str], int]]:
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
                    source="flext-core/src/flext_core/_utilities/string_parser.py",
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
                    source="flext-core/src/flext_core/_utilities/string_parser.py",
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
            source="flext-core/src/flext_core/_utilities/string_parser.py",
        )
        components.append("".join(current))

        return FlextResult[tuple[list[str], int]].ok((components, escape_count))

    def _process_all_patterns(
        self,
        text: str,
        patterns: list[tuple[str, str] | tuple[str, str, int]],
    ) -> FlextResult[tuple[str, int]]:
        """Process all regex patterns and return final text and count."""
        result_text = text
        applied_patterns = 0

        for i, pattern_tuple in enumerate(patterns):
            # Extract pattern components from tuple
            pattern_result = self._extract_pattern_components(pattern_tuple)
            if pattern_result.is_failure:
                return FlextResult[tuple[str, int]].fail(
                    pattern_result.error
                    or "Unknown error extracting pattern components",
                )

            pattern, replacement, flags = pattern_result.unwrap()

            # Apply the pattern using model - import locally to avoid circular import
            from flext_core.utilities import FlextUtilities

            params_result = FlextUtilities.Model.from_kwargs(
                FlextModelsCollections.PatternApplicationParams,
                text=result_text,
                pattern=pattern,
                replacement=replacement,
                flags=flags,
                pattern_index=i,
                total_patterns=len(patterns),
            )
            if params_result.is_failure:
                return FlextResult[tuple[str, int]].fail(
                    params_result.error or "Unknown error creating params",
                )

            apply_result = self._apply_single_pattern(params_result.unwrap())
            if apply_result.is_failure:
                return FlextResult[tuple[str, int]].fail(
                    apply_result.error or "Unknown error applying pattern",
                )

            result_text = apply_result.unwrap()
            applied_patterns += 1

        return FlextResult[tuple[str, int]].ok((result_text, applied_patterns))


__all__ = ["FlextUtilitiesStringParser"]
