"""FlextUtilities - Core Utilities Module.

This module provides essential utility functions and helper classes used
throughout the FLEXT ecosystem, including caching, configuration, data mapping,
domain operations, generators, reliability patterns, string parsing, and validation.

Scope: Utility functions for caching, configuration management, data mapping,
domain operations, ID generation, reliability patterns, string parsing,
and validation with consistent FlextResult-based error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core._utilities import (
    FlextUtilitiesCache,
    FlextUtilitiesConfiguration,
    FlextUtilitiesDataMapper,
    FlextUtilitiesDomain,
    FlextUtilitiesGenerators,
    FlextUtilitiesReliability,
    FlextUtilitiesStringParser,
    FlextUtilitiesTextProcessor,
    FlextUtilitiesTypeChecker,
    FlextUtilitiesTypeGuards,
    FlextUtilitiesValidation,
)
from flext_core._utilities.string_parser import ParseOptions
from flext_core.result import FlextResult


class FlextUtilities:
    """Utility functions for validation, generation, and data processing.

    Provides enterprise-grade utility functions for common operations
    throughout the FLEXT ecosystem, implementing structural typing via
    FlextProtocols.Utility (duck typing - no inheritance required).

    Core Features:
    - Cache: Data normalization and cache key generation
    - Validation: Comprehensive input validation
    - Generators: ID, UUID, timestamp generation
    - TextProcessor: Text cleaning and processing
    - TypeGuards: Runtime type checking
    - Reliability: Timeout and retry patterns
    - TypeChecker: Runtime type introspection
    - Configuration: Parameter access/manipulation
    """

    class Cache:
        """Cache utilities for data normalization and key generation."""

        normalize_component = staticmethod(FlextUtilitiesCache.normalize_component)
        sort_key = staticmethod(FlextUtilitiesCache.sort_key)
        sort_dict_keys = staticmethod(FlextUtilitiesCache.sort_dict_keys)
        clear_object_cache = staticmethod(FlextUtilitiesCache.clear_object_cache)
        has_cache_attributes = staticmethod(FlextUtilitiesCache.has_cache_attributes)
        generate_cache_key = staticmethod(FlextUtilitiesCache.generate_cache_key)

    class Validation:
        """Validation utilities for input checking."""

        # Delegate all static methods to FlextUtilitiesValidation
        validate_uri = staticmethod(FlextUtilitiesValidation.validate_uri)
        validate_port_number = staticmethod(
            FlextUtilitiesValidation.validate_port_number
        )
        validate_pipeline = staticmethod(FlextUtilitiesValidation.validate_pipeline)
        validate_required_string = staticmethod(
            FlextUtilitiesValidation.validate_required_string
        )
        validate_choice = staticmethod(FlextUtilitiesValidation.validate_choice)
        validate_length = staticmethod(FlextUtilitiesValidation.validate_length)
        validate_pattern = staticmethod(FlextUtilitiesValidation.validate_pattern)
        validate_non_negative = staticmethod(
            FlextUtilitiesValidation.validate_non_negative
        )
        validate_positive = staticmethod(FlextUtilitiesValidation.validate_positive)
        validate_range = staticmethod(FlextUtilitiesValidation.validate_range)
        validate_callable = staticmethod(FlextUtilitiesValidation.validate_callable)
        validate_timeout = staticmethod(FlextUtilitiesValidation.validate_timeout)
        validate_http_status_codes = staticmethod(
            FlextUtilitiesValidation.validate_http_status_codes
        )
        validate_iso8601_timestamp = staticmethod(
            FlextUtilitiesValidation.validate_iso8601_timestamp
        )
        validate_hostname = staticmethod(FlextUtilitiesValidation.validate_hostname)
        validate_identifier = staticmethod(FlextUtilitiesValidation.validate_identifier)
        validate_domain_event = staticmethod(
            FlextUtilitiesValidation.validate_domain_event
        )
        normalize_component = staticmethod(
            FlextUtilitiesValidation._normalize_component  # noqa: SLF001
        )
        sort_key = staticmethod(FlextUtilitiesValidation._sort_key)  # noqa: SLF001
        sort_dict_keys = staticmethod(FlextUtilitiesValidation._sort_dict_keys)  # noqa: SLF001
        _normalize_pydantic_value = staticmethod(
            FlextUtilitiesValidation._normalize_pydantic_value  # noqa: SLF001
        )
        _normalize_dataclass_value_instance = staticmethod(
            FlextUtilitiesValidation._normalize_dataclass_value_instance  # noqa: SLF001
        )
        _normalize_mapping = staticmethod(FlextUtilitiesValidation._normalize_mapping)  # noqa: SLF001
        _normalize_sequence = staticmethod(FlextUtilitiesValidation._normalize_sequence)  # noqa: SLF001
        _normalize_set = staticmethod(FlextUtilitiesValidation._normalize_set)  # noqa: SLF001
        _normalize_vars = staticmethod(FlextUtilitiesValidation._normalize_vars)  # noqa: SLF001
        _generate_key_pydantic = staticmethod(
            FlextUtilitiesValidation._generate_key_pydantic  # noqa: SLF001
        )

    class Generators:
        """ID and data generation utilities."""

        # Delegate all static methods to FlextUtilitiesGenerators
        generate_id = staticmethod(FlextUtilitiesGenerators.generate_id)
        generate_iso_timestamp = staticmethod(
            FlextUtilitiesGenerators.generate_iso_timestamp
        )
        generate_datetime_utc = staticmethod(
            FlextUtilitiesGenerators.generate_datetime_utc
        )
        generate_correlation_id = staticmethod(
            FlextUtilitiesGenerators.generate_correlation_id
        )
        generate_short_id = staticmethod(FlextUtilitiesGenerators.generate_short_id)
        generate_entity_id = staticmethod(FlextUtilitiesGenerators.generate_entity_id)
        generate_correlation_id_with_context = staticmethod(
            FlextUtilitiesGenerators.generate_correlation_id_with_context
        )
        generate_batch_id = staticmethod(FlextUtilitiesGenerators.generate_batch_id)
        generate_transaction_id = staticmethod(
            FlextUtilitiesGenerators.generate_transaction_id
        )
        generate_saga_id = staticmethod(FlextUtilitiesGenerators.generate_saga_id)
        generate_event_id = staticmethod(FlextUtilitiesGenerators.generate_event_id)
        generate_command_id = staticmethod(FlextUtilitiesGenerators.generate_command_id)
        generate_query_id = staticmethod(FlextUtilitiesGenerators.generate_query_id)
        generate_aggregate_id = staticmethod(
            FlextUtilitiesGenerators.generate_aggregate_id
        )
        generate_entity_version = staticmethod(
            FlextUtilitiesGenerators.generate_entity_version
        )
        ensure_id = staticmethod(FlextUtilitiesGenerators.ensure_id)
        ensure_trace_context = staticmethod(
            FlextUtilitiesGenerators.ensure_trace_context
        )
        ensure_dict = staticmethod(FlextUtilitiesGenerators.ensure_dict)
        generate_operation_id = staticmethod(
            FlextUtilitiesGenerators.generate_operation_id
        )
        create_dynamic_type_subclass = staticmethod(
            FlextUtilitiesGenerators.create_dynamic_type_subclass
        )

    class TextProcessor:
        """Text processing and cleaning utilities."""

        # Delegate all static methods to FlextUtilitiesTextProcessor
        clean_text = staticmethod(FlextUtilitiesTextProcessor.clean_text)
        truncate_text = staticmethod(FlextUtilitiesTextProcessor.truncate_text)
        safe_string = staticmethod(FlextUtilitiesTextProcessor.safe_string)

    class TypeGuards:
        """Runtime type checking utilities."""

        # Delegate all static methods to FlextUtilitiesTypeGuards
        is_string_non_empty = staticmethod(FlextUtilitiesTypeGuards.is_string_non_empty)
        is_dict_non_empty = staticmethod(FlextUtilitiesTypeGuards.is_dict_non_empty)
        is_list_non_empty = staticmethod(FlextUtilitiesTypeGuards.is_list_non_empty)

    class Reliability:
        """Reliability patterns with timeout and retry."""

        # Delegate all static methods to FlextUtilitiesReliability
        with_timeout = staticmethod(FlextUtilitiesReliability.with_timeout)
        retry = staticmethod(FlextUtilitiesReliability.retry)
        calculate_delay = staticmethod(FlextUtilitiesReliability.calculate_delay)
        with_retry = staticmethod(FlextUtilitiesReliability.with_retry)

    class TypeChecker:
        """Runtime type introspection for handlers."""

        # Delegate all methods to FlextUtilitiesTypeChecker
        compute_accepted_message_types = staticmethod(
            FlextUtilitiesTypeChecker.compute_accepted_message_types
        )
        can_handle_message_type = staticmethod(
            FlextUtilitiesTypeChecker.can_handle_message_type
        )

    class Configuration:
        """Configuration parameter utilities."""

        # Delegate all static methods to FlextUtilitiesConfiguration
        get_parameter = staticmethod(FlextUtilitiesConfiguration.get_parameter)
        set_parameter = staticmethod(FlextUtilitiesConfiguration.set_parameter)
        get_singleton = staticmethod(FlextUtilitiesConfiguration.get_singleton)
        set_singleton = staticmethod(FlextUtilitiesConfiguration.set_singleton)
        validate_config_class = staticmethod(
            FlextUtilitiesConfiguration.validate_config_class
        )
        create_settings_config = staticmethod(
            FlextUtilitiesConfiguration.create_settings_config
        )
        build_options_from_kwargs = staticmethod(
            FlextUtilitiesConfiguration.build_options_from_kwargs
        )

    class DataMapper:
        """Data mapping and transformation utilities."""

        # Delegate all static methods to FlextUtilitiesDataMapper
        convert_to_int_safe = staticmethod(FlextUtilitiesDataMapper.convert_to_int_safe)
        map_dict_keys = staticmethod(FlextUtilitiesDataMapper.map_dict_keys)
        build_flags_dict = staticmethod(FlextUtilitiesDataMapper.build_flags_dict)
        collect_active_keys = staticmethod(FlextUtilitiesDataMapper.collect_active_keys)
        transform_values = staticmethod(FlextUtilitiesDataMapper.transform_values)
        filter_dict = staticmethod(FlextUtilitiesDataMapper.filter_dict)
        invert_dict = staticmethod(FlextUtilitiesDataMapper.invert_dict)
        is_json_primitive = staticmethod(FlextUtilitiesDataMapper.is_json_primitive)
        convert_to_json_value = FlextUtilitiesDataMapper.convert_to_json_value
        convert_dict_to_json = FlextUtilitiesDataMapper.convert_dict_to_json
        convert_list_to_json = FlextUtilitiesDataMapper.convert_list_to_json
        # Type-safe converters
        ensure_str = staticmethod(FlextUtilitiesDataMapper.ensure_str)
        ensure_str_list = staticmethod(FlextUtilitiesDataMapper.ensure_str_list)
        ensure_str_or_none = staticmethod(FlextUtilitiesDataMapper.ensure_str_or_none)

    class Domain:
        """Domain-specific utilities."""

        # Delegate all static methods to FlextUtilitiesDomain
        compare_entities_by_id = staticmethod(
            FlextUtilitiesDomain.compare_entities_by_id
        )
        hash_entity_by_id = staticmethod(FlextUtilitiesDomain.hash_entity_by_id)
        compare_value_objects_by_value = staticmethod(
            FlextUtilitiesDomain.compare_value_objects_by_value
        )
        hash_value_object_by_value = staticmethod(
            FlextUtilitiesDomain.hash_value_object_by_value
        )
        validate_entity_has_id = staticmethod(
            FlextUtilitiesDomain.validate_entity_has_id
        )
        validate_value_object_immutable = staticmethod(
            FlextUtilitiesDomain.validate_value_object_immutable
        )

    class StringParser:
        """String parsing utilities - facade for FlextUtilitiesStringParser.

        Provides a public interface to string parsing functionality with FlextResult
        return types for railway-oriented error handling.
        """

        def __init__(self) -> None:
            """Initialize StringParser with internal implementation."""
            self._parser = FlextUtilitiesStringParser()

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
            """Parse delimited text with flexible options.

            Args:
                text: Text to parse
                delimiter: Delimiter character/string
                options: ParseOptions for advanced configuration
                strip: Strip whitespace from each component
                remove_empty: Remove empty components after stripping
                validator: Optional validation function for each component

            Returns:
                FlextResult[list[str]]: Parsed components or error

            """
            return self._parser.parse_delimited(
                text,
                delimiter,
                options=options,
                strip=strip,
                remove_empty=remove_empty,
                validator=validator,
            )

        def normalize_whitespace(
            self, text: str, pattern: str = r"\s+", replacement: str = " "
        ) -> FlextResult[str]:
            r"""Normalize whitespace in text.

            Args:
                text: Text to normalize
                pattern: Regex pattern to match (default: r"\s+")
                replacement: Replacement string (default: " ")

            Returns:
                FlextResult[str]: Normalized text or error

            """
            return self._parser.normalize_whitespace(
                text, pattern=pattern, replacement=replacement
            )

        def split_on_char_with_escape(
            self, text: str, split_char: str, escape_char: str = "\\"
        ) -> FlextResult[list[str]]:
            r"""Split text on character with escape support.

            Args:
                text: Text to split
                split_char: Character to split on
                escape_char: Escape character (default: "\\")

            Returns:
                FlextResult[list[str]]: Split components or error

            """
            return self._parser.split_on_char_with_escape(text, split_char, escape_char)

        def apply_regex_pipeline(
            self, text: str, patterns: list[tuple[str, str] | tuple[str, str, int]]
        ) -> FlextResult[str]:
            """Apply sequence of regex replacements to text.

            Args:
                text: Text to process
                patterns: List of (pattern, replacement) or (pattern, replacement, flags) tuples

            Returns:
                FlextResult[str]: Processed text or error

            """
            return self._parser.apply_regex_pipeline(text, patterns)

        def get_object_key(self, obj: object) -> str:
            """Get string key representation of object.

            Args:
                obj: Object to get key for

            Returns:
                str: String representation suitable for use as key

            """
            return self._parser.get_object_key(obj)


__all__ = ["FlextUtilities"]
