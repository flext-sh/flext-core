r"""Schema and entry processing system with validation and transformation.

Provides FlextProcessors with efficient data processing capabilities including entry
validation, regex-based pattern matching, configuration management, and pipeline processing.

Usage:
    processors = FlextProcessors()

    # Create and process entries
    entry_result = processors.create_entry({"id": "user123", "name": "John"}, "USER")
    if entry_result.success:
        entry = entry_result.unwrap()

    # Regex processing
    regex_proc = processors.create_regex_processor(r"user_(\d+)")

    # Pipeline processing
    pipeline = processors.create_processing_pipeline()
    results = pipeline.process(entries)

Features:
    - Entry validation with type safety
    - Regex-based pattern matching and extraction
    - Configurable processing pipelines
    - Service registry integration
    - FlextResult error handling
        validate_entry(entry) -> FlextResult[None] # Validate entry structure and content
        is_identifier_whitelisted(identifier) -> bool # Check whitelist membership

    BaseProcessor Methods:
        __init__(validator=None)       # Initialize with optional validator
        validate_input(entry) -> FlextResult[None] # Validate entry using configured validator
        transform_data(entry) -> FlextResult[Entry] # Transform entry (default: unchanged)
        process_data(entry) -> FlextResult[dict] # Extract information from entry
        process(request) -> FlextResult[Entry] # Process request into domain object
        build(domain, correlation_id="") -> dict # Build final result from domain

    RegexProcessor Methods:
        __init__(pattern, validator=None) # Initialize with regex pattern
        extract_identifier_from_content(content) -> FlextResult[str]
        validate_content_format(content) -> FlextResult[bool]
        process_data(entry) -> FlextResult[dict] # Process with regex-specific logic

    ConfigProcessor Methods:
        __init__()                     # Initialize with config cache
        validate_configuration_attribute(obj, attr_name, validator) -> FlextResult[bool]
        get_config_value(config, key) -> FlextResult[object]

    ValidatingProcessor Methods:
        __init__(name="entry_validator", validator=None) # Initialize with FlextHandlers
        handle(request) -> FlextResult[object] # Handle using FlextHandlers patterns
        process_entry(entry) -> FlextResult[Entry] # Process entry with validation

    ProcessingPipeline Methods:
        __init__(input_processor=None, output_processor=None) # Initialize pipeline
        add_step(step) -> ProcessingPipeline # Add processing step as handler
        process(data) -> FlextResult[object] # Process through handler pipeline

    Sorter Methods:
        sort_entries(entries, key_func=None, reverse=False) -> FlextResult[list[Entry]]

Usage Examples:
    Basic entry processing:
        processor = FlextProcessors()
        entry_result = FlextProcessors.create_entry({
            "entry_type": "user",
            "clean_content": "john_doe",
            "original_content": "John Doe",
            "identifier": "user_john_doe"
        })
        if entry_result.success:
            entry = entry_result.unwrap()
            validation_result = processor.process_entries([entry])

    Regex processing:
        regex_result = FlextProcessors.create_regex_processor(r"user_(\w+)")
        if regex_result.success:
            regex_processor = regex_result.unwrap()
            identifier_result = regex_processor.extract_identifier_from_content("user_john")

    Processing pipeline:
        pipeline_result = FlextProcessors.create_processing_pipeline(
            input_processor=lambda x: FlextResult.ok(x.upper()),
            output_processor=lambda x: FlextResult.ok(f"processed_{x}")
        )
        if pipeline_result.success:
            pipeline = pipeline_result.unwrap()
            result = pipeline.process("test_data")

Integration:
    FlextProcessors integrates with FlextResult for railway-oriented programming,
    FlextServices.ServiceRegistry for processor registration and discovery,
    FlextHandlers for processing patterns, and FlextValidations
    for efficient data validation across the FLEXT ecosystem.

"""

from __future__ import annotations

import re
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, cast

from pydantic import Field

from flext_core.constants import FlextConstants
from flext_core.handlers import FlextHandlers
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.services import FlextServices
from flext_core.typings import FlextTypes
from flext_core.validations import FlextValidations


class FlextProcessors:
    """Consolidated FLEXT processors system for entry processing, validation and transformation.

    Provides unified data processing capabilities including entry validation, regex pattern
    matching, configuration management, and pipeline processing. All processor types are
    organized as nested classes for consistent configuration and type safety.

    Key Features:
        - Entry validation with type safety
        - Regex-based pattern extraction
        - Pipeline processing with handlers
        - Configuration validation
        - Service registry integration
        - FlextResult error handling
    """

    # =============================================================================
    # ENTRY TYPE DEFINITIONS AND MANAGEMENT
    # =============================================================================

    class EntryType(StrEnum):
        """Base enumeration for standardized entry type classification."""

        # Standard entry types
        USER = "user"
        GROUP = "group"
        ROLE = "role"
        PERMISSION = "permission"
        CONFIG = "config"
        DATA = "data"
        UNKNOWN = "unknown"

    class Entry(FlextModels.Config):
        """Base entry value object for schema and data processing.

        Provides immutable, validated entry representations with standardized
        fields for type identification, content management, and identifier tracking.
        """

        # Entry identification and typing
        entry_type: str
        identifier: str

        # Content management
        clean_content: str
        original_content: str

        # Optional metadata
        metadata: dict[str, object] = Field(default_factory=dict)

        def __eq__(self, other: object) -> bool:
            """Entries are equal if they have same type and identifier."""
            if not isinstance(other, self.__class__):
                return False
            return (
                self.entry_type == other.entry_type
                and self.identifier == other.identifier
            )

        def __hash__(self) -> int:
            """Hash based on entry type and identifier."""
            return hash((self.entry_type, self.identifier))

    class EntryValidator(FlextValidations.Domain.BaseValidator):
        """Entry validator using FLEXT validation framework."""

        def __init__(self, whitelist: list[str] | None = None) -> None:
            """Initialize entry validator with optional identifier whitelist."""
            super().__init__()
            self.whitelist = whitelist or []

        def validate_entry(self, entry: FlextProcessors.Entry) -> FlextResult[None]:
            """Validate entry content and structure using FlextValidations."""
            # Validate entry_type is not empty
            if not entry.entry_type or not entry.entry_type.strip():
                return FlextResult[None].fail(
                    "Entry type is required and cannot be empty",
                )

            # Validate clean_content is not empty
            if not entry.clean_content or not entry.clean_content.strip():
                return FlextResult[None].fail(
                    "Clean content is required and cannot be empty",
                )

            # Validate identifier is not empty and within reasonable length
            if not entry.identifier or not entry.identifier.strip():
                return FlextResult[None].fail(
                    "Identifier is required and cannot be empty",
                )

            if len(entry.identifier) > FlextConstants.Validation.MAX_NAME_LENGTH:
                return FlextResult[None].fail(
                    f"Identifier must be {FlextConstants.Validation.MAX_NAME_LENGTH} characters or less",
                )

            return FlextResult[None].ok(None)

        def is_identifier_whitelisted(self, identifier: str) -> bool:
            """Check if identifier is in whitelist."""
            return not self.whitelist or identifier in self.whitelist

    # =============================================================================
    # PROCESSING COMPONENTS
    # =============================================================================

    class BaseProcessor(
        FlextServices.ServiceProcessor[
            "FlextProcessors.Entry",
            "FlextProcessors.Entry",
            dict[str, object],
        ],
    ):
        """Base processor using FLEXT service architecture for entry processing."""

        def __init__(
            self,
            validator: FlextProcessors.EntryValidator | None = None,
        ) -> None:
            """Initialize processor with optional validator using service patterns."""
            super().__init__()
            self.validator = validator

        def validate_input(self, entry: FlextProcessors.Entry) -> FlextResult[None]:
            """Validate entry input using configured validator."""
            if self.validator is None:
                return FlextResult[None].ok(None)
            return self.validator.validate_entry(entry)

        def transform_data(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[FlextProcessors.Entry]:
            """Transform entry data - default implementation returns entry unchanged."""
            return FlextResult[FlextProcessors.Entry].ok(entry)

        def process_data(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[dict[str, object]]:
            """Process entry and extract information following service patterns."""
            try:
                info = {
                    "entry_type": entry.entry_type,
                    "identifier": entry.identifier,
                    "content_length": len(entry.clean_content),
                    "has_metadata": bool(entry.metadata),
                }
                return FlextResult[dict[str, object]].ok(dict(info))
            except Exception as e:
                return FlextResult[dict[str, object]].fail(
                    f"Failed to process entry: {e}",
                )

        def validate_entry(self, entry: FlextProcessors.Entry) -> FlextResult[None]:
            """Legacy method for backward compatibility."""
            return self.validate_input(entry)

        def extract_info_from_entry(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[dict[str, object]]:
            """Legacy method for backward compatibility."""
            return self.process_data(entry)

        def process(
            self,
            request: FlextProcessors.Entry,
        ) -> FlextResult[FlextProcessors.Entry]:
            """Process request into domain object (required by ServiceProcessor)."""
            # Validate input first
            validation_result = self.validate_input(request)
            if validation_result.is_failure:
                return FlextResult[FlextProcessors.Entry].fail(
                    validation_result.error or "Validation failed",
                )

            # Transform the data
            return self.transform_data(request)

        def build(
            self,
            domain: FlextProcessors.Entry,
            *,
            correlation_id: str = "",
        ) -> dict[str, object]:
            """Build final result from domain object (required by ServiceProcessor)."""
            # Use process_data to build the final result
            result = self.process_data(domain)
            if result.is_success:
                info = result.unwrap()
                # Add correlation_id if provided
                if correlation_id:
                    info["correlation_id"] = correlation_id
                return info
            # Return empty dict on failure (could be improved)
            return {"error": result.error or "Processing failed"}

    class DefaultProcessor(BaseProcessor):
        """Default concrete processor implementation."""

        def __init__(
            self,
            validator: FlextProcessors.EntryValidator | None = None,
        ) -> None:
            """Initialize default processor."""
            super().__init__(validator)

        def process_data(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[dict[str, object]]:
            """Default processing implementation."""
            return super().process_data(entry)

    class RegexProcessor(BaseProcessor):
        """Regex-based processor for pattern matching and extraction."""

        def __init__(
            self,
            pattern: str,
            validator: FlextProcessors.EntryValidator | None = None,
        ) -> None:
            """Initialize regex processor with pattern."""
            super().__init__(validator)
            try:
                self.pattern = re.compile(pattern)
            except re.error:
                # Fallback to basic pattern
                self.pattern = re.compile(r".*")

        def extract_identifier_from_content(self, content: str) -> FlextResult[str]:
            """Extract identifier from content using regex pattern."""
            try:
                match = self.pattern.search(content)
                if match:
                    # Use first group if available, otherwise full match
                    identifier = match.group(1) if match.groups() else match.group(0)
                    return FlextResult[str].ok(identifier.strip())
                return FlextResult[str].fail("No identifier found in content")
            except Exception as e:
                return FlextResult[str].fail(f"Regex extraction failed: {e}")

        def validate_content_format(self, content: str) -> FlextResult[bool]:
            """Validate content format against regex pattern."""
            try:
                matches = bool(self.pattern.search(content))
                return FlextResult[bool].ok(matches)
            except Exception as e:
                return FlextResult[bool].fail(f"Content validation failed: {e}")

        def process_data(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[dict[str, object]]:
            """Process entry with regex-specific logic."""
            # First do the base processing
            base_result = super().process_data(entry)
            if base_result.is_failure:
                return base_result

            # Add regex-specific processing
            info = base_result.unwrap()

            # Extract identifier using regex
            identifier_result = self.extract_identifier_from_content(
                entry.clean_content,
            )
            if identifier_result.is_success:
                info["extracted_identifier"] = identifier_result.unwrap()

            # Validate content format
            format_result = self.validate_content_format(entry.clean_content)
            if format_result.is_success:
                info["content_matches_pattern"] = format_result.unwrap()
            return FlextResult[dict[str, object]].ok(info)

    class ConfigProcessor:
        """Configuration processor for handling config validation and management."""

        def __init__(self) -> None:
            """Initialize configuration processor."""
            self.config_cache: dict[str, object] = {}

        def validate_configuration_attribute(
            self,
            obj: object,
            attribute_name: str,
            validator: Callable[[object], bool],
        ) -> FlextResult[bool]:
            """Validate a configuration attribute."""
            try:
                if not hasattr(obj, attribute_name):
                    return FlextResult[bool].fail(
                        f"Attribute '{attribute_name}' not found",
                    )

                value = getattr(obj, attribute_name)
                is_valid = validator(value)
                return FlextResult[bool].ok(is_valid)
            except Exception as e:
                return FlextResult[bool].fail(f"Attribute validation failed: {e}")

        def get_config_value(
            self,
            config: dict[str, object],
            key: str,
        ) -> FlextResult[object]:
            """Get configuration value with validation."""
            try:
                if key not in config:
                    return FlextResult[object].fail(
                        f"Configuration key '{key}' not found",
                    )
                return FlextResult[object].ok(config[key])
            except Exception as e:
                return FlextResult[object].fail(f"Config retrieval failed: {e}")

    class ValidatingProcessor(FlextHandlers.Implementation.BasicHandler):
        """Entry processor with validation using FlextHandlers."""

        def __init__(
            self,
            name: str = "entry_validator",
            validator: FlextProcessors.EntryValidator | None = None,
        ) -> None:
            """Initialize validating processor with FlextHandlers patterns."""
            super().__init__(name)
            self.validator = validator

        def handle(self, request: object) -> FlextResult[object]:
            """Handle entry validation using FlextHandlers patterns."""
            if self.validator is None:
                return FlextResult[object].ok(request)

            if isinstance(request, FlextProcessors.Entry):
                validation_result = self.validator.validate_entry(request)
                if validation_result.is_failure:
                    return FlextResult[object].fail(
                        validation_result.error or "Validation failed",
                    )
                return FlextResult[object].ok(request)

            return FlextResult[object].fail("Invalid request type for entry validation")

        def process_entry(
            self,
            entry: FlextProcessors.Entry,
        ) -> FlextResult[FlextProcessors.Entry]:
            """Process entry using handler validation patterns."""
            # Use handler's validation process
            result = self.handle(entry)
            if result.is_failure:
                return FlextResult[FlextProcessors.Entry].fail(
                    result.error or "Validation failed",
                )

            # Return validated entry
            validated_entry = result.unwrap()
            if isinstance(validated_entry, FlextProcessors.Entry):
                return FlextResult[FlextProcessors.Entry].ok(validated_entry)

            return FlextResult[FlextProcessors.Entry].fail(
                "Handler returned invalid type",
            )

    class ProcessingPipeline:
        """Processing pipeline using handler patterns."""

        def __init__(
            self,
            input_processor: Callable[[object], FlextResult[object]] | None = None,
            output_processor: Callable[[object], FlextResult[object]] | None = None,
        ) -> None:
            """Initialize processing pipeline with minimal overhead.

            Keeps the same public API while avoiding dynamic handler wrappers
            for each step to reduce per-call overhead in hot paths.
            """

            def default_processor(x: object) -> FlextResult[object]:
                return FlextResult[object].ok(x)

            self.input_processor = input_processor or default_processor
            self.output_processor = output_processor or default_processor
            # Store steps as plain callables for performance
            self._steps: list[Callable[[object], FlextResult[object]]] = []

        def add_step(
            self,
            step: Callable[[object], FlextResult[object]],
        ) -> FlextProcessors.ProcessingPipeline:
            """Add a processing step.

            Steps are stored directly and invoked without intermediate wrappers
            to minimize function call overhead while preserving behavior.
            """
            self._steps.append(step)
            return self

        def process(self, data: object) -> FlextResult[object]:
            """Process data through the pipeline with minimal overhead."""
            try:
                # Input processing
                current_result = self.input_processor(data)
                if current_result.is_failure:
                    return current_result

                # Process steps directly
                current_data = current_result.unwrap()
                for step in self._steps:
                    step_result = step(current_data)
                    if step_result.is_failure:
                        return step_result
                    current_data = step_result.unwrap()

                # Output processing
                return self.output_processor(current_data)
            except Exception as e:
                return FlextResult[object].fail(f"Pipeline processing failed: {e}")

    class Sorter:
        """Entry sorting functionality."""

        @staticmethod
        def sort_entries(
            entries: list[FlextProcessors.Entry],
            key_func: Callable[[FlextProcessors.Entry], str] | None = None,
            *,
            reverse: bool = False,
        ) -> FlextResult[list[FlextProcessors.Entry]]:
            """Sort entries using key function."""

            def default_key_func(entry: FlextProcessors.Entry) -> str:
                return entry.identifier

            try:
                actual_key_func = key_func if key_func is not None else default_key_func
                sorted_entries = sorted(entries, key=actual_key_func, reverse=reverse)
                return FlextResult[list[FlextProcessors.Entry]].ok(sorted_entries)
            except Exception as e:
                return FlextResult[list[FlextProcessors.Entry]].fail(
                    f"Sorting failed: {e}",
                )

    class FileWriter(Protocol):
        """Protocol for file writing operations."""

        def write_header(self, header: str) -> FlextResult[None]:
            """Write header to file."""
            ...

        def write_entry(self, entry: FlextProcessors.Entry) -> FlextResult[None]:
            """Write entry to file."""
            ...

    # =============================================================================
    # FACTORY METHODS AND UTILITIES
    # =============================================================================

    @classmethod
    def create_entry(
        cls,
        data: dict[str, object],
        entry_type: str | None = None,
    ) -> FlextResult[FlextProcessors.Entry]:
        """Create entry instance with validation."""
        try:
            # Set defaults
            entry_data = dict(data)
            entry_data.setdefault("entry_type", entry_type or cls.EntryType.UNKNOWN)
            entry_data.setdefault("metadata", {})

            # Ensure required fields
            required_fields = [
                "entry_type",
                "clean_content",
                "original_content",
                "identifier",
            ]
            missing_fields = [
                field for field in required_fields if field not in entry_data
            ]
            if missing_fields:
                return FlextResult[FlextProcessors.Entry].fail(
                    f"Missing required fields: {missing_fields}",
                )

            entry = cls.Entry.model_validate(entry_data)
            return FlextResult[FlextProcessors.Entry].ok(entry)

        except Exception as e:
            return FlextResult[FlextProcessors.Entry].fail(
                f"Entry creation failed: {e}",
            )

    @classmethod
    def create_regex_processor(
        cls,
        pattern: str,
        validator: FlextProcessors.EntryValidator | None = None,
    ) -> FlextResult[FlextProcessors.RegexProcessor]:
        """Create regex processor with pattern validation."""
        try:
            processor = cls.RegexProcessor(pattern, validator)
            return FlextResult[FlextProcessors.RegexProcessor].ok(processor)
        except Exception as e:
            return FlextResult[FlextProcessors.RegexProcessor].fail(
                f"Regex processor creation failed: {e}",
            )

    @classmethod
    def create_processing_pipeline(
        cls,
        input_processor: Callable[
            [object],
            FlextResult[object],
        ]
        | None = None,
        output_processor: Callable[
            [object],
            FlextResult[object],
        ]
        | None = None,
    ) -> FlextResult[FlextProcessors.ProcessingPipeline]:
        """Create processing pipeline with optional processors."""
        try:
            pipeline = cls.ProcessingPipeline(input_processor, output_processor)
            return FlextResult[FlextProcessors.ProcessingPipeline].ok(pipeline)
        except Exception as e:
            return FlextResult[FlextProcessors.ProcessingPipeline].fail(
                f"Pipeline creation failed: {e}",
            )

    @classmethod
    def validate_configuration(cls, config: object) -> FlextResult[dict[str, object]]:
        """Validate configuration dictionary."""
        try:
            # Basic validation - can be extended
            if not isinstance(config, dict):
                return FlextResult[dict[str, object]].fail(
                    "Configuration must be a dictionary",
                )

            # Type narrowing: we know config is dict at this point
            typed_config = cast(
                "dict[str, object]",
                config,
            )  # Cast to proper type after isinstance check

            # Validate configuration structure
            for key, value in typed_config.items():
                # Validate value type (allow None)
                if value is not None and not isinstance(
                    value,
                    (str, int, float, bool, list, dict),
                ):
                    return FlextResult[dict[str, object]].fail(
                        f"Configuration value for '{key}' must be a basic type",
                    )

            return FlextResult[dict[str, object]].ok(typed_config)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Configuration validation failed: {e}",
            )

    # =============================================================================
    # PROCESSING OPERATIONS
    # =============================================================================

    def __init__(self) -> None:
        """Initialize processors system with FlextServices.ServiceRegistry."""
        self.service_registry = FlextServices.ServiceRegistry()
        self.config_processor = self.ConfigProcessor()

    def register_processor(
        self,
        name: str,
        processor: FlextProcessors.BaseProcessor,
    ) -> FlextResult[None]:
        """Register a named processor using service registry patterns."""
        try:
            service_info: dict[str, object] = {
                "name": name,
                "type": "processor",
                "instance": processor,
                "status": "active",
            }
            registration_result = self.service_registry.register(service_info)
            if registration_result.is_failure:
                return FlextResult[None].fail(
                    f"Service registration failed: {registration_result.error}",
                )
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Processor registration failed: {e}")

    def get_processor(self, name: str) -> FlextResult[FlextProcessors.BaseProcessor]:
        """Get registered processor using service discovery patterns."""
        try:
            discovery_result = self.service_registry.discover(name)
            if discovery_result.is_failure:
                return FlextResult[FlextProcessors.BaseProcessor].fail(
                    f"Processor '{name}' not found: {discovery_result.error}",
                )

            service_info = discovery_result.unwrap()
            if "instance" in service_info:
                processor = service_info["instance"]
                if isinstance(processor, FlextProcessors.BaseProcessor):
                    return FlextResult[FlextProcessors.BaseProcessor].ok(processor)
            return FlextResult[FlextProcessors.BaseProcessor].fail(
                f"Service '{name}' is not a valid processor",
            )
        except Exception as e:
            return FlextResult[FlextProcessors.BaseProcessor].fail(
                f"Processor retrieval failed: {e}",
            )

    def process_entries(
        self,
        entries: list[FlextProcessors.Entry],
        processor_name: str | None = None,
    ) -> FlextResult[list[FlextProcessors.Entry]]:
        """Process a list of entries."""
        try:
            if processor_name:
                processor_result = self.get_processor(processor_name)
                if processor_result.is_failure:
                    return FlextResult[list[FlextProcessors.Entry]].fail(
                        processor_result.error or "Processor not found",
                    )
                processor = processor_result.unwrap()
            else:
                processor = self.DefaultProcessor()

            processed_entries: list[FlextProcessors.Entry] = []
            for entry in entries:
                validation_result = processor.validate_entry(entry)
                if validation_result.is_success:
                    processed_entries.append(entry)

            return FlextResult[list[FlextProcessors.Entry]].ok(processed_entries)
        except Exception as e:
            return FlextResult[list[FlextProcessors.Entry]].fail(
                f"Entries processing failed: {e}",
            )

    # =============================================================================
    # FLEXT PROCESSORS CONFIGURATION METHODS
    # =============================================================================

    @classmethod
    def configure_processors_system(
        cls,
        config: FlextTypes.Config.ConfigDict,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Configure processors system with validation."""
        try:
            validated_config = dict(config)

            # Set default values for processors-specific settings
            validated_config.setdefault("enable_regex_caching", True)
            validated_config.setdefault("enable_pipeline_validation", True)
            validated_config.setdefault("max_processing_errors", 100)
            validated_config.setdefault("enable_entry_deduplication", True)
            validated_config.setdefault("processing_timeout_seconds", 30)

            return FlextResult[FlextTypes.Config.ConfigDict].ok(validated_config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to configure processors system: {e}",
            )

    @classmethod
    def get_processors_system_config(
        cls,
    ) -> FlextResult[FlextTypes.Config.ConfigDict]:
        """Get current processors system configuration."""
        try:
            config: FlextTypes.Config.ConfigDict = {
                # Core configuration
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                # Processors-specific settings
                "enable_regex_caching": True,
                "enable_pipeline_validation": True,
                "max_processing_errors": 100,
                "enable_entry_deduplication": True,
                "processing_timeout_seconds": 30,
                # Runtime information
                "supported_processor_types": [
                    "BaseProcessor",
                    "RegexProcessor",
                    "ConfigProcessor",
                    "ProcessingPipeline",
                ],
                "supported_entry_types": [
                    "USER",
                    "GROUP",
                    "ROLE",
                    "PERMISSION",
                    "CONFIG",
                    "DATA",
                ],
                "processing_features": [
                    "entry_validation",
                    "regex_processing",
                    "pipeline_processing",
                    "configuration_validation",
                    "entry_sorting",
                ],
            }

            return FlextResult[FlextTypes.Config.ConfigDict].ok(config)

        except Exception as e:
            return FlextResult[FlextTypes.Config.ConfigDict].fail(
                f"Failed to get processors system config: {e}",
            )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextProcessors",
]
