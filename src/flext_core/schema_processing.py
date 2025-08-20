"""Reusable schema and entry processing components."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Generic, Protocol, TypeVar, cast

from flext_core.result import FlextResult
from flext_core.value_objects import FlextValueObject

# Define TypeVar locally
EntryT = TypeVar("EntryT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class FlextEntryType(Enum):
    """Base enumeration for entry types."""


class FlextBaseEntry(FlextValueObject):
    """Base entry value object for schema/ACL processing."""

    entry_type: str
    clean_content: str
    original_content: str
    identifier: str


class FlextEntryValidator(Protocol):
    """Protocol for entry validation."""

    def is_valid(self, entry: object) -> bool:
        """Check if entry is valid."""
        ...

    def is_whitelisted(self, identifier: str) -> bool:
        """Check if an identifier is whitelisted."""
        ...


class FlextBaseProcessor(ABC, Generic[EntryT]):  # noqa: UP046
    """Base processor for entries with configurable validation."""

    def __init__(self, validator: FlextEntryValidator | None = None) -> None:
        """Initialize processor with optional validator."""
        self.validator = validator
        self._extracted_entries: list[EntryT] = []

    @abstractmethod
    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier from content (e.g., OID, DN, etc.)."""
        ...

    @abstractmethod
    def _create_entry(
        self,
        entry_type: str,
        clean_content: str,
        original_content: str,
        identifier: str,
    ) -> FlextResult[EntryT]:
        """Create concrete entry instance."""
        ...

    def extract_entry_info(
        self,
        content: str,
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[EntryT]:
        """Extract entry information from content with type safety."""
        # Step 1: Extract and validate identifier
        identifier_validation = self._validate_identifier_extraction(content)
        if identifier_validation.is_failure:
            return FlextResult[EntryT].fail(
                identifier_validation.error or "Identifier validation failed",
            )

        identifier = identifier_validation.data

        # Step 2: Create and validate entry
        clean_content = (
            content.replace(f"{prefix}: ", "").strip() if prefix else content.strip()
        )

        entry_validation = self._validate_entry_creation(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if entry_validation.is_failure:
            return entry_validation

        entry = entry_validation.data
        if entry is None:
            return FlextResult[EntryT].fail("Entry validation returned None")

        self._extracted_entries.append(entry)
        return FlextResult[EntryT].ok(entry)

    def _validate_identifier_extraction(self, content: str) -> FlextResult[str]:
        """Validate identifier extraction step."""
        identifier_result = self._extract_identifier(content)
        if identifier_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to extract identifier: {identifier_result.error}",
            )

        identifier = identifier_result.data

        if self.validator and not self.validator.is_whitelisted(identifier):
            return FlextResult[str].fail(f"Identifier {identifier} not whitelisted")

        return FlextResult[str].ok(identifier)

    def _validate_entry_creation(
        self,
        entry_type: str,
        clean_content: str,
        content: str,
        identifier: str,
    ) -> FlextResult[EntryT]:
        """Validate an entry creation step."""
        entry_result = self._create_entry(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if entry_result.is_failure:
            return entry_result

        entry = entry_result.data
        if entry is None:
            return FlextResult[EntryT].fail("Entry creation returned None")

        if self.validator and not self.validator.is_valid(entry):
            return FlextResult[EntryT].fail(f"Entry validation failed for {identifier}")

        return FlextResult[EntryT].ok(entry)

    def process_content_lines(
        self,
        lines: list[str],
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[list[EntryT]]:
        """Process multiple content lines and return successful entries."""
        results: list[EntryT] = []
        errors: list[str] = []

        for line in lines:
            if not line.strip():
                continue

            result: FlextResult[EntryT] = self.extract_entry_info(
                line,
                entry_type,
                prefix,
            )
            if result.is_success:
                if result.data is not None:
                    results.append(result.data)
            else:
                errors.append(f"Line '{line[:50]}...': {result.error}")

        if errors and not results:
            return FlextResult[list[EntryT]].fail(
                f"All entries failed: {'; '.join(errors[:3])}"
            )

        # Return success even if some entries failed (partial success)
        return FlextResult[list[EntryT]].ok(results)

    def get_extracted_entries(self) -> list[EntryT]:
        """Get all successfully extracted entries."""
        return self._extracted_entries.copy()

    def clear_extracted_entries(self) -> None:
        """Clear extracted entries cache."""
        self._extracted_entries.clear()


class FlextRegexProcessor(FlextBaseProcessor[EntryT], ABC):
    """Regex-based processor for entries with pattern matching."""

    def __init__(
        self,
        identifier_pattern: str,
        validator: FlextEntryValidator | None = None,
    ) -> None:
        """Initialize with a regex pattern for identifier extraction."""
        super().__init__(validator)
        self.identifier_pattern = re.compile(identifier_pattern)

    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier using a regex pattern."""
        match = self.identifier_pattern.search(content)
        if not match:
            return FlextResult[str].fail(
                f"No identifier found matching pattern in: {content[:50]}",
            )

        return FlextResult[str].ok(match.group(1))


class FlextConfigAttributeValidator:
    """Utility for validating configuration attributes."""

    @staticmethod
    def has_attribute(config: object, attribute: str) -> bool:
        """Check if config has a specified attribute."""
        return hasattr(config, attribute)

    @staticmethod
    def has_rules_config(config: object) -> bool:
        """Check if config has rules_config attribute."""
        return hasattr(config, "rules_config")

    @staticmethod
    def validate_required_attributes(
        config: object,
        required: list[str],
    ) -> FlextResult[bool]:
        """Validate config has required attributes - FACADE to base_validation."""
        # ARCHITECTURAL DECISION: Use centralized validation to eliminate duplication
        if not isinstance(config, dict):
            # Convert an object to dict for schema validator compatibility
            config_dict = getattr(config, "__dict__", {})
        else:
            config_dict = cast("dict[str, object]", config)

        # Simple validation: check if all required keys are present
        missing = [field for field in required if field not in config_dict]
        if missing:
            return FlextResult[bool].fail(
                f"Missing required attributes: {', '.join(missing)}",
            )
        return FlextResult[bool].ok(True)  # noqa: FBT003


class FlextBaseConfigManager:
    """Base configuration manager with attribute validation."""

    def __init__(self, config: object) -> None:
        """Initialize with a configuration object."""
        self.config = config
        self.validator = FlextConfigAttributeValidator()

    def get_config_value(self, key: str, default: object = None) -> object:
        """Get configuration value with optional default."""
        return getattr(self.config, key, default)

    def validate_config(
        self,
        required_attrs: list[str] | None = None,
    ) -> FlextResult[bool]:
        """Validate config has required attributes - FACADE to base_validation."""
        if required_attrs:
            # ARCHITECTURAL DECISION: Use centralized validation directly
            # to eliminate duplication
            return self.validator.validate_required_attributes(
                self.config,
                required_attrs,
            )
        return FlextResult[bool].ok(True)  # noqa: FBT003


class FlextBaseSorter[EntryT]:
    """Base sorter for entries with configurable sort key extraction."""

    def __init__(self, key_extractor: Callable[[object], object] | None = None) -> None:
        """Initialize with optional key extractor function."""
        self.key_extractor: Callable[[object], object] = key_extractor or (lambda x: x)

    def sort_entries(self, entries: list[EntryT]) -> list[EntryT]:
        """Sort entries using the configured key extractor."""
        try:
            entries.sort(key=self.key_extractor)  # type: ignore[arg-type]
            return entries
        except Exception:
            # Return unsorted if sort fails
            return entries


class FlextBaseFileWriter(ABC):
    """Base file writer with common file operations."""

    @abstractmethod
    def write_header(self, output_file: object) -> None:
        """Write file header."""
        ...

    @abstractmethod
    def write_entry(self, output_file: object, entry: object) -> None:
        """Write single entry."""
        ...

    def write_entries(
        self,
        output_file: object,
        entries: list[object],
    ) -> FlextResult[None]:
        """Write multiple entries with header."""
        try:
            self.write_header(output_file)
            for entry in entries:
                self.write_entry(output_file, entry)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to write entries: {e}")


class FlextProcessingPipeline[InputT, OutputT]:
    """Generic processing pipeline for chaining operations."""

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.steps: list[Callable[[object], FlextResult[object]]] = []

    def add_step(
        self,
        step: Callable[[object], FlextResult[object]],
    ) -> FlextProcessingPipeline[InputT, OutputT]:
        """Add a processing step to a pipeline."""
        self.steps.append(step)
        return self

    def process(self, input_data: InputT) -> FlextResult[OutputT]:
        """Process input through all pipeline steps."""
        current_data: object = input_data
        for step in self.steps:
            result = step(current_data)
            if result.is_failure:
                return FlextResult[OutputT].fail(
                    result.error or "Processing step failed"
                )
            current_data = result.data
        return FlextResult[OutputT].ok(cast("OutputT", current_data))


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    # Backward-compatible export names
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "EntryType",
    "EntryValidator",
    # New names
    "FlextBaseEntry",
    "FlextBaseProcessor",
    "FlextEntryType",
    "FlextEntryValidator",
    "FlextProcessingPipeline",
    "FlextRegexProcessor",
    "ProcessingPipeline",
]

# Backward-compatible aliases
BaseEntry = FlextBaseEntry
EntryType = FlextEntryType
EntryValidator = FlextEntryValidator
BaseProcessor = FlextBaseProcessor
ProcessingPipeline = FlextProcessingPipeline
BaseFileWriter = FlextBaseFileWriter
RegexProcessor = FlextRegexProcessor
ConfigAttributeValidator = FlextConfigAttributeValidator
BaseConfigManager = FlextBaseConfigManager
BaseSorter = FlextBaseSorter
