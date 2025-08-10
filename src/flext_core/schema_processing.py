"""Reusable schema and entry processing components.

Provides data validation, transformation, and pipeline processing patterns
for LDIF, ACL, and structured data across FLEXT ecosystem projects.

Classes:
    BaseEntry: Immutable value object for data representation.
    EntryValidator: Protocol for validation logic.
    BaseProcessor: Abstract processor with pipeline integration.
    ProcessingPipeline: Composable processing stages.

"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .result import FlextResult
from .typings import EntryT
from .value_objects import FlextValueObject

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def is_valid(self, entry: EntryT) -> bool:
        """Check if entry is valid."""
        ...

    def is_whitelisted(self, identifier: str) -> bool:
        """Check if identifier is whitelisted."""
        ...


class FlextBaseProcessor[EntryT](ABC):
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
        if not identifier_validation.success:
            return FlextResult.fail(
                identifier_validation.error or "Identifier validation failed",
            )

        identifier = identifier_validation.data
        if identifier is None:
            return FlextResult.fail("Identifier validation returned None")

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
        if not entry_validation.success:
            return entry_validation

        entry = entry_validation.data
        if entry is None:
            return FlextResult.fail("Entry validation returned None")

        self._extracted_entries.append(entry)
        return FlextResult.ok(entry)

    def _validate_identifier_extraction(self, content: str) -> FlextResult[str]:
        """Validate identifier extraction step."""
        identifier_result = self._extract_identifier(content)
        if not identifier_result.success:
            return FlextResult.fail(
                f"Failed to extract identifier: {identifier_result.error}",
            )

        identifier = identifier_result.data
        if identifier is None:
            return FlextResult.fail("Identifier extraction returned None")

        if self.validator and not self.validator.is_whitelisted(identifier):
            return FlextResult.fail(f"Identifier {identifier} not whitelisted")

        return FlextResult.ok(identifier)

    def _validate_entry_creation(
        self,
        entry_type: str,
        clean_content: str,
        content: str,
        identifier: str,
    ) -> FlextResult[EntryT]:
        """Validate entry creation step."""
        entry_result = self._create_entry(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if not entry_result.success:
            return entry_result

        entry = entry_result.data
        if entry is None:
            return FlextResult.fail("Entry creation returned None")

        if self.validator and not self.validator.is_valid(entry):
            return FlextResult.fail(f"Entry validation failed for {identifier}")

        return FlextResult.ok(entry)

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

            result = self.extract_entry_info(line, entry_type, prefix)
            if result.success:
                if result.data is not None:
                    results.append(result.data)
            else:
                errors.append(f"Line '{line[:50]}...': {result.error}")

        if errors and not results:
            return FlextResult.fail(f"All entries failed: {'; '.join(errors[:3])}")

        # Return success even if some entries failed (partial success)
        return FlextResult.ok(results)

    def get_extracted_entries(self) -> list[EntryT]:
        """Get all successfully extracted entries."""
        return self._extracted_entries.copy()

    def clear_extracted_entries(self) -> None:
        """Clear extracted entries cache."""
        self._extracted_entries.clear()


class FlextRegexProcessor(FlextBaseProcessor[EntryT]):
    """Regex-based processor for entries with pattern matching."""

    def __init__(
        self,
        identifier_pattern: str,
        validator: FlextEntryValidator | None = None,
    ) -> None:
        """Initialize with regex pattern for identifier extraction."""
        super().__init__(validator)
        self.identifier_pattern = re.compile(identifier_pattern)

    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier using regex pattern."""
        match = self.identifier_pattern.search(content)
        if not match:
            return FlextResult.fail(
                f"No identifier found matching pattern in: {content[:50]}",
            )

        return FlextResult.ok(match.group(1))


class FlextConfigAttributeValidator:
    """Utility for validating configuration attributes."""

    @staticmethod
    def has_attribute(config: object, attribute: str) -> bool:
        """Check if config has specified attribute."""
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
            # Convert object to dict for schema validator compatibility
            config_dict = getattr(config, "__dict__", {})
        else:
            config_dict = config

        # Simple validation: check if all required keys are present
        missing = [field for field in required if field not in config_dict]
        if missing:
            return FlextResult.fail(
                f"Missing required attributes: {', '.join(missing)}"
            )
        return FlextResult.ok(True)


class FlextBaseConfigManager:
    """Base configuration manager with attribute validation."""

    def __init__(self, config: object) -> None:
        """Initialize with configuration object."""
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
        return FlextResult.ok(True)


class FlextBaseSorter[T]:
    """Base sorter for entries with configurable sort key extraction."""

    def __init__(self, key_extractor: Callable[[T], object] | None = None) -> None:
        """Initialize with optional key extractor function."""
        self.key_extractor = key_extractor or (lambda x: x)

    def sort_entries(self, entries: list[T]) -> list[T]:
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
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Failed to write entries: {e}")


class FlextProcessingPipeline[T, U]:
    """Generic processing pipeline for chaining operations."""

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.steps: list[Callable[[object], FlextResult[object]]] = []

    def add_step(
        self,
        step: Callable[[T], FlextResult[U]],
    ) -> FlextProcessingPipeline[T, U]:
        """Add processing step to pipeline."""
        self.steps.append(step)  # type: ignore[arg-type]
        return self

    def process(self, input_data: T) -> FlextResult[U]:
        """Process input through all pipeline steps."""
        current_data: object = input_data
        for step in self.steps:
            result = step(current_data)
            if not result.success:
                return FlextResult.fail(result.error or "Processing step failed")
            current_data = result.data
        return FlextResult.ok(current_data)  # type: ignore[arg-type]


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    # Backward-compat export names expected by tests
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

# Backward-compat aliases
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
