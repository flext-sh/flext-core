"""Schema and Entry Processing - Reusable Components for LDIF/Schema Processing.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT

Provides base classes for schema/ACL processing that can be extended by
specific implementations.
Reduces complexity by extracting common patterns into reusable components.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeVar

from .result import FlextResult
from .value_objects import FlextValueObject

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")
EntryT = TypeVar("EntryT")


class EntryType(Enum):
    """Base enumeration for entry types."""


class BaseEntry(FlextValueObject):
    """Base entry value object for schema/ACL processing."""

    entry_type: str
    clean_content: str
    original_content: str
    identifier: str


class EntryValidator(Protocol[EntryT]):  # type: ignore[misc]
    """Protocol for entry validation."""

    def is_valid(self, entry: EntryT) -> bool:
        """Check if entry is valid."""
        ...

    def is_whitelisted(self, identifier: str) -> bool:
        """Check if identifier is whitelisted."""
        ...


class BaseProcessor[EntryT](ABC):
    """Base processor for schema/ACL entries with configurable validation."""

    def __init__(self, validator: EntryValidator[EntryT] | None = None) -> None:
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
        self, content: str, entry_type: str, prefix: str = ""
    ) -> FlextResult[EntryT]:
        """Extract entry information from content with type safety."""
        # Step 1: Extract and validate identifier
        identifier_validation = self._validate_identifier_extraction(content)
        if not identifier_validation.is_success:
            return FlextResult.fail(
                identifier_validation.error or "Identifier validation failed"
            )

        identifier = identifier_validation.data
        if identifier is None:
            return FlextResult.fail("Identifier validation returned None")

        # Step 2: Create and validate entry
        clean_content = (
            content.replace(f"{prefix}: ", "").strip() if prefix else content.strip()
        )

        entry_validation = self._validate_entry_creation(
            entry_type, clean_content, content, identifier
        )
        if not entry_validation.is_success:
            return entry_validation

        entry = entry_validation.data
        if entry is None:
            return FlextResult.fail("Entry validation returned None")

        self._extracted_entries.append(entry)
        return FlextResult.ok(entry)

    def _validate_identifier_extraction(self, content: str) -> FlextResult[str]:
        """Validate identifier extraction step."""
        identifier_result = self._extract_identifier(content)
        if not identifier_result.is_success:
            return FlextResult.fail(
                f"Failed to extract identifier: {identifier_result.error}"
            )

        identifier = identifier_result.data
        if identifier is None:
            return FlextResult.fail("Identifier extraction returned None")

        if self.validator and not self.validator.is_whitelisted(identifier):
            return FlextResult.fail(f"Identifier {identifier} not whitelisted")

        return FlextResult.ok(identifier)

    def _validate_entry_creation(
        self, entry_type: str, clean_content: str, content: str, identifier: str
    ) -> FlextResult[EntryT]:
        """Validate entry creation step."""
        entry_result = self._create_entry(
            entry_type, clean_content, content, identifier
        )
        if not entry_result.is_success:
            return entry_result

        entry = entry_result.data
        if entry is None:
            return FlextResult.fail("Entry creation returned None")

        if self.validator and not self.validator.is_valid(entry):
            return FlextResult.fail(f"Entry validation failed for {identifier}")

        return FlextResult.ok(entry)

    def process_content_lines(
        self, lines: list[str], entry_type: str, prefix: str = ""
    ) -> FlextResult[list[EntryT]]:
        """Process multiple content lines and return successful entries."""
        results: list[EntryT] = []
        errors: list[str] = []

        for line in lines:
            if not line.strip():
                continue

            result = self.extract_entry_info(line, entry_type, prefix)
            if result.is_success:
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


class RegexProcessor(BaseProcessor[EntryT]):
    """Regex-based processor for entries with pattern matching."""

    def __init__(
        self, identifier_pattern: str, validator: EntryValidator[EntryT] | None = None
    ) -> None:
        """Initialize with regex pattern for identifier extraction."""
        super().__init__(validator)
        self.identifier_pattern = re.compile(identifier_pattern)

    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier using regex pattern."""
        match = self.identifier_pattern.search(content)
        if not match:
            return FlextResult.fail(
                f"No identifier found matching pattern in: {content[:50]}"
            )

        return FlextResult.ok(match.group(1))


class ConfigAttributeValidator:
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
        config: object, required: list[str]
    ) -> FlextResult[bool]:
        """Validate that config has all required attributes."""
        missing = [attr for attr in required if not hasattr(config, attr)]
        if missing:
            return FlextResult.fail(f"Missing required attributes: {missing}")
        return FlextResult.ok(True)  # noqa: FBT003


class BaseConfigManager:
    """Base configuration manager with attribute validation."""

    def __init__(self, config: object) -> None:
        """Initialize with configuration object."""
        self.config = config
        self.validator = ConfigAttributeValidator()

    def get_config_value(self, key: str, default: object = None) -> object:
        """Get configuration value with optional default."""
        return getattr(self.config, key, default)

    def validate_config(
        self, required_attrs: list[str] | None = None
    ) -> FlextResult[bool]:
        """Validate configuration has required attributes."""
        if required_attrs:
            return self.validator.validate_required_attributes(
                self.config, required_attrs
            )
        return FlextResult.ok(True)  # noqa: FBT003


class BaseSorter[T]:
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


class BaseFileWriter(ABC):
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
        self, output_file: object, entries: list[object]
    ) -> FlextResult[None]:
        """Write multiple entries with header."""
        try:
            self.write_header(output_file)
            for entry in entries:
                self.write_entry(output_file, entry)
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Failed to write entries: {e}")


class ProcessingPipeline[T, U]:
    """Generic processing pipeline for chaining operations."""

    def __init__(self) -> None:
        """Initialize empty pipeline."""
        self.steps: list[Callable[[object], FlextResult[object]]] = []

    def add_step(self, step: Callable[[T], FlextResult[U]]) -> ProcessingPipeline[T, U]:
        """Add processing step to pipeline."""
        self.steps.append(step)  # type: ignore[arg-type]
        return self

    def process(self, input_data: T) -> FlextResult[U]:
        """Process input through all pipeline steps."""
        current_data = input_data
        for step in self.steps:
            result = step(current_data)
            if not result.is_success:
                return result  # type: ignore[return-value]
            current_data = result.data  # type: ignore[assignment]
        return FlextResult.ok(current_data)  # type: ignore[arg-type]
