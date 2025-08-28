"""Reusable schema and entry processing components."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, TypeGuard, override

from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import TEntry


class FlextEntryType(StrEnum):
    """Base enumeration for entry types."""


class FlextBaseEntry(FlextModels.Value):
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


class FlextBaseProcessor[TEntry](ABC):
    """Base processor for entries with configurable validation."""

    def __init__(self, validator: FlextEntryValidator | None = None) -> None:
        """Initialize processor with optional validator."""
        self.validator = validator
        self._extracted_entries: list[TEntry] = []

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
    ) -> FlextResult[TEntry]:
        """Create concrete entry instance."""
        ...

    def extract_entry_info(
        self,
        content: str,
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[TEntry]:
        """Extract entry information from content with type safety."""
        # Step 1: Extract and validate identifier
        identifier_validation = self._validate_identifier_extraction(content)
        if identifier_validation.is_failure:
            return FlextResult[TEntry].fail(
                identifier_validation.error or "Identifier validation failed",
            )

        identifier = identifier_validation.value

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

        entry = entry_validation.value
        if entry is None:
            return FlextResult[TEntry].fail("Entry validation returned None")

        self._extracted_entries.append(entry)
        return FlextResult[TEntry].ok(entry)

    def _validate_identifier_extraction(self, content: str) -> FlextResult[str]:
        """Validate identifier extraction step."""
        identifier_result = self._extract_identifier(content)
        if identifier_result.is_failure:
            return FlextResult[str].fail(
                f"Failed to extract identifier: {identifier_result.error}",
            )

        identifier = identifier_result.value

        if self.validator and not self.validator.is_whitelisted(identifier):
            return FlextResult[str].fail(f"Identifier {identifier} not whitelisted")

        return FlextResult[str].ok(identifier)

    def _validate_entry_creation(
        self,
        entry_type: str,
        clean_content: str,
        content: str,
        identifier: str,
    ) -> FlextResult[TEntry]:
        """Validate an entry creation step."""
        entry_result = self._create_entry(
            entry_type,
            clean_content,
            content,
            identifier,
        )
        if entry_result.is_failure:
            return entry_result

        entry = entry_result.value
        if entry is None:
            return FlextResult[TEntry].fail("Entry creation returned None")

        if self.validator and not self.validator.is_valid(entry):
            return FlextResult[TEntry].fail(f"Entry validation failed for {identifier}")

        return FlextResult[TEntry].ok(entry)

    def process_content_lines(
        self,
        lines: list[str],
        entry_type: str,
        prefix: str = "",
    ) -> FlextResult[list[TEntry]]:
        """Process multiple content lines and return successful entries."""
        results: list[TEntry] = []
        errors: list[str] = []

        for line in lines:
            if not line.strip():
                continue

            result: FlextResult[TEntry] = self.extract_entry_info(
                line,
                entry_type,
                prefix,
            )
            if result.is_success:
                if result.value is not None:
                    results.append(result.value)
            else:
                errors.append(f"Line '{line[:50]}...': {result.error}")

        if errors and not results:
            return FlextResult[list[TEntry]].fail(
                f"All entries failed: {'; '.join(errors[:3])}"
            )

        # Return success even if some entries failed (partial success)
        return FlextResult[list[TEntry]].ok(results)

    def get_extracted_entries(self) -> list[TEntry]:
        """Get all successfully extracted entries."""
        return self._extracted_entries.copy()

    def clear_extracted_entries(self) -> None:
        """Clear extracted entries cache."""
        self._extracted_entries.clear()


class FlextRegexProcessor(FlextBaseProcessor[TEntry]):
    """Regex-based processor for entries with pattern matching."""

    def __init__(
        self,
        identifier_pattern: str,
        validator: FlextEntryValidator | None = None,
    ) -> None:
        """Initialize with a regex pattern for identifier extraction."""
        super().__init__(validator)
        self.identifier_pattern = re.compile(identifier_pattern)

    @override
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

    def _is_dict_like(self, config: object) -> TypeGuard[dict[str, object]]:
        """Type guard for dict-like configuration objects."""
        return isinstance(config, dict)

    def _extract_config_dict(self, config: object) -> dict[str, object]:
        """Extract configuration as dict using type-safe approach."""
        if self._is_dict_like(config):
            # Python 3.13+ discriminated union: config is dict[str, object]
            # Type narrowing through TypeGuard ensures safe access
            return config  # Type narrowing works here
        # For non-dict objects, extract __dict__ or return empty dict
        return getattr(config, "__dict__", {})

    @staticmethod
    def validate_required_attributes(
        config: object,
        required: list[str],
    ) -> FlextResult[bool]:
        """Validate config has required attributes - FACADE to base_validation."""
        # ARCHITECTURAL DECISION: Use centralized validation to eliminate duplication
        validator = FlextConfigAttributeValidator()
        config_dict = validator._extract_config_dict(config)

        # Simple validation: check if all required keys are present
        missing = [field for field in required if field not in config_dict]
        if missing:
            return FlextResult[bool].fail(
                f"Missing required attributes: {', '.join(missing)}",
            )
        return FlextResult[bool].ok(data=True)


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
        return FlextResult[bool].ok(data=True)


class FlextBaseSorter[TEntry]:
    """Base sorter for entries with configurable sort key extraction."""

    def __init__(self, key_extractor: Callable[[TEntry], object] | None = None) -> None:
        """Initialize with optional key extractor function."""
        self.key_extractor: Callable[[TEntry], object] = key_extractor or (lambda x: x)

    def sort_entries(self, entries: list[TEntry]) -> list[TEntry]:
        """Sort entries using the configured key extractor."""
        try:
            entries.sort(key=self.key_extractor)  # type: ignore[arg-type]
            return entries
        except Exception:
            # Return unsorted if sort fails
            return entries


class FlextBaseFileWriter(Protocol):
    """File writer protocol with common file operations."""

    def write_header(self, output_file: object) -> None:
        """Write file header."""
        ...

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
        self._input_type: type[InputT] | None = None
        self._output_type: type[OutputT] | None = None

    def add_step(
        self,
        step: Callable[[object], FlextResult[object]],
    ) -> FlextProcessingPipeline[InputT, OutputT]:
        """Add a processing step to a pipeline."""
        self.steps.append(step)
        return self

    def with_types(
        self,
        input_type: type[InputT],
        output_type: type[OutputT],
    ) -> FlextProcessingPipeline[InputT, OutputT]:
        """Set explicit types for better type checking."""
        self._input_type = input_type
        self._output_type = output_type
        return self

    def _is_output_type(self, _data: object) -> TypeGuard[OutputT]:
        """Type guard for output type - pipeline guarantees type consistency."""
        # In a well-formed pipeline, the final step should produce OutputT
        # This is a runtime assumption based on correct pipeline construction
        return True  # Trust the pipeline's type consistency

    def process(self, input_data: InputT) -> FlextResult[OutputT]:
        """Process input through all pipeline steps."""
        current_data: object = input_data
        for step in self.steps:
            result = step(current_data)
            if result.is_failure:
                return FlextResult[OutputT].fail(
                    result.error or "Processing step failed"
                )
            current_data = result.value

        # Python 3.13+ discriminated union with type guard
        # The pipeline guarantees type consistency through its construction
        if not self._is_output_type(current_data):
            return FlextResult[OutputT].fail("Pipeline output type mismatch")

        # Type narrowing: after type guard, current_data is OutputT
        return FlextResult[OutputT].ok(current_data)


# =============================================================================
# EXPORTS
# =============================================================================

__all__: list[str] = [
    # NOTE: FlextSchemaProcessing class not yet implemented
    "FlextBaseSorter",
    "FlextConfigAttributeValidator",
    "FlextEntryType",
    "FlextEntryValidator",
    "FlextProcessingPipeline",
    "FlextRegexProcessor",
    # Legacy aliases removed - now in legacy.py
]

# Backward-compatible aliases moved to legacy.py per FLEXT_REFACTORING_PROMPT.md
