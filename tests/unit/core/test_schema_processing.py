"""Comprehensive tests for schema_processing module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Tests all schema processing functionality including base classes, processors,
validators, and pipeline components for complete coverage.
"""

from __future__ import annotations

from enum import Enum

import pytest

from flext_core import (
    BaseConfigManager,
    BaseEntry,
    BaseFileWriter,
    BaseProcessor,
    BaseSorter,
    ConfigAttributeValidator,
    EntryType,
    FlextResult,
    FlextValueObject,
    ProcessingPipeline,
    RegexProcessor,
)


class MockEntry(FlextValueObject):
    """Mock entry for testing."""

    entry_type: str
    clean_content: str
    original_content: str
    identifier: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Mock business rules validation - always returns success."""
        return FlextResult.ok(None)


class ConcreteRegexProcessor(RegexProcessor[MockEntry]):
    """Concrete implementation of RegexProcessor for testing."""

    def _create_entry(
        self,
        entry_type: str,
        clean_content: str,
        original_content: str,
        identifier: str,
    ) -> FlextResult[MockEntry]:
        """Create concrete MockEntry instance."""
        try:
            entry = MockEntry(
                entry_type=entry_type,
                clean_content=clean_content,
                original_content=original_content,
                identifier=identifier,
            )
            return FlextResult.ok(entry)
        except Exception as e:
            return FlextResult.fail(f"Failed to create entry: {e}")


class ConcreteBaseProcessor(BaseProcessor[MockEntry]):
    """Concrete implementation of BaseProcessor for testing."""

    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract identifier using simple string parsing."""
        if "id:" in content:
            parts = content.split("id:")
            if len(parts) > 1:
                identifier = parts[1].split()[0] if parts[1].split() else "unknown"
                return FlextResult.ok(identifier)
        return FlextResult.fail("No identifier found")

    def _create_entry(
        self,
        entry_type: str,
        clean_content: str,
        original_content: str,
        identifier: str,
    ) -> FlextResult[MockEntry]:
        """Create concrete MockEntry instance."""
        try:
            entry = MockEntry(
                entry_type=entry_type,
                clean_content=clean_content,
                original_content=original_content,
                identifier=identifier,
            )
            return FlextResult.ok(entry)
        except Exception as e:
            return FlextResult.fail(f"Failed to create entry: {e}")


class ConcreteBaseEntry(BaseEntry):
    """Concrete implementation of BaseEntry for testing."""

    def validate_business_rules(self) -> FlextResult[None]:
        """Mock domain validation - always returns success."""
        return FlextResult.ok(data=None)


class MockEntryValidator:
    """Mock validator for testing."""

    def __init__(self, *, is_valid: bool = True, is_whitelisted: bool = True) -> None:
        """Initialize mock validator."""
        self.is_valid_result = is_valid
        self.is_whitelisted_result = is_whitelisted

    def is_valid(self, entry: MockEntry) -> bool:  # noqa: ARG002
        """Mock validation."""
        return self.is_valid_result

    def is_whitelisted(self, identifier: str) -> bool:  # noqa: ARG002
        """Mock whitelist check."""
        return self.is_whitelisted_result


class MockProcessor(BaseProcessor[MockEntry]):
    """Mock processor for testing."""

    def _extract_identifier(self, content: str) -> FlextResult[str]:
        """Extract mock identifier."""
        if "invalid" in content:
            return FlextResult.fail("Invalid content")
        return FlextResult.ok(f"id_{content.rsplit(maxsplit=1)[-1]}")

    def _create_entry(
        self,
        entry_type: str,
        clean_content: str,
        original_content: str,
        identifier: str,
    ) -> FlextResult[MockEntry]:
        """Create mock entry."""
        if "fail_create" in clean_content:
            return FlextResult.fail("Creation failed")
        return FlextResult.ok(
            MockEntry(
                entry_type=entry_type,
                clean_content=clean_content,
                original_content=original_content,
                identifier=identifier,
            ),
        )


class MockFileWriter(BaseFileWriter):
    """Mock file writer for testing."""

    def __init__(self, *, fail_write: bool = False) -> None:
        """Initialize mock writer."""
        self.fail_write = fail_write
        self.written_entries: list[object] = []

    def write_header(self, output_file: object) -> None:  # noqa: ARG002
        """Write mock header."""
        if self.fail_write:
            msg = "Write failed"
            raise ValueError(msg)

    def write_entry(self, output_file: object, entry: object) -> None:  # noqa: ARG002
        """Write mock entry."""
        self.written_entries.append(entry)


class TestEntryType:
    """Test entry type enumeration."""

    def test_entry_type_enum(self) -> None:
        """Test entry type enum exists."""
        from enum import EnumMeta  # noqa: PLC0415

        assert isinstance(EntryType, EnumMeta)
        assert issubclass(EntryType, Enum)


class TestBaseEntry:
    """Test base entry value object."""

    def test_base_entry_creation(self) -> None:
        """Test base entry can be created through concrete implementation."""
        entry = ConcreteBaseEntry(
            entry_type="test",
            clean_content="clean",
            original_content="original",
            identifier="id_123",
        )
        assert entry.entry_type == "test"
        assert entry.clean_content == "clean"
        assert entry.original_content == "original"
        assert entry.identifier == "id_123"

    def test_base_entry_immutability(self) -> None:
        """Test base entry is immutable."""
        entry = ConcreteBaseEntry(
            entry_type="test",
            clean_content="clean",
            original_content="original",
            identifier="id_123",
        )
        # Should not be able to modify (frozen Pydantic model)
        from pydantic_core import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError, match="Instance is frozen"):
            entry.entry_type = "new_type"


class TestConfigAttributeValidator:
    """Test configuration attribute validator."""

    def test_has_attribute_true(self) -> None:
        """Test has_attribute returns True for existing attribute."""

        class TestConfig:
            def __init__(self) -> None:
                self.test_attr = "value"

        config = TestConfig()
        assert ConfigAttributeValidator.has_attribute(config, "test_attr")

    def test_has_attribute_false(self) -> None:
        """Test has_attribute returns False for missing attribute."""

        class EmptyConfig:
            pass

        config = EmptyConfig()
        assert not ConfigAttributeValidator.has_attribute(config, "missing_attr")

    def test_has_rules_config_true(self) -> None:
        """Test has_rules_config returns True when attribute exists."""

        class ConfigWithRules:
            def __init__(self) -> None:
                self.rules_config = {}

        config = ConfigWithRules()
        assert ConfigAttributeValidator.has_rules_config(config)

    def test_has_rules_config_false(self) -> None:
        """Test has_rules_config returns False when attribute missing."""

        class ConfigNoRules:
            pass

        config = ConfigNoRules()
        assert not ConfigAttributeValidator.has_rules_config(config)

    def test_validate_required_attributes_success(self) -> None:
        """Test validate_required_attributes succeeds with all attributes."""

        class ConfigWithAttrs:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = "value2"

        config = ConfigWithAttrs()
        result = ConfigAttributeValidator.validate_required_attributes(
            config,
            ["attr1", "attr2"],
        )
        assert result.success
        assert result.data is True

    def test_validate_required_attributes_failure(self) -> None:
        """Test validate_required_attributes fails with missing attributes."""

        # Create a simple object with only attr1
        class SimpleConfig:
            def __init__(self) -> None:
                self.attr1 = "value1"

        config = SimpleConfig()

        result = ConfigAttributeValidator.validate_required_attributes(
            config,
            ["attr1", "attr2", "attr3"],
        )
        assert not result.success
        assert "Missing required attributes" in (result.error or "")
        assert "attr2" in (result.error or "")
        assert "attr3" in (result.error or "")


class TestBaseConfigManager:
    """Test base configuration manager."""

    def test_init(self) -> None:
        """Test config manager initialization."""

        class TestConfig:
            pass

        config = TestConfig()
        manager = BaseConfigManager(config)
        assert manager.config is config
        assert isinstance(manager.validator, ConfigAttributeValidator)

    def test_get_config_value_exists(self) -> None:
        """Test get_config_value returns existing value."""

        class ConfigWithValue:
            def __init__(self) -> None:
                self.test_key = "test_value"

        config = ConfigWithValue()
        manager = BaseConfigManager(config)

        value = manager.get_config_value("test_key")
        assert value == "test_value"

    def test_get_config_value_default(self) -> None:
        """Test get_config_value returns default for missing key."""

        class EmptyConfig:
            pass

        config = EmptyConfig()
        manager = BaseConfigManager(config)

        value = manager.get_config_value("missing_key", "default_value")
        assert value == "default_value"

    def test_validate_config_success(self) -> None:
        """Test validate_config succeeds with required attributes."""

        class ConfigWithAttrs:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = "value2"

        config = ConfigWithAttrs()
        manager = BaseConfigManager(config)

        result = manager.validate_config(["attr1", "attr2"])
        assert result.success
        assert result.data is True

    def test_validate_config_no_requirements(self) -> None:
        """Test validate_config succeeds with no requirements."""

        class SimpleConfig:
            pass

        config = SimpleConfig()
        manager = BaseConfigManager(config)

        result = manager.validate_config()
        assert result.success
        assert result.data is True

    def test_validate_config_failure(self) -> None:
        """Test validate_config fails with missing attributes."""

        # Create a simple object with only attr1
        class SimpleConfig:
            def __init__(self) -> None:
                self.attr1 = "value1"

        config = SimpleConfig()
        manager = BaseConfigManager(config)

        result = manager.validate_config(["attr1", "missing_attr"])
        assert not result.success


class TestBaseProcessor:
    """Test base processor functionality."""

    def test_init_without_validator(self) -> None:
        """Test processor initialization without validator."""
        processor = MockProcessor()
        assert processor.validator is None
        assert len(processor._extracted_entries) == 0

    def test_init_with_validator(self) -> None:
        """Test processor initialization with validator."""
        validator = MockEntryValidator()
        processor = MockProcessor(validator)
        assert processor.validator is validator

    def test_extract_entry_info_success(self) -> None:
        """Test successful entry extraction."""
        processor = MockProcessor()
        result = processor.extract_entry_info("test content 123", "test_type")

        assert result.success
        assert result.data is not None
        assert result.data.entry_type == "test_type"
        assert result.data.identifier == "id_123"
        assert len(processor._extracted_entries) == 1

    def test_extract_entry_info_with_prefix(self) -> None:
        """Test entry extraction with prefix."""
        processor = MockProcessor()
        result = processor.extract_entry_info(
            "prefix: test content 456",
            "test_type",
            "prefix",
        )

        assert result.success
        assert result.data is not None
        assert result.data.clean_content == "test content 456"

    def test_extract_entry_info_identifier_failure(self) -> None:
        """Test entry extraction with identifier failure."""
        processor = MockProcessor()
        result = processor.extract_entry_info("invalid content", "test_type")

        assert not result.success
        assert "Failed to extract identifier" in (result.error or "")

    def test_extract_entry_info_creation_failure(self) -> None:
        """Test entry extraction with creation failure."""
        processor = MockProcessor()
        result = processor.extract_entry_info("fail_create content 789", "test_type")

        assert not result.success

    def test_extract_entry_info_validator_not_whitelisted(self) -> None:
        """Test entry extraction with validator rejection."""
        validator = MockEntryValidator(is_whitelisted=False)
        processor = MockProcessor(validator)
        result = processor.extract_entry_info("test content 123", "test_type")

        assert not result.success
        assert "not whitelisted" in (result.error or "")

    def test_extract_entry_info_validator_not_valid(self) -> None:
        """Test entry extraction with invalid entry."""
        validator = MockEntryValidator(is_valid=False)
        processor = MockProcessor(validator)
        result = processor.extract_entry_info("test content 123", "test_type")

        assert not result.success
        assert "Entry validation failed" in (result.error or "")

    def test_process_content_lines_success(self) -> None:
        """Test processing multiple content lines successfully."""
        processor = MockProcessor()
        lines = [
            "test content 111",
            "test content 222",
            "test content 333",
        ]

        result = processor.process_content_lines(lines, "test_type")
        assert result.success
        assert len(result.data or []) == 3

    def test_process_content_lines_with_empty_lines(self) -> None:
        """Test processing with empty lines."""
        processor = MockProcessor()
        lines = [
            "test content 111",
            "",
            "   ",
            "test content 222",
        ]

        result = processor.process_content_lines(lines, "test_type")
        assert result.success
        assert len(result.data or []) == 2

    def test_process_content_lines_partial_failure(self) -> None:
        """Test processing with some failures."""
        processor = MockProcessor()
        lines = [
            "test content 111",
            "invalid content",
            "test content 333",
        ]

        result = processor.process_content_lines(lines, "test_type")
        assert result.success  # Partial success allowed
        assert len(result.data or []) == 2

    def test_process_content_lines_all_failure(self) -> None:
        """Test processing with all failures."""
        processor = MockProcessor()
        lines = [
            "invalid content",
            "invalid content",
        ]

        result = processor.process_content_lines(lines, "test_type")
        assert not result.success
        assert "All entries failed" in (result.error or "")

    def test_get_extracted_entries(self) -> None:
        """Test getting extracted entries."""
        processor = MockProcessor()
        processor.extract_entry_info("test content 123", "test_type")
        processor.extract_entry_info("test content 456", "test_type")

        entries = processor.get_extracted_entries()
        assert len(entries) == 2
        assert entries[0].identifier == "id_123"
        assert entries[1].identifier == "id_456"

    def test_clear_extracted_entries(self) -> None:
        """Test clearing extracted entries."""
        processor = MockProcessor()
        processor.extract_entry_info("test content 123", "test_type")

        assert len(processor._extracted_entries) == 1
        processor.clear_extracted_entries()
        assert len(processor._extracted_entries) == 0


class TestRegexProcessor:
    """Test regex-based processor."""

    def test_init(self) -> None:
        """Test regex processor initialization."""
        processor = ConcreteRegexProcessor(r"id:(\w+)")
        assert processor.identifier_pattern.pattern == r"id:(\w+)"

    def test_extract_identifier_success(self) -> None:
        """Test successful identifier extraction with regex."""
        processor = ConcreteRegexProcessor(r"id:(\w+)")
        result = processor._extract_identifier("content with id:test123 here")

        assert result.success
        assert result.data == "test123"

    def test_extract_identifier_no_match(self) -> None:
        """Test identifier extraction with no regex match."""
        processor = ConcreteRegexProcessor(r"id:(\w+)")
        result = processor._extract_identifier("content without identifier")

        assert not result.success
        assert "No identifier found" in (result.error or "")


class TestBaseSorter:
    """Test base sorter functionality."""

    def test_init_default_key_extractor(self) -> None:
        """Test sorter initialization with default key extractor."""
        sorter = BaseSorter[str]()
        assert sorter.key_extractor is not None

    def test_init_custom_key_extractor(self) -> None:
        """Test sorter initialization with custom key extractor."""

        def key_func(x: str) -> int:
            return len(x)

        sorter = BaseSorter[str](key_func)
        assert sorter.key_extractor is key_func

    def test_sort_entries_success(self) -> None:
        """Test successful entry sorting."""
        sorter = BaseSorter[str](len)
        entries = ["hello", "hi", "world"]

        sorted_entries = sorter.sort_entries(entries)
        assert sorted_entries == ["hi", "hello", "world"]

    def test_sort_entries_exception_handling(self) -> None:
        """Test sort exception handling."""

        def failing_key(x: object) -> object:  # noqa: ARG001
            msg = "Sort failed"
            raise ValueError(msg)

        sorter = BaseSorter[str](failing_key)
        entries = ["hello", "world"]

        # Should return unsorted entries on exception
        sorted_entries = sorter.sort_entries(entries)
        assert sorted_entries == entries


class TestBaseFileWriter:
    """Test base file writer functionality."""

    def test_write_entries_success(self) -> None:
        """Test successful entries writing."""
        from typing import cast  # noqa: PLC0415

        class TestOutputFile:
            def __init__(self) -> None:
                self.written_data: list[str] = []

            def write(self, data: str) -> None:
                self.written_data.append(data)

        writer = MockFileWriter()
        entries = cast("list[object]", ["entry1", "entry2", "entry3"])
        output_file = TestOutputFile()

        result = writer.write_entries(output_file, entries)
        assert result.success
        assert len(writer.written_entries) == 3

    def test_write_entries_failure(self) -> None:
        """Test entries writing with failure."""
        from typing import cast  # noqa: PLC0415

        class TestOutputFile:
            def __init__(self) -> None:
                self.written_data: list[str] = []

            def write(self, data: str) -> None:
                self.written_data.append(data)

        writer = MockFileWriter(fail_write=True)
        entries = cast("list[object]", ["entry1", "entry2"])
        output_file = TestOutputFile()

        result = writer.write_entries(output_file, entries)
        assert not result.success
        assert "Failed to write entries" in (result.error or "")


class TestProcessingPipeline:
    """Test processing pipeline functionality."""

    def test_init(self) -> None:
        """Test pipeline initialization."""
        pipeline = ProcessingPipeline[str, str]()
        assert len(pipeline.steps) == 0

    def test_add_step(self) -> None:
        """Test adding processing step."""
        pipeline = ProcessingPipeline[str, str]()

        def step_func(data: str) -> FlextResult[str]:
            return FlextResult.ok(data.upper())

        result_pipeline = pipeline.add_step(step_func)
        assert result_pipeline is pipeline
        assert len(pipeline.steps) == 1

    def test_process_success(self) -> None:
        """Test successful pipeline processing."""
        pipeline = ProcessingPipeline[str, str]()

        def step1(data: str) -> FlextResult[str]:
            return FlextResult.ok(data.upper())

        def step2(data: str) -> FlextResult[str]:
            return FlextResult.ok(f"processed_{data}")

        pipeline.add_step(step1).add_step(step2)
        result = pipeline.process("hello")

        assert result.success
        assert result.data == "processed_HELLO"

    def test_process_failure(self) -> None:
        """Test pipeline processing with failure."""
        pipeline = ProcessingPipeline[str, str]()

        def step1(data: str) -> FlextResult[str]:
            return FlextResult.ok(data.upper())

        def step2(_data: str) -> FlextResult[str]:
            return FlextResult.fail("Processing failed")

        pipeline.add_step(step1).add_step(step2)
        result = pipeline.process("hello")

        assert not result.success
        assert result.error == "Processing failed"

    def test_process_empty_pipeline(self) -> None:
        """Test processing with empty pipeline."""
        pipeline = ProcessingPipeline[str, str]()
        result = pipeline.process("hello")

        assert result.success
        assert result.data == "hello"


class TestEntryValidatorProtocol:
    """Test entry validator protocol."""

    def test_protocol_implementation(self) -> None:
        """Test that our mock implements the protocol correctly."""
        validator = MockEntryValidator()

        # Should be usable as EntryValidator
        assert callable(validator.is_valid)
        assert callable(validator.is_whitelisted)

        # Test actual calls
        mock_entry = MockEntry(
            entry_type="test",
            clean_content="clean",
            original_content="original",
            identifier="id_123",
        )

        assert validator.is_valid(mock_entry) is True
        assert validator.is_whitelisted("test_id") is True
