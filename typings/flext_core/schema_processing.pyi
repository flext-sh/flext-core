from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Protocol, TypeVar

from _typeshed import Incomplete

from flext_core.result import FlextResult
from flext_core.typings import EntryT
from flext_core.value_objects import FlextValueObject

__all__ = [
    "BaseEntry",
    "BaseFileWriter",
    "BaseProcessor",
    "EntryType",
    "EntryValidator",
    "FlextBaseEntry",
    "FlextBaseProcessor",
    "FlextEntryType",
    "FlextEntryValidator",
    "FlextProcessingPipeline",
    "FlextRegexProcessor",
    "ProcessingPipeline",
]

class FlextEntryType(Enum): ...

class FlextBaseEntry(FlextValueObject, ABC):
    entry_type: str
    clean_content: str
    original_content: str
    identifier: str

class FlextEntryValidator(Protocol):
    def is_valid(self, entry: EntryT) -> bool: ...
    def is_whitelisted(self, identifier: str) -> bool: ...

_EntryTypeVar = TypeVar("_EntryTypeVar")

class FlextBaseProcessor[EntryTypeVar](ABC):
    validator: Incomplete
    def __init__(self, validator: FlextEntryValidator | None = None) -> None: ...
    def extract_entry_info(
        self, content: str, entry_type: str, prefix: str = ""
    ) -> FlextResult[_EntryTypeVar]: ...
    def process_content_lines(
        self, lines: list[str], entry_type: str, prefix: str = ""
    ) -> FlextResult[list[_EntryTypeVar]]: ...
    def get_extracted_entries(self) -> list[_EntryTypeVar]: ...
    def clear_extracted_entries(self) -> None: ...

class FlextRegexProcessor(FlextBaseProcessor[EntryT], ABC):
    identifier_pattern: Incomplete
    def __init__(
        self, identifier_pattern: str, validator: FlextEntryValidator | None = None
    ) -> None: ...

class FlextConfigAttributeValidator:
    @staticmethod
    def has_attribute(config: object, attribute: str) -> bool: ...
    @staticmethod
    def has_rules_config(config: object) -> bool: ...
    @staticmethod
    def validate_required_attributes(
        config: object, required: list[str]
    ) -> FlextResult[bool]: ...

class FlextBaseConfigManager:
    config: Incomplete
    validator: Incomplete
    def __init__(self, config: object) -> None: ...
    def get_config_value(self, key: str, default: object = None) -> object: ...
    def validate_config(
        self, required_attrs: list[str] | None = None
    ) -> FlextResult[bool]: ...

class FlextBaseSorter[T]:
    key_extractor: Incomplete
    def __init__(self, key_extractor: Callable[[T], object] | None = None) -> None: ...
    def sort_entries(self, entries: list[T]) -> list[T]: ...

class FlextBaseFileWriter(ABC):
    @abstractmethod
    def write_header(self, output_file: object) -> None: ...
    @abstractmethod
    def write_entry(self, output_file: object, entry: object) -> None: ...
    def write_entries(
        self, output_file: object, entries: list[object]
    ) -> FlextResult[None]: ...

class FlextProcessingPipeline[T, U]:
    steps: list[Callable[[object], FlextResult[object]]]
    def __init__(self) -> None: ...
    def add_step(
        self, step: Callable[[T], FlextResult[U]]
    ) -> FlextProcessingPipeline[T, U]: ...
    def process(self, input_data: T) -> FlextResult[U]: ...

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
