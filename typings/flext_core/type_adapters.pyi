from datetime import datetime
from typing import TypeVar

from _typeshed import Incomplete
from pydantic import TypeAdapter

from flext_core.result import FlextResult

__all__ = [
    "ConfigDict_Type",
    "ConnectionString",
    "EmailAddress",
    "EntityId",
    "ErrorCode",
    "ErrorMessage",
    "EventList",
    "Host",
    "Metadata",
    "MigrationHelpers",
    "Percentage",
    "Port",
    "SchemaHelpers",
    "SerializationHelpers",
    "ServiceName",
    "Timestamp",
    "TypeAdapterExamples",
    "TypeAdapterFactory",
    "ValidationAdapters",
    "Version",
]

_T = TypeVar("_T")
EntityId = str
Version = int
Timestamp = datetime
Host = str
Port = int
EmailAddress = str
ServiceName = str
ErrorCode = str
ErrorMessage = str
type Metadata = dict[str, object]
type EventList = list[dict[str, object]]
type ConfigDict_Type = dict[str, object]
ConnectionString = str
Percentage = float

class TypeAdapterFactory:
    @staticmethod
    def create_adapter(type_: type[_T]) -> TypeAdapter[_T]: ...
    @staticmethod
    def create_list_adapter(item_type: type[_T]) -> TypeAdapter[list[_T]]: ...
    @staticmethod
    def create_dict_adapter(value_type: type[_T]) -> TypeAdapter[dict[str, _T]]: ...

class ValidationAdapters:
    entity_id_adapter: Incomplete
    version_adapter: Incomplete
    timestamp_adapter: Incomplete
    host_adapter: Incomplete
    port_adapter: Incomplete
    email_adapter: Incomplete
    service_name_adapter: Incomplete
    error_code_adapter: Incomplete
    error_message_adapter: Incomplete
    metadata_adapter: Incomplete
    event_list_adapter: Incomplete
    config_dict_adapter: Incomplete
    connection_string_adapter: Incomplete
    percentage_adapter: Incomplete
    @classmethod
    def validate_entity_id(cls, value: object) -> FlextResult[str]: ...
    @classmethod
    def validate_version(cls, value: object) -> FlextResult[int]: ...
    @classmethod
    def validate_email(cls, value: object) -> FlextResult[str]: ...
    @classmethod
    def validate_service_name(cls, value: object) -> FlextResult[str]: ...
    @classmethod
    def validate_host_port(
        cls, host: object, port: object
    ) -> FlextResult[tuple[str, int]]: ...
    @classmethod
    def validate_percentage(cls, value: object) -> FlextResult[float]: ...

class SerializationHelpers:
    @staticmethod
    def to_json(adapter: TypeAdapter[_T], value: _T) -> FlextResult[str]: ...
    @staticmethod
    def from_json(adapter: TypeAdapter[_T], json_str: str) -> FlextResult[_T]: ...
    @staticmethod
    def to_dict(
        adapter: TypeAdapter[_T], value: _T
    ) -> FlextResult[dict[str, object]]: ...
    @staticmethod
    def from_dict(
        adapter: TypeAdapter[_T], data: dict[str, object]
    ) -> FlextResult[_T]: ...

class SchemaHelpers:
    @staticmethod
    def generate_schema(
        adapter: TypeAdapter[_T],
    ) -> FlextResult[dict[str, object]]: ...
    @staticmethod
    def generate_multiple_schemas(
        adapters: dict[str, TypeAdapter[object]],
    ) -> FlextResult[dict[str, dict[str, object]]]: ...

class TypeAdapterExamples:
    @staticmethod
    def user_validation_example() -> None: ...
    @staticmethod
    def configuration_validation_example() -> None: ...

class MigrationHelpers:
    @staticmethod
    def convert_basemodel_to_dataclass(model_class: type) -> str: ...
    @staticmethod
    def create_adapter_for_legacy_model(model_class: type[_T]) -> TypeAdapter[_T]: ...
