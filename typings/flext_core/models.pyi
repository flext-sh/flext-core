from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    ValidationInfo,
    computed_field,
)

from flext_core.result import FlextResult
from flext_core.root_models import (
    FlextEntityId,
    FlextEventList,
    FlextMetadata,
    FlextTimestamp,
    FlextVersion,
)

# Type aliases for JSON schema
type JsonSchemaValue = (
    str | int | float | bool | None | dict[str, object] | list[object]
)
type JsonSchemaFieldInfo = dict[str, JsonSchemaValue]
type JsonSchemaDefinition = dict[str, JsonSchemaValue]

__all__ = [
    "DomainEventDict",
    "FlextAuth",
    "FlextConnectionDict",
    "FlextData",
    "FlextDatabaseModel",
    "FlextEntity",
    "FlextEntityDict",
    "FlextEntityFactory",
    "FlextFactory",
    "FlextFieldValidationInfo",
    "FlextLegacyConfig",
    "FlextModel",
    "FlextModelDict",
    "FlextObs",
    "FlextOperationDict",
    "FlextOperationModel",
    "FlextOracleModel",
    "FlextServiceModel",
    "FlextSingerStreamModel",
    "FlextValue",
    "FlextValueObjectDict",
    "JsonSchemaDefinition",
    "JsonSchemaFieldInfo",
    "JsonSchemaValue",
    "create_database_model",
    "create_operation_model",
    "create_oracle_model",
    "create_service_model",
    "model_to_dict_safe",
    "validate_all_models",
]

class FlextModel(BaseModel):
    model_config: ClassVar[ConfigDict]
    @property
    @computed_field
    def model_type(self) -> str: ...
    @property
    @computed_field
    def model_namespace(self) -> str: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def validate_with_context(
        self, context: dict[str, object] | None = None
    ) -> FlextResult[None]: ...
    def to_dict(
        self, *, by_alias: bool = True, exclude_none: bool = True
    ) -> dict[str, object]: ...
    def to_typed_dict(self, *, by_alias: bool = False) -> dict[str, object]: ...
    def to_json_schema(
        self, *, by_alias: bool = True, mode: str = "validation"
    ) -> JsonSchemaDefinition: ...
    def to_json_schema_serialization(self) -> JsonSchemaDefinition: ...
    def to_openapi_schema(self) -> JsonSchemaDefinition: ...
    def to_camel_case_dict(self) -> dict[str, object]: ...
    def serialize_model_type(self, value: str) -> str: ...
    def serialize_model_namespace(self, value: str) -> str: ...
    def serialize_model_for_api(
        self,
        serializer: Callable[[FlextModel], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...

class FlextValue(FlextModel, ABC):
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]: ...

class FlextEntity(FlextModel, ABC):
    id: FlextEntityId
    version: FlextVersion
    created_at: FlextTimestamp
    updated_at: FlextTimestamp
    domain_events: FlextEventList
    metadata: FlextMetadata
    @classmethod
    def validate_entity_id(
        cls, v: str | FlextEntityId, info: ValidationInfo
    ) -> FlextEntityId: ...
    @classmethod
    def validate_version(cls, v: int | FlextVersion) -> FlextVersion: ...
    @property
    def entity_type(self) -> str: ...
    @property
    def entity_age_seconds(self) -> float: ...
    @property
    def is_new_entity(self) -> bool: ...
    @property
    def has_events(self) -> bool: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def increment_version(self) -> FlextResult[Self]: ...
    def with_version(self, new_version: int) -> Self: ...
    def add_domain_event(
        self,
        event_type_or_dict: str | dict[str, object],
        event_data: dict[str, object] | None = None,
    ) -> FlextResult[None]: ...
    def clear_events(self) -> list[object]: ...
    def validate_field(self, field_name: str, _value: object) -> FlextResult[None]: ...
    def validate_all_fields(self) -> FlextResult[None]: ...
    def copy_with(self, **changes: object) -> FlextResult[Self]: ...
    def validate_business_rules(self) -> FlextResult[None]: ...
    def serialize_timestamps(self, value: FlextTimestamp) -> str: ...
    def serialize_version(self, value: FlextVersion) -> dict[str, object]: ...
    def serialize_metadata(self, value: FlextMetadata) -> dict[str, object]: ...
    def serialize_entity_for_api(
        self,
        serializer: Callable[[FlextEntity], dict[str, object]],
        info: SerializationInfo,
    ) -> dict[str, object]: ...

class FlextFactory:
    @classmethod
    def register(
        cls, name: str, factory_or_class: type[FlextModel] | object
    ) -> None: ...
    @classmethod
    def create(cls, name: str, **kwargs: object) -> FlextResult[object]: ...
    @classmethod
    def create_entity_factory(
        cls, entity_class: type[FlextModel], defaults: dict[str, object] | None = None
    ) -> object: ...
    @staticmethod
    def create_model[T: FlextModel](
        model_class: type[T], **kwargs: object
    ) -> FlextResult[T]: ...

class FlextData: ...
class FlextAuth: ...
class FlextObs: ...

FlextEntityFactory = FlextFactory
type DomainEventDict = dict[str, object]
type FlextEntityDict = dict[str, object]
type FlextValueObjectDict = dict[str, object]
type FlextOperationDict = dict[str, object]
type FlextConnectionDict = dict[str, object]
type FlextModelDict = dict[str, JsonSchemaDefinition]
type FlextValidationContext = dict[str, object]
type FlextFieldValidationInfo = dict[str, JsonSchemaDefinition]
FlextDatabaseModel = FlextModel
FlextOracleModel = FlextModel
FlextLegacyConfig = FlextModel
FlextOperationModel = FlextModel
FlextServiceModel = FlextModel
FlextSingerStreamModel = FlextModel

def create_database_model(**kwargs: object) -> FlextResult[FlextModel]: ...
def create_oracle_model(**kwargs: object) -> FlextResult[FlextModel]: ...
def create_operation_model(**kwargs: object) -> FlextResult[FlextModel]: ...
def create_service_model(**kwargs: object) -> FlextResult[FlextModel]: ...
def validate_all_models(models: list[FlextModel]) -> FlextResult[None]: ...
def model_to_dict_safe(model: FlextModel) -> dict[str, object]: ...
