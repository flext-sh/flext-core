from _typeshed import Incomplete
from pydantic import BaseModel

from flext_core.mixins import FlextSerializableMixin
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextFieldId as FlextFieldId,
    FlextFieldName as FlextFieldName,
    FlextFieldTypeStr as FlextFieldTypeStr,
    FlextValidator as FlextValidator,
    TAnyDict,
    TFieldInfo,
    TFieldMetadata,
)

__all__ = [
    "FlextFieldCore",
    "FlextFieldCoreMetadata",
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldName",
    "FlextFieldRegistry",
    "FlextFieldTypeStr",
    "FlextFields",
    "FlextValidator",
    "flext_create_boolean_field",
    "flext_create_integer_field",
    "flext_create_string_field",
]

class FlextFieldCore(BaseModel, FlextSerializableMixin):
    model_config: Incomplete
    field_id: FlextFieldId
    field_name: FlextFieldName
    field_type: FlextFieldTypeStr
    required: bool
    default_value: str | int | float | bool | None
    min_value: int | float | None
    max_value: int | float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None
    allowed_values: list[object]
    description: str | None
    example: str | int | float | bool | None
    deprecated: bool
    sensitive: bool
    indexed: bool
    tags: list[str]
    validator: object
    def validate_field_value(self, value: object) -> tuple[bool, str | None]: ...
    def has_tag(self, tag: str) -> bool: ...
    def get_field_schema(self) -> TAnyDict: ...
    def get_field_metadata(self) -> TFieldMetadata: ...
    def validate_value(self, value: object) -> FlextResult[object]: ...
    def serialize_value(self, value: object) -> object: ...
    def deserialize_value(self, value: object) -> object: ...
    def get_default_value(self) -> str | int | float | bool | None: ...
    def is_required(self) -> bool: ...
    def is_deprecated(self) -> bool: ...
    def is_sensitive(self) -> bool: ...
    def get_field_info(self) -> TFieldInfo: ...
    @property
    def metadata(self) -> FlextFieldMetadata: ...

class FlextFieldMetadata(BaseModel):
    model_config: Incomplete
    field_id: FlextFieldId
    field_name: FlextFieldName
    field_type: FlextFieldTypeStr
    required: bool
    default_value: str | int | float | bool | None
    min_value: int | float | None
    max_value: int | float | None
    min_length: int | None
    max_length: int | None
    pattern: str | None
    allowed_values: list[object]
    description: str | None
    example: str | int | float | bool | None
    tags: list[str]
    deprecated: bool
    sensitive: bool
    indexed: bool
    internal: bool
    unique: bool
    custom_properties: TAnyDict
    @classmethod
    def from_field(cls, field: FlextFieldCore) -> FlextFieldMetadata: ...
    def to_dict(self) -> TAnyDict: ...
    @classmethod
    def from_dict(cls, data: TAnyDict) -> FlextFieldMetadata: ...

class FlextFieldRegistry(BaseModel):
    model_config: Incomplete
    fields_dict: dict[FlextFieldId, FlextFieldCore]
    field_names_dict: dict[FlextFieldName, FlextFieldId]
    def register_field(self, field: FlextFieldCore) -> FlextResult[None]: ...
    def get_field(self, field_id: FlextFieldId) -> FlextFieldCore | None: ...
    def get_all_fields(self) -> dict[FlextFieldId, FlextFieldCore]: ...
    def get_field_by_id(
        self, field_id: FlextFieldId
    ) -> FlextResult[FlextFieldCore]: ...
    def get_field_by_name(
        self, field_name: FlextFieldName
    ) -> FlextResult[FlextFieldCore]: ...
    def list_field_names(self) -> list[FlextFieldName]: ...
    def list_field_ids(self) -> list[FlextFieldId]: ...
    def get_field_count(self) -> int: ...
    def clear_registry(self) -> None: ...
    def remove_field(self, field_id: FlextFieldId) -> bool: ...
    def validate_all_fields(self, data: TAnyDict) -> FlextResult[None]: ...
    def get_fields_by_type(self, field_type: object) -> list[FlextFieldCore]: ...

class FlextFields:
    @classmethod
    def create_string_field(
        cls, field_id: FlextFieldId, field_name: FlextFieldName, **field_config: object
    ) -> FlextFieldCore: ...
    @classmethod
    def create_integer_field(
        cls, field_id: FlextFieldId, field_name: FlextFieldName, **field_config: object
    ) -> FlextFieldCore: ...
    @classmethod
    def create_boolean_field(
        cls, field_id: FlextFieldId, field_name: FlextFieldName, **field_config: object
    ) -> FlextFieldCore: ...
    @classmethod
    def register_field(cls, field: FlextFieldCore) -> FlextResult[None]: ...
    @classmethod
    def get_field_by_id(cls, field_id: FlextFieldId) -> FlextResult[FlextFieldCore]: ...
    @classmethod
    def get_field_by_name(
        cls, field_name: FlextFieldName
    ) -> FlextResult[FlextFieldCore]: ...
    @classmethod
    def list_field_names(cls) -> list[FlextFieldName]: ...
    @classmethod
    def get_field_count(cls) -> int: ...
    @classmethod
    def clear_registry(cls) -> None: ...
    @classmethod
    def string_field(cls, name: str, **kwargs: object) -> FlextFieldCore: ...
    @classmethod
    def integer_field(cls, name: str, **kwargs: object) -> FlextFieldCore: ...
    @classmethod
    def boolean_field(cls, name: str, **kwargs: object) -> FlextFieldCore: ...

FlextFieldCoreMetadata = FlextFieldMetadata

def flext_create_string_field(name: str, **kwargs: object) -> FlextFieldCore: ...
def flext_create_integer_field(name: str, **kwargs: object) -> FlextFieldCore: ...
def flext_create_boolean_field(name: str, **kwargs: object) -> FlextFieldCore: ...
