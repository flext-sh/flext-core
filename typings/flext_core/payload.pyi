from collections.abc import Callable, Mapping
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict

from flext_core.mixins import FlextLoggableMixin, FlextSerializableMixin
from flext_core.result import FlextResult
from flext_core.typings import TValue

__all__ = [
    "FLEXT_SERIALIZATION_VERSION",
    "SERIALIZATION_FORMAT_BINARY",
    "SERIALIZATION_FORMAT_JSON",
    "SERIALIZATION_FORMAT_JSON_COMPRESSED",
    "FlextEvent",
    "FlextMessage",
    "FlextPayload",
    "create_cross_service_event",
    "create_cross_service_message",
    "get_serialization_metrics",
    "validate_cross_service_protocol",
]

FLEXT_SERIALIZATION_VERSION: str
SERIALIZATION_FORMAT_JSON: str
SERIALIZATION_FORMAT_JSON_COMPRESSED: str
SERIALIZATION_FORMAT_BINARY: str
T = TypeVar("T")  # noqa: PYI001

class FlextPayload[T](BaseModel, FlextSerializableMixin, FlextLoggableMixin):
    model_config: ClassVar[ConfigDict] = ...
    data: T | None
    metadata: dict[str, object]
    @classmethod
    def create(cls, data: T, **metadata: object) -> FlextResult[FlextPayload[T]]: ...
    def with_metadata(self, **additional: TValue) -> FlextPayload[T]: ...
    def enrich_metadata(self, additional: dict[str, object]) -> FlextPayload[T]: ...
    @classmethod
    def create_from_dict(
        cls, data_dict: object
    ) -> FlextResult[FlextPayload[object]]: ...
    @classmethod
    def from_dict(
        cls, data_dict: dict[str, object] | Mapping[str, object] | object
    ) -> FlextResult[FlextPayload[object]]: ...
    def has_data(self) -> bool: ...
    def get_data(self) -> FlextResult[T]: ...
    def get_data_or_default(self, default: T) -> T: ...
    def transform_data(
        self, transformer: Callable[[T], object]
    ) -> FlextResult[FlextPayload[object]]: ...
    def get_metadata(
        self, key: str, default: object | None = None
    ) -> object | None: ...
    def has_metadata(self, key: str) -> bool: ...
    def serialize_data_for_json(self, value: T | None) -> object: ...
    def serialize_metadata_enhanced(
        self, value: dict[str, object]
    ) -> dict[str, object]: ...
    def serialize_payload_for_api(
        self,
        serializer: Callable[[FlextPayload[T]], dict[str, object] | object],
        info: object,
    ) -> dict[str, object] | object: ...
    def to_dict(self) -> dict[str, object]: ...
    def to_dict_basic(self) -> dict[str, object]: ...
    def to_cross_service_dict(
        self,
        *,
        includeType_info: bool = True,  # noqa: N803
        protocol_version: str = ...,  # noqa: N803
    ) -> dict[str, object]: ...
    @classmethod
    def from_cross_service_dict(
        cls, cross_service_dict: dict[str, object]
    ) -> FlextResult[FlextPayload[T]]: ...
    def to_json_string(
        self,
        *,
        compressed: bool = False,
        includeType_info: bool = True,  # noqa: N803
    ) -> FlextResult[str]: ...
    @classmethod
    def from_json_string(cls, json_str: str) -> FlextResult[FlextPayload[T]]: ...
    def get_serialization_size(self) -> dict[str, int | float]: ...
    def __getattr__(self, name: str) -> object: ...
    def __contains__(self, key: str) -> bool: ...
    def __hash__(self) -> int: ...
    def has(self, key: str) -> bool: ...
    def get(self, key: str, default: object | None = None) -> object | None: ...
    def keys(self) -> list[str]: ...
    def items(self) -> list[tuple[str, object]]: ...

class FlextMessage(FlextPayload[str]):
    @classmethod
    def create_message(
        cls, message: str, *, level: str = "info", source: str | None = None
    ) -> FlextResult[FlextMessage]: ...
    @property
    def level(self) -> str: ...
    @property
    def source(self) -> str | None: ...
    @property
    def correlation_id(self) -> str | None: ...
    @property
    def text(self) -> str | None: ...
    def to_cross_service_dict(
        self,
        *,
        includeType_info: bool = True,  # noqa: N803
        protocol_version: str = ...,  # noqa: N803
    ) -> dict[str, object]: ...
    @classmethod
    def from_cross_service_dict(
        cls, cross_service_dict: dict[str, object]
    ) -> FlextResult[FlextPayload[str]]: ...

class FlextEvent(FlextPayload[Mapping[str, object]]):
    @classmethod
    def create_event(
        cls,
        event_type: str,
        event_data: Mapping[str, object],
        *,
        aggregate_id: str | None = None,
        version: int | None = None,
    ) -> FlextResult[FlextEvent]: ...
    @property
    def event_type(self) -> str | None: ...
    @property
    def aggregate_id(self) -> str | None: ...
    @property
    def aggregateType(self) -> str | None: ...  # noqa: N802
    @property
    def version(self) -> int | None: ...
    @property
    def correlation_id(self) -> str | None: ...
    def to_cross_service_dict(
        self,
        *,
        includeType_info: bool = True,  # noqa: N803
        protocol_version: str = ...,  # noqa: N803
    ) -> dict[str, object]: ...
    @classmethod
    def from_cross_service_dict(
        cls, cross_service_dict: dict[str, object]
    ) -> FlextResult[FlextPayload[Mapping[str, object]]]: ...

def create_cross_service_event(
    event_type: str,
    event_data: dict[str, object],
    correlation_id: str | None = None,
    **kwargs: object,
) -> FlextResult[FlextEvent]: ...
def create_cross_service_message(
    messageText: str,  # noqa: N803
    correlation_id: str | None = None,
    **kwargs: object,  # noqa: N803
) -> FlextResult[FlextMessage]: ...
def get_serialization_metrics(payload: object | None = None) -> dict[str, object]: ...
def validate_cross_service_protocol(payload: object) -> FlextResult[None]: ...
