from collections.abc import Iterator
from datetime import datetime

from pydantic import RootModel

from flext_core.result import FlextResult

__all__ = [
    "EmailAddress",
    "EntityId",
    "ErrorCode",
    "ErrorMessage",
    "FlextConnectionString",
    "FlextEmailAddress",
    "FlextEntityId",
    "FlextErrorCode",
    "FlextErrorMessage",
    "FlextEventList",
    "FlextHost",
    "FlextMetadata",
    "FlextPercentage",
    "FlextPort",
    "FlextServiceName",
    "FlextTimestamp",
    "FlextVersion",
    "Host",
    "Metadata",
    "Port",
    "ServiceName",
    "Timestamp",
    "Version",
    "create_email",
    "create_entity_id",
    "create_host_port",
    "create_service_name",
    "create_version",
    "from_legacy_dict",
    "to_legacy_dict",
]

class FlextEntityId(RootModel[str]):
    root: str
    @classmethod
    def validate_id(cls, v: str) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class FlextVersion(RootModel[int]):
    root: int
    def __int__(self) -> int: ...
    def __add__(self, other: object) -> FlextVersion: ...
    def __sub__(self, other: object) -> FlextVersion: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def increment(self) -> FlextVersion: ...

class FlextTimestamp(RootModel[datetime]):
    root: datetime
    def __lt__(self, other: FlextTimestamp) -> bool: ...
    def __le__(self, other: FlextTimestamp) -> bool: ...
    def __gt__(self, other: FlextTimestamp) -> bool: ...
    def __ge__(self, other: FlextTimestamp) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @classmethod
    def now(cls) -> FlextTimestamp: ...

class FlextMetadata(RootModel[dict[str, object]]):
    root: dict[str, object]
    def get(self, key: str, default: object = None) -> object: ...
    def set(self, key: str, value: object) -> FlextMetadata: ...

class FlextEventList(RootModel[list[dict[str, object]]]):
    root: list[dict[str, object]]
    def add_event(self, event_type: str, data: dict[str, object]) -> FlextEventList: ...
    def clear(self) -> tuple[FlextEventList, list[dict[str, object]]]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...
    def __iter__(self) -> Iterator[dict[str, object]]: ...

class FlextHost(RootModel[str]):
    root: str

class FlextPort(RootModel[int]):
    root: int
    def __int__(self) -> int: ...

class FlextConnectionString(RootModel[str]):
    root: str

class FlextEmailAddress(RootModel[str]):
    root: str
    @property
    def domain(self) -> str: ...

class FlextServiceName(RootModel[str]):
    root: str

class FlextPercentage(RootModel[float]):
    root: float
    def __float__(self) -> float: ...
    def as_decimal(self) -> float: ...

class FlextErrorCode(RootModel[str]):
    root: str

class FlextErrorMessage(RootModel[str]):
    root: str

def create_entity_id(value: str) -> FlextResult[FlextEntityId]: ...
def create_version(value: int) -> FlextResult[FlextVersion]: ...
def create_email(value: str) -> FlextResult[FlextEmailAddress]: ...
def create_service_name(value: str) -> FlextResult[FlextServiceName]: ...
def create_host_port(
    host: str, port: int
) -> FlextResult[tuple[FlextHost, FlextPort]]: ...
def from_legacy_dict(data: dict[str, object]) -> FlextMetadata: ...
def to_legacy_dict(metadata: FlextMetadata) -> dict[str, object]: ...

type EntityId = FlextEntityId
type Version = FlextVersion
type Timestamp = FlextTimestamp
type Metadata = FlextMetadata
type Host = FlextHost
type Port = FlextPort
type EmailAddress = FlextEmailAddress
type ServiceName = FlextServiceName
type ErrorCode = FlextErrorCode
type ErrorMessage = FlextErrorMessage
