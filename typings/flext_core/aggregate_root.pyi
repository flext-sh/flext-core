from typing import ClassVar

from pydantic import ConfigDict

from flext_core.models import FlextEntity
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult
from flext_core.typings import TAnyDict

__all__ = ["FlextAggregateRoot"]

class FlextAggregateRoot(FlextEntity):
    model_config: ClassVar[ConfigDict]
    def __init__(
        self, entity_id: str | None = None, version: int = 1, **data: object
    ) -> None: ...
    def add_domain_event(
        self,
        event_type_or_dict: str | dict[str, object],
        event_data: dict[str, object] | None = None,
    ) -> FlextResult[None]: ...
    def add_typed_domain_event(
        self, event_type: str, event_data: TAnyDict
    ) -> object: ...
    def add_event_object(self, event: FlextEvent) -> None: ...
    def get_domain_events(self) -> list[FlextEvent]: ...
    def clear_domain_events(self) -> list[dict[str, object]]: ...
    def has_domain_events(self) -> bool: ...
