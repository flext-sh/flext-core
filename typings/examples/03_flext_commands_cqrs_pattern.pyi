from typing import Protocol

from _typeshed import Incomplete

from flext_core import (
    FlextCommands,
    FlextResult,
    TAnyObject,
    TEntityId as TEntityId,
    TErrorMessage as TErrorMessage,
    TUserData as TUserData,
)

from .shared_domain import (
    SharedDomainFactory as SharedDomainFactory,
    log_domain_operation as log_domain_operation,
)

MIN_USER_AGE: int
MAX_USER_AGE: int
MIN_DELETION_REASON_LENGTH: int

class QueryHandlerProtocol(Protocol):
    def handle(self, query: object) -> FlextResult[object]: ...

class DomainEvent:
    event_id: TEntityId
    event_type: Incomplete
    data: Incomplete
    timestamp: Incomplete
    correlation_id: Incomplete
    def __init__(self, event_type: str, data: TAnyObject) -> None: ...
    def to_dict(self) -> dict[str, object]: ...

class EventStore:
    events: list[DomainEvent]
    def __init__(self) -> None: ...
    def append_event(self, event: DomainEvent) -> FlextResult[TEntityId]: ...
    def get_events_by_correlation(self, correlation_id: str) -> list[DomainEvent]: ...
    def get_all_events(self) -> list[DomainEvent]: ...

event_store: Incomplete

class BaseCommandHandler:
    handler_id: TEntityId
    handler_type: Incomplete
    def __init__(self, handler_type: str) -> None: ...
    def create_query_projection(self, shared_user: object) -> dict[str, object]: ...
    def store_domain_event(
        self, event_type: str, data: TAnyObject
    ) -> FlextResult[TEntityId]: ...

class DemonstrationFlowHelper:
    @staticmethod
    def print_section_header(example_num: int, title: str) -> None: ...
    @staticmethod
    def handle_result_with_state_update(
        result: FlextResult[TAnyObject],
        success_message: str,
        state_dict: dict[TEntityId, TUserData],
        user_id: TEntityId | None = None,
    ) -> FlextResult[TAnyObject]: ...

class CreateUserCommand(FlextCommands.Command):
    name: str
    email: str
    age: int
    def validate_command(self) -> FlextResult[None]: ...

class UpdateUserCommand(FlextCommands.Command):
    target_user_id: TEntityId
    name: str | None
    email: str | None
    def validate_command(self) -> FlextResult[None]: ...

class DeleteUserCommand(FlextCommands.Command):
    target_user_id: TEntityId
    reason: str
    def validate_command(self) -> FlextResult[None]: ...

class GetUserQuery(FlextCommands.Query):
    target_user_id: TEntityId

class ListUsersQuery(FlextCommands.Query):
    active_only: bool
    min_age: int | None
    max_age: int | None

class GetUserEventsQuery(FlextCommands.Query):
    correlation_id: str

class CreateUserCommandHandler(
    BaseCommandHandler, FlextCommands.Handler[CreateUserCommand, TAnyObject]
):
    users_db: Incomplete
    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None: ...
    def handle(self, command: CreateUserCommand) -> FlextResult[TAnyObject]: ...

class UpdateUserCommandHandler(
    BaseCommandHandler, FlextCommands.Handler[UpdateUserCommand, TAnyObject]
):
    users_db: Incomplete
    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None: ...
    def handle(self, command: UpdateUserCommand) -> FlextResult[TAnyObject]: ...

class DeleteUserCommandHandler(
    BaseCommandHandler, FlextCommands.Handler[DeleteUserCommand, TAnyObject]
):
    users_db: Incomplete
    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None: ...
    def handle(self, command: DeleteUserCommand) -> FlextResult[TAnyObject]: ...

class GetUserQueryHandler(FlextCommands.QueryHandler[GetUserQuery, TAnyObject]):
    users_db: Incomplete
    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None: ...
    def handle(self, query: GetUserQuery) -> FlextResult[TAnyObject]: ...

class ListUsersQueryHandler(
    FlextCommands.QueryHandler[ListUsersQuery, list[TAnyObject]]
):
    users_db: Incomplete
    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None: ...
    def handle(self, query: ListUsersQuery) -> FlextResult[list[TAnyObject]]: ...

class GetUserEventsQueryHandler(
    FlextCommands.QueryHandler[GetUserEventsQuery, list[TAnyObject]]
):
    def handle(self, query: GetUserEventsQuery) -> FlextResult[list[TAnyObject]]: ...

def setup_command_bus() -> FlextResult[FlextCommands.Bus]: ...
def setup_query_handlers(users_db: dict[TEntityId, TUserData]) -> dict[str, object]: ...

class UserManagementApplicationService:
    command_bus: Incomplete
    query_handlers: Incomplete
    service_id: TEntityId
    def __init__(
        self, command_bus: FlextCommands.Bus, query_handlers: dict[str, object]
    ) -> None: ...
    def create_user(
        self, name: str, email: str, age: int
    ) -> FlextResult[TAnyObject]: ...
    def update_user(
        self, user_id: TEntityId, name: str | None = None, email: str | None = None
    ) -> FlextResult[TAnyObject]: ...
    def delete_user(
        self, user_id: TEntityId, reason: str
    ) -> FlextResult[TAnyObject]: ...
    def get_user(self, user_id: TEntityId) -> FlextResult[TAnyObject]: ...
    def list_users(
        self,
        min_age: int | None = None,
        max_age: int | None = None,
        *,
        active_only: bool = True,
    ) -> FlextResult[list[TAnyObject]]: ...
    def get_user_events(self, correlation_id: str) -> FlextResult[list[TAnyObject]]: ...

class CQRSDemonstrator:
    users_db: dict[TEntityId, TUserData]
    app_service: UserManagementApplicationService | None
    created_user_id: TEntityId | None
    def __init__(self) -> None: ...
    def setup_cqrs_infrastructure(self) -> FlextResult[None]: ...
    def demonstrate_user_creation(self) -> FlextResult[TEntityId]: ...
    def demonstrate_user_update(self) -> FlextResult[None]: ...
    def demonstrate_user_queries(self) -> FlextResult[None]: ...
    def demonstrate_event_sourcing(self) -> FlextResult[None]: ...
    def demonstrate_user_deletion(self) -> FlextResult[None]: ...
    def demonstrate_validation_failure(self) -> FlextResult[None]: ...

def main() -> None: ...
