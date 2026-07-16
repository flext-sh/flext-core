"""Static identity test data helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

from flext_core import m
from tests.typings import p, t


class TestsFlextModelsTestDataIdentityMixin:
    """Static identity test data helpers."""

    class FalseSettings:
        app_name: str = "app"
        version: str = "1.0.0"
        enable_caching: bool = False
        timeout_seconds: float = 1.0
        dispatcher_auto_context: bool = False
        dispatcher_enable_logging: bool = False

        @classmethod
        def fetch_global(
            cls,
            *,
            overrides: t.ScalarMapping | None = None,
        ) -> Self:
            """Return a new instance for testing."""
            _ = overrides
            return cls()

        def model_copy(
            self,
            *,
            update: t.JsonMapping | None = None,
            deep: bool = False,
        ) -> Self:
            return self

        def model_dump(self) -> t.ScalarMapping:
            return dict[str, t.Scalar]()

    class Identifiers(m.BaseModel):
        """Test identifiers and IDs."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        user_id: Annotated[
            str,
            m.Field(
                description="Default test user identifier",
            ),
        ] = "test_user_123"
        session_id: Annotated[
            str,
            m.Field(
                description="Default test session identifier",
            ),
        ] = "test_session_123"
        service_name: Annotated[
            str,
            m.Field(
                description="Default test service name",
            ),
        ] = "test_service"
        operation_id: Annotated[
            str,
            m.Field(
                description="Default test operation identifier",
            ),
        ] = "test_operation"
        request_id: Annotated[
            str,
            m.Field(
                description="Default test request identifier",
            ),
        ] = "test-request-456"
        correlation_id: Annotated[
            str,
            m.Field(
                description="Default test correlation identifier",
            ),
        ] = "test-corr-123"

    class Names(m.BaseModel):
        """Test module and component names."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        module_name: Annotated[
            str,
            m.Field(
                description="Default test module name",
            ),
        ] = "test_module"
        handler_name: Annotated[
            str,
            m.Field(
                description="Default test handler name",
            ),
        ] = "test_handler"
        chain_name: Annotated[str, m.Field(description="Default test chain name")] = (
            "test_chain"
        )
        command_type: Annotated[
            str,
            m.Field(
                description="Default test command type",
            ),
        ] = "test_command"
        query_type: Annotated[str, m.Field(description="Default test query type")] = (
            "test_query"
        )
        logger_name: Annotated[
            str,
            m.Field(
                description="Default test logger name",
            ),
        ] = "test_logger"
        app_name: Annotated[
            str,
            m.Field(
                description="Default test application name",
            ),
        ] = "test-app"
        validation_app: Annotated[
            str,
            m.Field(
                description="Default validation test application name",
            ),
        ] = "validation-test"
        source_service: Annotated[
            str,
            m.Field(
                description="Default source service name",
            ),
        ] = "test_service"


__all__: list[str] = ["TestsFlextModelsTestDataIdentityMixin"]
