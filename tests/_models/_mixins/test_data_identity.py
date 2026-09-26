"""Static identity test data helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m


class TestsFlextModelsTestDataIdentityMixin:
    """Static identity test data helpers."""

    class Names(m.BaseModel):
        """Test module and component names."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        module_name: Annotated[str, m.Field(description="Default test module name")] = (
            "test_module"
        )
        handler_name: Annotated[
            str, m.Field(description="Default test handler name")
        ] = "test_handler"
        chain_name: Annotated[str, m.Field(description="Default test chain name")] = (
            "test_chain"
        )
        command_type: Annotated[
            str, m.Field(description="Default test command type")
        ] = "test_command"
        query_type: Annotated[str, m.Field(description="Default test query type")] = (
            "test_query"
        )
        logger_name: Annotated[str, m.Field(description="Default test logger name")] = (
            "test_logger"
        )
        app_name: Annotated[
            str, m.Field(description="Default test application name")
        ] = "test-app"
        validation_app: Annotated[
            str, m.Field(description="Default validation test application name")
        ] = "validation-test"
        source_service: Annotated[
            str, m.Field(description="Default source service name")
        ] = "test_service"


__all__: list[str] = ["TestsFlextModelsTestDataIdentityMixin"]
