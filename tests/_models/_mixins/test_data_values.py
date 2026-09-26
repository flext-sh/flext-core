"""Static value test data helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m


class TestsFlextModelsTestDataValuesMixin:
    """Static value test data helpers."""

    class Data(m.BaseModel):
        """Test field names and data values."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        field_name: Annotated[str, m.Field(description="Default test field name")] = (
            "test_field"
        )
        config_key: Annotated[str, m.Field(description="Default test settings key")] = (
            "test_key"
        )
        username: Annotated[str, m.Field(description="Default test username")] = (
            "test_user"
        )
        email: Annotated[str, m.Field(description="Default test email")] = (
            "test@example.com"
        )
        string_value: Annotated[
            str, m.Field(description="Default test string value")
        ] = "test_value"
        input_data: Annotated[str, m.Field(description="Default test input data")] = (
            "test_input"
        )
        request_data: Annotated[
            str, m.Field(description="Default test request data")
        ] = "test_request"
        result_data: Annotated[str, m.Field(description="Default test result data")] = (
            "test_result"
        )
        message: Annotated[str, m.Field(description="Default test message")] = (
            "test_message"
        )

    # --- from test_container.py ---


__all__: list[str] = ["TestsFlextModelsTestDataValuesMixin"]
