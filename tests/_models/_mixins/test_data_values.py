"""Static value test data helpers."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m


class TestsFlextModelsTestDataValuesMixin:
    """Static value test data helpers."""

    class ErrorData(m.BaseModel):
        """Test error codes and messages."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        error_code: Annotated[
            str,
            m.Field(
                description="Default test error code",
            ),
        ] = "TEST_ERROR_001"
        validation_error: Annotated[
            str,
            m.Field(
                description="Default validation error message",
            ),
        ] = "test_error"
        operation_error: Annotated[
            str,
            m.Field(
                description="Default operation error message",
            ),
        ] = "Op failed"
        settings_error: Annotated[
            str,
            m.Field(
                description="Default configuration error message",
            ),
        ] = "Settings failed"
        timeout_error: Annotated[
            str,
            m.Field(
                description="Default timeout error message",
            ),
        ] = "Operation timeout"

    class Data(m.BaseModel):
        """Test field names and data values."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

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
        password: Annotated[str, m.Field(description="Default test password")] = (
            "test_pass"
        )
        string_value: Annotated[
            str,
            m.Field(
                description="Default test string value",
            ),
        ] = "test_value"
        input_data: Annotated[str, m.Field(description="Default test input data")] = (
            "test_input"
        )
        request_data: Annotated[
            str,
            m.Field(
                description="Default test request data",
            ),
        ] = "test_request"
        result_data: Annotated[
            str,
            m.Field(
                description="Default test result data",
            ),
        ] = "test_result"
        message: Annotated[str, m.Field(description="Default test message")] = (
            "test_message"
        )

    class PatternData(m.BaseModel):
        """Test patterns and formats."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        slug_input: Annotated[
            str,
            m.Field(
                description="Input value for slug conversion tests",
            ),
        ] = "Test_String"
        slug_expected: Annotated[
            str,
            m.Field(
                description="Expected slug conversion output",
            ),
        ] = "test_string"
        uuid_format: Annotated[
            str,
            m.Field(
                description="Sample UUID format for tests",
            ),
        ] = "550e8400-e29b-41d4-a716-446655440000"

    class NumericValues(m.BaseModel):
        """Test port and numeric values."""

        model_config: ClassVar[p.ConfigDict] = m.ConfigDict(frozen=True)

        port: Annotated[int, m.Field(description="Default test port")] = 8080
        timeout: Annotated[int, m.Field(description="Default timeout in seconds")] = 30
        retry_count: Annotated[int, m.Field(description="Default retry count")] = 3
        batch_size: Annotated[int, m.Field(description="Default test batch size")] = 100

    # --- from test_container.py ---


__all__: list[str] = ["TestsFlextModelsTestDataValuesMixin"]
