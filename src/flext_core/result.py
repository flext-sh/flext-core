"""Type-safe result type for operations."""

from __future__ import annotations

from types import TracebackType
from typing import Annotated, ClassVar, Self, TypeIs, overload, override

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field
from returns.result import Failure, Result, Success

from flext_core import (
    FlextModelsContainers as mc,
    FlextProtocolsLogging as pl,
    FlextRuntime,
    c,
    t,
)
from flext_core._result_parts.composition import FlextResultCompositionMixin
from flext_core._result_parts.transforms import FlextResultTransformsMixin
from flext_core._result_parts.unwrap import FlextResultUnwrapMixin


class FlextResult[T](
    BaseModel,
    FlextResultCompositionMixin[T],
    FlextResultTransformsMixin[T],
    FlextResultUnwrapMixin[T],
):
    """Type-safe result with monadic railway-oriented operations."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    result_success: Annotated[bool, Field(alias="success")] = True
    result_error: Annotated[str | None, Field(alias="error")] = None
    result_error_code: Annotated[str | None, Field(alias="error_code")] = None
    result_error_data: Annotated[mc.ConfigMap | None, Field(alias="error_data")] = None

    _payload: T | None = PrivateAttr(default=None)
    _exception: BaseException | None = PrivateAttr(default=None)
    _result: Result[T, str] | None = PrivateAttr(default=None)
    _result_logger: pl.Logger | None = PrivateAttr(default=None)

    @property
    @override
    def success(self) -> bool:
        """Success flag."""
        return self.result_success

    @property
    @override
    def error(self) -> str | None:
        """Error message."""
        return self.result_error

    @property
    @override
    def error_code(self) -> str | None:
        """Error code."""
        return self.result_error_code

    @property
    @override
    def error_data(self) -> t.JsonMapping | None:
        """Error metadata."""
        data = self.result_error_data
        if data is None:
            return None
        normalized_raw: t.JsonDict = {}
        for key, value in data.root.items():
            normalized_raw[key] = FlextRuntime.normalize_to_metadata(value)
        return normalized_raw

    @override
    def __repr__(self) -> str:
        if self.success:
            return f"r[T].ok({self.value!r})"
        return f"r[T].fail({self.error!r})"

    @override
    def __bool__(self) -> bool:
        return self.success

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        pass

    @overload
    def __or__(self, default: T) -> T: ...
    @overload
    def __or__[DefaultT](self, default: T | DefaultT) -> T | DefaultT: ...

    @override
    def __or__[DefaultT](self, default: T | DefaultT) -> T | DefaultT:
        return self.unwrap_or(default)

    @property
    @override
    def exception(self) -> BaseException | None:
        return self._exception

    @computed_field
    @property
    @override
    def failure(self) -> bool:
        return not self.success

    @property
    @override
    def value(self) -> T:
        if not self.success:
            msg = c.ERR_RESULT_CANNOT_ACCESS_VALUE.format(error=self.error)
            raise RuntimeError(msg)
        if self._payload is None:
            msg_0 = "Successful result must have a non-None payload"
            raise ValueError(msg_0)
        return self._payload

    @staticmethod
    @override
    def _validate_error_data(
        error_data: t.JsonMapping | t.ConfigModelInput | None,
    ) -> mc.ConfigMap | None:
        normalized_error_data = FlextRuntime.normalize_model_input_mapping(error_data)
        return (
            None
            if normalized_error_data is None
            else mc.ConfigMap.model_validate(normalized_error_data)
        )

    def __init__(
        self,
        error_code: str | None = None,
        error_data: t.JsonMapping | t.ConfigModelInput | None = None,
        *,
        value: T | None = None,
        error: str | None = None,
        success: bool = True,
    ) -> None:
        """Initialize a FlextResult with optional value, error, and metadata."""
        super().__init__(
            error=error,
            error_code=error_code,
            success=success,
            error_data=FlextResult._validate_error_data(error_data),
        )
        if success:
            self._payload = value

    @property
    def _returns_result(self) -> Result[T, str]:
        if self._result is None:
            if self.success:
                self._result = Success(self.value)
            else:
                self._result = Failure(self.error or "")
        return self._result

    @staticmethod
    def successful_result[V](obj: FlextResult[V] | V) -> TypeIs[FlextResult[V]]:
        """Type guard for successful result."""
        return isinstance(obj, FlextResult) and obj.success

    @staticmethod
    def failed_result[V](obj: FlextResult[V] | V) -> TypeIs[FlextResult[V]]:
        """Type guard for failed result."""
        return isinstance(obj, FlextResult) and obj.failure


r = FlextResult

__all__: list[str] = ["FlextResult", "r"]
