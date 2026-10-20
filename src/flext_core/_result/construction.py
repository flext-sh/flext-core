"""Construction operations for FlextResult.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Never, Self

from pydantic import BaseModel

from flext_core import c, p
from flext_core._result import FlextResultBase, FlextResultBehavior
from flext_core._utilities import FlextUtilitiesMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import t


class FlextResultConstruction[T](FlextResultBehavior[T], ABC):
    """Factory methods for the concrete result facade.

    Contracts are typed as ``p.Result``. Payload-changing operations delegate
    to generic construction hooks supplied by the concrete facade through MRO.
    """

    @staticmethod
    @abstractmethod
    def _create_success[V](value: V) -> p.Result[V]:
        """Construct a payload through the concrete facade's MRO owner."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _create_failure(
        error: str,
        error_code: str | None,
        error_data: t.JsonDict | None,
        exception: BaseException | None,
    ) -> p.Result[Never]:
        """Construct normalized failure state through the concrete facade."""
        raise NotImplementedError

    @staticmethod
    def require_error(source: p.FailureLike) -> str:
        """Extract error text from any failed Result.

        Empty failures (``fail(None)`` / ``fail("")``) are valid railway values;
        return ``""`` so combinators re-wrap instead of raising.

        Returns:
            The resulting ``str``.

        """
        error = source.error
        return error or ""

    @classmethod
    def ok[V](cls: type[Self], value: V) -> p.Result[V]:
        """Create a successful result carrying ``value``.

        Returns:
            The resulting ``p.Result[V]``.

        """
        cls.reject_banned_result_parameterization()
        return cls._create_success(value)

    @classmethod
    def from_failure(cls: type[Self], source: p.FailureLike) -> p.Result[T]:
        """Build a failure copying the error metadata of another result.

        Args:
            source: The failed result whose error state is copied.

        Returns:
            The resulting ``p.Result[T]``.

        Raises:
            ValueError: If ``source`` is a successful result.

        """
        if source.success:
            msg = c.ERR_RESULT_FAILURE_REQUIRED
            raise ValueError(msg)
        return cls.fail(
            cls.require_error(source),
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def _extract_exception_error_code(
        cls,
        exception: BaseException | None,
    ) -> str | None:
        if exception is None:
            return None
        error_code = getattr(exception, "error_code", None)
        return error_code if isinstance(error_code, str) and error_code else None

    @classmethod
    def _redacted_error_data_keys(
        cls,
        exception: BaseException | None,
    ) -> frozenset[str]:
        """Union of fleet-sensitive keys and the exception's excluded context keys.

        Returns:
            The resulting ``frozenset[str]``.

        """
        excluded: set[str] = set(c.SENSITIVE_ERROR_DATA_KEYS)
        if exception is not None:
            class_excluded = getattr(type(exception), "excluded_context_keys", None)
            if class_excluded:
                excluded.update(class_excluded)
        return frozenset(excluded)

    @classmethod
    def _extract_exception_error_data(
        cls,
        exception: BaseException | None,
    ) -> t.JsonDict | None:
        if exception is None:
            return None
        payload: t.JsonDict | None = None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, c.FIELD_ATTRIBUTES, None)
        if raw_attributes is not None:
            payload = cls.validate_error_data(raw_attributes)
        correlation_id = getattr(exception, "correlation_id", None)
        if isinstance(correlation_id, str) and correlation_id:
            if payload is None:
                payload = {}
            payload[c.ContextKey.CORRELATION_ID] = correlation_id
        return payload or None

    @classmethod
    def from_result[V](cls: type[Self], source: p.Result[V]) -> p.Result[V]:
        """Copy an abstract result into this concrete result family.

        Returns:
            The resulting ``p.Result[V]``.

        """
        cls.reject_banned_result_parameterization()
        if source.success:
            try:
                return cls._create_success(source.value)
            except ValueError as exc:
                return cls._failure(str(exc), exception=exc)
        return cls._copy_failure(source)

    @classmethod
    def _copy_failure(cls, source: p.FailureLike) -> p.Result[Never]:
        """Copy failure state without coupling its absent payload to a type.

        Returns:
            A failure carrying the original metadata.

        """
        return cls._failure(
            source.error or "",
            error_code=source.error_code,
            error_data=source.error_data,
            exception=source.exception,
        )

    @classmethod
    def create_from_callable[V](
        cls: type[Self],
        func: Callable[[], V | None],
        error_code: str | None = None,
    ) -> p.Result[V]:
        """Invoke a callable, turning ``None`` and exceptions into failures.

        Args:
            func: The callable producing the payload.
            error_code: Optional error code carried by the failure results.

        Returns:
            The resulting ``p.Result[V]``.

        """
        cls.reject_banned_result_parameterization()
        try:
            value = func()
            if value is None:
                return cls._failure("Callable returned None", error_code=error_code)
            return cls._create_success(value)
        except c.EXC_BROAD_RUNTIME as exc:
            return cls._failure(str(exc), error_code=error_code, exception=exc)

    @classmethod
    def _filter_sensitive_error_data(
        cls,
        payload: t.JsonDict | None,
        exception: BaseException | None = None,
    ) -> t.JsonDict | None:
        """Drop sensitive keys from any error_data mapping before storage.

        Returns:
            The resulting ``t.JsonDict | None``.

        """
        if payload is None:
            return None
        redacted_keys = cls._redacted_error_data_keys(exception)
        filtered = FlextUtilitiesMetadata.redact_metadata(
            payload,
            keys=tuple(redacted_keys),
        )
        return filtered or None

    @classmethod
    def fail(
        cls: type[Self],
        error: str | None,
        *,
        error_code: str | None = None,
        error_data: t.JsonMapping | p.ConfigModelInput | None = None,
        exception: BaseException | None = None,
    ) -> p.Result[T]:
        """Build a failed result, resolving codes and redacting error data.

        A ``None`` error normalizes to the empty string; an omitted
        ``error_code`` is extracted from the exception, and an omitted
        ``error_data`` is extracted from the exception's metadata with
        sensitive keys redacted and the correlation id preserved.

        Args:
            error: The human-readable error message, or ``None``.
            error_code: Optional machine-readable error code.
            error_data: Optional structured error payload to validate and filter.
            exception: Optional exception bound to the failure.

        Returns:
            The resulting ``p.Result[T]``.

        """
        cls.reject_banned_result_parameterization()
        return cls._failure(
            error, error_code=error_code, error_data=error_data, exception=exception
        )

    @classmethod
    def _failure(
        cls,
        error: str | None,
        *,
        error_code: str | None = None,
        error_data: t.JsonMapping | p.ConfigModelInput | None = None,
        exception: BaseException | None = None,
    ) -> p.Result[Never]:
        """Normalize failure state before the concrete generic factory consumes it.

        Returns:
            A result carrying normalized, redacted failure state.

        """
        error_msg = error if error is not None else ""
        resolved_error_code = error_code or cls._extract_exception_error_code(exception)
        # Why: redact caller-supplied AND auto-extracted error_data (security mro-8taj)
        resolved_error_data = cls._filter_sensitive_error_data(
            cls.validate_error_data(error_data)
            if error_data is not None
            else cls._extract_exception_error_data(exception),
            exception,
        )
        return cls._create_failure(
            error_msg,
            resolved_error_code,
            resolved_error_data,
            exception,
        )

    @classmethod
    def fail_op(
        cls: type[Self],
        operation: str,
        exc: Exception | str | None = None,
    ) -> p.Result[T]:
        """Build a failure with a standardized ``<operation> failed`` message.

        Args:
            operation: The operation name leading the error message.
            exc: Optional exception or message appended to the error.

        Returns:
            The resulting ``p.Result[T]``.

        """
        if isinstance(exc, Exception):
            return cls.fail(f"{operation} failed: {exc}", exception=exc)
        error_msg = (
            f"{operation} failed" if exc is None else f"{operation} failed: {exc}"
        )
        return cls.fail(error_msg)

    @classmethod
    def from_validation[ModelT: BaseModel](
        cls: type[Self],
        data: p.ModelInput,
        model: t.ModelClass[ModelT],
    ) -> p.Result[ModelT]:
        """Validate data against a model, turning validation errors into failures.

        Args:
            data: The input passed to the model's validation.
            model: The model class validating the data.

        Returns:
            The resulting ``p.Result[ModelT]``.

        """
        try:
            validated: ModelT = model.model_validate(data)
            return cls._create_success(validated)
        except c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return cls._failure(str(exc), exception=exc)

    @staticmethod
    def successful_result[ValueT](obj: ValueT) -> bool:
        """Check whether an object is a successful result instance.

        Returns:
            The resulting ``bool``.

        """
        return isinstance(obj, FlextResultBase) and obj.success

    @staticmethod
    def failed_result[ValueT](obj: ValueT) -> bool:
        """Check whether an object is a failed result instance.

        Returns:
            The resulting ``bool``.

        """
        return isinstance(obj, FlextResultBase) and not obj.success


__all__: list[str] = ["FlextResultConstruction"]
