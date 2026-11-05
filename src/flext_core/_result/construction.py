"""Factory methods and construction combinators for FlextResult.

Supplies class-level constructor methods such as ``ok``, ``fail``, ``fail_op``,
``from_failure``, ``from_result``, and ``from_validation``. Enforces standardized
error code resolution, error metadata validation, and sensitive data redaction.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-027 (Facade Single-Owner Law):
      ``flext/docs/architecture/adr/027-facade-single-owner-law.md``
    - :mod:`flext_core._result.behavior`: Pythonic behavioral extensions.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Never, Self

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._result import FlextResultBase, FlextResultBehavior
from flext_core._utilities import FlextUtilitiesMetadata

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultConstruction[T](FlextResultBehavior[T], ABC):
    """Constructor and factory layer for railway result instances.

    Provides high-level factory methods for building success and failure results,
    converting exceptions into explicit failure values, and copying error context.

    Type Parameters:
        T: The payload type carried upon successful completion.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``

    """

    @staticmethod
    @abstractmethod
    def _create_success[V](value: V) -> flext_core.p.Result[V]:
        """Construct a validated success payload through the concrete facade.

        Args:
            value: The payload instance.

        Returns:
            p.Result[V]: Success result instance.

        """
        raise NotImplementedError(value)

    @staticmethod
    @abstractmethod
    def _create_failure(
        error: str,
        error_code: str | None,
        error_data: flext_core.t.JsonDict | None,
        exception: BaseException | None,
    ) -> flext_core.p.Result[Never]:
        """Construct normalized failure state through the concrete facade.

        Args:
            error: Descriptive failure message.
            error_code: Optional machine-readable error code.
            error_data: Optional structured metadata dictionary.
            exception: Optional causal exception.

        Returns:
            p.Result[Never]: Failure result instance.

        """
        msg = f"{error}|{error_code}|{error_data}|{exception}"
        raise NotImplementedError(msg)

    @staticmethod
    def require_error(source: flext_core.p.FailureLike) -> str:
        """Extract the error description string from a failed result.

        Args:
            source: A failure-like container exposing error attributes.

        Returns:
            str: The error message string, or empty string if None.

        """
        error = source.error
        return error or ""

    @classmethod
    def ok[V](cls: type[Self], value: V) -> flext_core.p.Result[V]:
        """Create a successful result carrying the provided payload.

        Args:
            value: The success payload instance.

        Returns:
            p.Result[V]: A validated success result wrapping ``value``.

        """
        cls.reject_banned_result_parameterization()
        return cls._create_success(value)

    @classmethod
    def from_failure(
        cls: type[Self], source: flext_core.p.FailureLike
    ) -> flext_core.p.Result[Never]:
        """Build a failure result copying error metadata from another failed result.

        Args:
            source: An existing failed result whose metadata is copied.

        Returns:
            p.Result[Never]: A new failure result matching the source error context.

        Raises:
            ValueError: If ``source`` indicates a successful result.

        """
        if source.success:
            msg = flext_core.c.ERR_RESULT_FAILURE_REQUIRED
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
        excluded: set[str] = set(flext_core.c.SENSITIVE_ERROR_DATA_KEYS)
        if exception is not None:
            class_excluded = getattr(type(exception), "excluded_context_keys", None)
            if class_excluded:
                excluded.update(class_excluded)
        return frozenset(excluded)

    @classmethod
    def _extract_exception_error_data(
        cls,
        exception: BaseException | None,
    ) -> flext_core.t.JsonDict | None:
        if exception is None:
            return None
        payload: flext_core.t.JsonDict | None = None
        metadata = getattr(exception, "metadata", None)
        raw_attributes = getattr(metadata, flext_core.c.FIELD_ATTRIBUTES, None)
        if raw_attributes is not None:
            payload = cls.validate_error_data(raw_attributes)
        correlation_id = getattr(
            exception, flext_core.c.ContextKey.CORRELATION_ID, None
        )
        if isinstance(correlation_id, str) and correlation_id:
            if payload is None:
                payload = {}
            payload[flext_core.c.ContextKey.CORRELATION_ID] = correlation_id
        return payload or None

    @classmethod
    def from_result[V](
        cls: type[Self], source: flext_core.p.Result[V]
    ) -> flext_core.p.Result[V]:
        """Convert or copy an abstract result into this concrete result family.

        Args:
            source: An abstract or concrete result protocol implementation.

        Returns:
            p.Result[V]: A concrete result instance mirroring the source outcome.

        """
        cls.reject_banned_result_parameterization()
        if source.success:
            try:
                return cls._create_success(source.value)
            except ValueError as exc:
                return cls._failure(str(exc), exception=exc)
        return cls._copy_failure(source)

    @classmethod
    def _copy_failure(
        cls, source: flext_core.p.FailureLike
    ) -> flext_core.p.Result[Never]:
        """Copy failure state without coupling its absent payload to a type.

        Args:
            source: A failure-like container exposing error attributes.

        Returns:
            p.Result[Never]: A failure carrying the original metadata.

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
    ) -> flext_core.p.Result[V]:
        """Invoke a callable, converting ``None`` returns and exceptions into failures.

        Args:
            func: The zero-argument callable producing the payload.
            error_code: Optional machine-readable error code for failure cases.

        Returns:
            p.Result[V]: A success result if ``func()`` returns a non-None value,
                otherwise a failure result.

        """
        cls.reject_banned_result_parameterization()
        try:
            value = func()
            if value is None:
                return cls._failure("Callable returned None", error_code=error_code)
            return cls._create_success(value)
        except flext_core.c.EXC_BROAD_RUNTIME as exc:
            return cls._failure(str(exc), error_code=error_code, exception=exc)

    @classmethod
    def _filter_sensitive_error_data(
        cls,
        payload: flext_core.t.JsonDict | None,
        exception: BaseException | None = None,
    ) -> flext_core.t.JsonDict | None:
        """Drop sensitive keys from any error_data mapping before storage.

        Args:
            payload: Raw error context dictionary.
            exception: Optional causal exception defining context exclusion keys.

        Returns:
            t.JsonDict | None: Filtered dictionary with sensitive fields redacted.

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
        error_data: flext_core.t.JsonMapping
        | flext_core.p.ConfigModelInput
        | None = None,
        exception: BaseException | None = None,
    ) -> flext_core.p.Result[Never]:
        """Build a failed result, resolving codes and redacting error data.

        A ``None`` error normalizes to the empty string; an omitted
        ``error_code`` is extracted from the exception, and an omitted
        ``error_data`` is extracted from the exception's metadata with
        sensitive keys redacted and the correlation id preserved.

        Args:
            error: The human-readable error message, or ``None``.
            error_code: Optional machine-readable error code.
            error_data: Optional structured error payload to validate and filter.
            exception: Optional causal exception bound to the failure.

        Returns:
            p.Result[Never]: A normalized failure result instance.

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
        error_data: flext_core.t.JsonMapping
        | flext_core.p.ConfigModelInput
        | None = None,
        exception: BaseException | None = None,
    ) -> flext_core.p.Result[Never]:
        """Normalize failure state before the concrete generic factory consumes it.

        Args:
            error: Raw error message string or None.
            error_code: Optional machine-readable error code.
            error_data: Optional structured metadata dictionary or model.
            exception: Optional causal exception.

        Returns:
            p.Result[Never]: Normalized failure result instance.

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
    ) -> flext_core.p.Result[Never]:
        """Build a failure with a standardized ``<operation> failed`` message.

        Args:
            operation: Name of the failed operation leading the error message.
            exc: Optional exception or message string appended to the error.

        Returns:
            p.Result[Never]: Standardized failure result instance.

        """
        if isinstance(exc, Exception):
            return cls.fail(f"{operation} failed: {exc}", exception=exc)
        error_msg = (
            f"{operation} failed" if exc is None else f"{operation} failed: {exc}"
        )
        return cls.fail(error_msg)

    @classmethod
    def from_validation[ModelT: flext_core.t.BaseModelType](
        cls: type[Self],
        data: flext_core.p.ModelInput,
        model: flext_core.t.ModelClass[ModelT],
    ) -> flext_core.p.Result[ModelT]:
        """Validate input data against a Pydantic model, capturing validation errors.

        Args:
            data: Raw input dictionary or mapping passed to model validation.
            model: The target Pydantic model class performing validation.

        Returns:
            p.Result[ModelT]: A success result carrying the validated model instance,
                or a failure result carrying validation error details.

        """
        try:
            validated: ModelT = model.model_validate(data)
            return cls._create_success(validated)
        except flext_core.c.EXC_ATTR_RUNTIME_VALIDATION as exc:
            return cls._failure(str(exc), exception=exc)

    @staticmethod
    def successful_result(
        obj: flext_core.p.SuccessCheckable | flext_core.t.JsonPayload,
    ) -> bool:
        """Determine whether an object is a successful result instance.

        Args:
            obj: The object candidate to inspect.

        Returns:
            bool: True if ``obj`` is a successful ``FlextResultBase`` instance.

        """
        return isinstance(obj, FlextResultBase) and obj.success

    @staticmethod
    def failed_result(
        obj: flext_core.p.SuccessCheckable | flext_core.t.JsonPayload,
    ) -> bool:
        """Determine whether an object is a failed result instance.

        Args:
            obj: The object candidate to inspect.

        Returns:
            bool: True if ``obj`` is a failed ``FlextResultBase`` instance.

        """
        return isinstance(obj, FlextResultBase) and not obj.success


__all__: list[str] = ["FlextResultConstruction"]
