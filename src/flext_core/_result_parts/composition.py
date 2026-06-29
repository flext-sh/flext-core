"""Composition helpers for FlextResult."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, MutableSequence, Sequence
from typing import cast

from pydantic import ValidationError

from flext_core._constants.errors import FlextConstantsErrors as c
from flext_core._protocols.result import FlextProtocolsResult as p
from flext_core._typings.base import FlextTypingBase as t

from .construction import FlextResultConstructionMixin


class FlextResultCompositionMixin[T](FlextResultConstructionMixin[T], ABC):
    """Sequence, resource, and decorator composition helpers."""

    @classmethod
    def accumulate_errors[ValueT](
        cls,
        *results: p.Result[ValueT],
    ) -> p.Result[Sequence[ValueT]]:
        """Collect successes or all errors combined."""
        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(result.error or "Unknown error")
        if errors:
            # Type bridge: accumulated failures carry no payload value.
            result_class = cast(
                "type[FlextResultConstructionMixin[Sequence[ValueT]]]", cls
            )
            return result_class.fail("; ".join(errors))
        return cls.ok(successes)

    @classmethod
    def traverse[V, U](
        cls,
        items: t.SequenceOf[V],
        func: Callable[[V], p.Result[U]],
        *,
        fail_fast: bool = True,
    ) -> p.Result[Sequence[U]]:
        """Map sequence via func; fail_fast stops on first error."""
        if fail_fast:
            results: MutableSequence[U] = []
            for item in items:
                result = func(item)
                if result.failure:
                    # Type bridge: fail-fast returns the sequence result shape.
                    result_class = cast(
                        "type[FlextResultConstructionMixin[Sequence[U]]]", cls
                    )
                    return result_class.fail(
                        result.error or "Unknown error",
                        error_code=result.error_code,
                        error_data=result.error_data,
                        exception=result.exception,
                    )
                results.append(result.value)
            return cls.ok(results)
        all_results = [cls.from_result(func(item)) for item in items]
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls,
        factory: Callable[[], R],
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> p.Result[U]:
        """Manage resource lifecycle with automatic cleanup."""
        resource = factory()
        try:
            return cls.from_result(op(resource))
        finally:
            if cleanup:
                cleanup(resource)

    @staticmethod
    def _model_error_message(error: BaseException) -> str:
        if isinstance(error, ValidationError):
            return str(error.errors())
        errors_fn = getattr(error, "errors", None)
        if callable(errors_fn):
            return str(errors_fn())
        return str(error)

    @classmethod
    def safe[U, **PFunc](
        cls,
        func: Callable[PFunc, U],
    ) -> Callable[PFunc, p.Result[U]]:
        """Decorator: wrap function in FlextResult, catch exceptions."""

        def wrapper(
            *args: PFunc.args,
            **kwargs: PFunc.kwargs,
        ) -> p.Result[U]:
            try:
                return cls.ok(func(*args, **kwargs))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                # Type bridge: decorator failures match the wrapped return type.
                result_class = cast("type[FlextResultConstructionMixin[U]]", cls)
                return result_class.fail(str(exc), exception=exc)

        return wrapper


__all__: list[str] = ["FlextResultCompositionMixin"]
