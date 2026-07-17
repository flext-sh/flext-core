"""Composition helpers for FlextResult."""

from __future__ import annotations

from abc import ABC
from typing import cast, TYPE_CHECKING

from pydantic import ValidationError

from flext_core._constants.errors import FlextConstantsErrors as c

from .construction import FlextResultConstructionMixin


if TYPE_CHECKING:
    from flext_core._protocols.result import FlextProtocolsResult as p
    from collections.abc import Callable, MutableSequence, Sequence
    from flext_core._typings.base import FlextTypingBase as t


class FlextResultCompositionMixin[T](FlextResultConstructionMixin[T], ABC):
    """Sequence, resource, and decorator composition helpers."""

    @classmethod
    def accumulate_errors[ValueT](
        cls, *results: p.Result[ValueT]
    ) -> p.Result[Sequence[ValueT]]:
        """Collect successes or all errors combined."""
        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(cls.require_error(result))
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
            result_class = cast("type[FlextResultConstructionMixin[Sequence[U]]]", cls)
            for item in items:
                try:
                    result = func(item)
                except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                    return result_class.fail(str(exc), exception=exc)
                if result.failure:
                    return result_class.from_failure(result)
                results.append(result.value)
            return cls.ok(results)
        item_result_class = cast("type[FlextResultConstructionMixin[U]]", cls)
        all_results: MutableSequence[p.Result[U]] = []
        for item in items:
            try:
                all_results.append(cls.from_result(func(item)))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                all_results.append(item_result_class.fail(str(exc), exception=exc))
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls,
        factory: Callable[[], R],
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> p.Result[U]:
        """Manage resource lifecycle with automatic cleanup."""
        result_class = cast("type[FlextResultConstructionMixin[U]]", cls)
        try:
            resource = factory()
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return result_class.fail(str(exc), exception=exc)
        result: p.Result[U]
        try:
            result = cls.from_result(op(resource))
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            result = result_class.fail(str(exc), exception=exc)
        if cleanup:
            try:
                cleanup(resource)
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return result_class.fail(str(exc), exception=exc)
        return result

    @staticmethod
    def _model_error_message(error: BaseException) -> str:
        if isinstance(error, ValidationError):
            return str(error.errors())
        errors_fn = getattr(error, "errors", None)
        if callable(errors_fn):
            return str(errors_fn())
        return str(error)

    @classmethod
    def safe[U, **PFunc](cls, func: Callable[PFunc, U]) -> Callable[PFunc, p.Result[U]]:
        """Wrap function in FlextResult, catching exceptions."""

        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> p.Result[U]:
            try:
                return cls.ok(func(*args, **kwargs))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                # Type bridge: decorator failures match the wrapped return type.
                result_class = cast("type[FlextResultConstructionMixin[U]]", cls)
                return result_class.fail(str(exc), exception=exc)

        return wrapper


__all__: list[str] = ["FlextResultCompositionMixin"]
