"""Composition helpers for FlextResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import ValidationError

from flext_core import c

from .transforms import FlextResultTransforms

if TYPE_CHECKING:
    from collections.abc import Callable, MutableSequence, Sequence

    from flext_core import p, t
    from flext_core import FlextResult as rt


class FlextResultComposition[T](FlextResultTransforms[T]):
    """Sequence, resource, and decorator composition helpers."""

    @classmethod
    def accumulate_errors[ValueT](
        cls: type[Self], *results: p.Result[ValueT]
    ) -> rt[Sequence[ValueT]]:
        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(cls.require_error(result))
        if errors:
            return cls.fail("; ".join(errors))
        return cls.ok(successes)

    @classmethod
    def traverse[V, U](
        cls: type[Self],
        items: t.SequenceOf[V],
        func: Callable[[V], p.Result[U]],
        *,
        fail_fast: bool = True,
    ) -> rt[Sequence[U]]:
        if fail_fast:
            results: MutableSequence[U] = []
            for item in items:
                try:
                    result = func(item)
                except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                    return cls.fail(str(exc), exception=exc)
                if result.failure:
                    return cls.from_failure(result)
                results.append(result.value)
            return cls.ok(results)
        all_results: MutableSequence[rt[U]] = []
        for item in items:
            try:
                all_results.append(cls.from_result(func(item)))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                all_results.append(cls.fail(str(exc), exception=exc))
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls: type[Self],
        factory: Callable[[], R],
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> rt[U]:
        try:
            resource = factory()
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return cls.fail(str(exc), exception=exc)
        result: rt[U]
        try:
            result = cls.from_result(op(resource))
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            result = cls.fail(str(exc), exception=exc)
        if cleanup:
            try:
                cleanup(resource)
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return cls.fail(str(exc), exception=exc)
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
    def safe[U, **PFunc](
        cls: type[Self], func: Callable[PFunc, U]
    ) -> Callable[PFunc, rt[U]]:
        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> rt[U]:
            try:
                return cls.ok(func(*args, **kwargs))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return cls.fail(str(exc), exception=exc)

        return wrapper


__all__: list[str] = ["FlextResultComposition"]
