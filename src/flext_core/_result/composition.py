"""Composition helpers for FlextResult.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from pydantic import ValidationError

from flext_core import c
from flext_core._result import FlextResultTransforms, copy_result, ok_result

if TYPE_CHECKING:
    from collections.abc import Callable, MutableSequence

    from flext_core import p, t


class FlextResultComposition[T](FlextResultTransforms[T]):
    """Sequence, resource, and decorator composition helpers.

    Combines many results (``accumulate_errors``, ``traverse``), wraps
    resource acquisition and cleanup around result-producing operations
    (``with_resource``), and guards plain functions via ``safe`` so runtime
    exceptions become ``r.Fail`` instead of raising.
    """

    @classmethod
    def accumulate_errors[ValueT](
        cls: type[Self],
        *results: p.Result[ValueT],
    ) -> p.Result[Sequence[ValueT]]:
        """Collect every payload, or fail joining all error messages.

        Args:
            *results: The results to combine.

        Returns:
            The resulting ``p.Result[Sequence[ValueT]]``.

        """
        from flext_core import r

        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(cls.require_error(result))
        if errors:
            return r[Sequence[ValueT]].fail("; ".join(errors))
        payload: Sequence[ValueT] = successes
        return ok_result(payload)

    @classmethod
    def traverse[V, U](
        cls: type[Self],
        items: t.SequenceOf[V],
        func: Callable[[V], p.Result[U]],
        *,
        fail_fast: bool = True,
    ) -> p.Result[Sequence[U]]:
        """Map a result-returning function over every item.

        With ``fail_fast`` the first failure (or catchable exception) returns
        immediately; otherwise every item runs and all error messages
        accumulate into one failure.

        Args:
            items: The input sequence mapped over.
            func: The result-returning function applied to each item.
            fail_fast: Whether to stop at the first failure.

        Returns:
            The resulting ``p.Result[Sequence[U]]``.

        """
        from flext_core import r

        if fail_fast:
            results: MutableSequence[U] = []
            for item in items:
                try:
                    result = func(item)
                except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                    return r[Sequence[U]].fail(str(exc), exception=exc)
                if result.failure:
                    return r[Sequence[U]].from_failure(result)
                results.append(result.value)
            payload: Sequence[U] = results
            return ok_result(payload)
        all_results: MutableSequence[p.Result[U]] = []
        for item in items:
            try:
                all_results.append(copy_result(func(item)))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                all_results.append(
                    r[U].fail(str(exc), exception=exc),
                )
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls: type[Self],
        factory: Callable[[], R],
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> p.Result[U]:
        """Run an operation over an acquired resource with optional cleanup.

        Any catchable exception raised while acquiring, operating, or cleaning
        up becomes a failure; a cleanup failure discards the operation result.

        Args:
            factory: The callable acquiring the resource.
            op: The result-returning operation over the resource.
            cleanup: The optional callable releasing the resource.

        Returns:
            The resulting ``p.Result[U]``.

        """
        from flext_core import r

        try:
            resource = factory()
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return r[U].fail(str(exc), exception=exc)
        result: p.Result[U]
        try:
            result = copy_result(op(resource))
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            result = r[U].fail(str(exc), exception=exc)
        if cleanup:
            try:
                cleanup(resource)
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return r[U].fail(str(exc), exception=exc)
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
        cls: type[Self],
        func: Callable[PFunc, U],
    ) -> Callable[PFunc, p.Result[U]]:
        """Wrap a callable so catchable runtime exceptions become failures.

        Args:
            func: The callable to guard.

        Returns:
            The resulting ``Callable[PFunc, p.Result[U]]``.

        """

        from flext_core import r

        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> p.Result[U]:
            try:
                return ok_result(func(*args, **kwargs))
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return r[U].fail(str(exc), exception=exc)

        return wrapper


__all__: list[str] = ["FlextResultComposition"]
