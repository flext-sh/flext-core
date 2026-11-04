"""Sequence combinators and functional composition helpers for FlextResult.

Supplies operations to combine multiple results (``accumulate_errors``), map
result-producing functions across collections with fail-fast or accumulation semantics
(``traverse``), manage resource lifetimes safely (``with_resource``), and wrap
arbitrary callables to capture exceptions into railway failures (``safe``).

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-027 (Facade Single-Owner Law):
      ``flext/docs/architecture/adr/027-facade-single-owner-law.md``
    - :mod:`flext_core._result.transforms`: Monadic transformation operations.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Self

from flext_core import c, m
from flext_core._result import FlextResultTransforms

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core import p


class FlextResultComposition[T](FlextResultTransforms[T], ABC):
    """Sequence, resource management, and functional decorator composition helpers.

    Provides multi-result combination, collection traversal, transactional
    resource scopes, and exception-shielding decorators.

    Type Parameters:
        T: The payload type carried upon successful completion.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``

    """

    @classmethod
    def accumulate_errors[ValueT](
        cls: type[Self],
        *results: p.Result[ValueT],
    ) -> p.Result[Sequence[ValueT]]:
        """Collect success payloads into a sequence or retain failure context.

        If a single failure is encountered, its original error message, code,
        metadata, and causal exception are preserved. If multiple failures occur,
        all error messages are joined while preserving the first failure's context.

        Args:
            *results: Variable number of result instances to combine.

        Returns:
            p.Result[Sequence[ValueT]]: A success result containing all payloads if all
                succeeded, or a combined failure result if any failed.

        """
        successes: MutableSequence[ValueT] = []
        errors: MutableSequence[str] = []
        first_failure: p.Result[ValueT] | None = None
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                errors.append(cls.require_error(result))
                if first_failure is None:
                    first_failure = result
        if first_failure is not None:
            if len(errors) == 1:
                return cls._copy_failure(first_failure)
            return cls._failure(
                "; ".join(errors),
                error_code=first_failure.error_code,
                error_data=first_failure.error_data,
                exception=first_failure.exception,
            )
        payload: Sequence[ValueT] = successes
        return cls._create_success(payload)

    @classmethod
    def traverse[V, U](
        cls: type[Self],
        items: Sequence[V],
        func: Callable[[V], p.Result[U]],
        *,
        fail_fast: bool = True,
        exceptions: tuple[type[Exception], ...] = c.CATCHABLE_RUNTIME_EXCEPTIONS,
    ) -> p.Result[Sequence[U]]:
        """Map a result-returning callable across a sequence of items.

        Args:
            items: The input sequence to iterate over.
            func: A callable mapping each item to a result instance.
            fail_fast: Whether to abort immediately at the first encountered failure.
            exceptions: Tuple of exception types to convert into failures.

        Returns:
            p.Result[Sequence[U]]: A success result carrying the mapped sequence, or
                the earliest failure encountered (if fail_fast) / accumulated failures.

        """
        if fail_fast:
            results: MutableSequence[U] = []
            for item in items:
                try:
                    result = func(item)
                except exceptions as exc:
                    if isinstance(exc, c.EXC_CANCELLATION):
                        raise
                    return cls._failure(str(exc), exception=exc)
                if result.failure:
                    return cls._copy_failure(result)
                results.append(result.value)
            payload: Sequence[U] = results
            return cls._create_success(payload)
        all_results: MutableSequence[p.Result[U]] = []
        for item in items:
            try:
                all_results.append(cls.from_result(func(item)))
            except exceptions as exc:
                if isinstance(exc, c.EXC_CANCELLATION):
                    raise
                all_results.append(
                    cls._failure(str(exc), exception=exc),
                )
        return cls.accumulate_errors(*all_results)

    @classmethod
    def with_resource[R, U](
        cls: type[Self],
        factory: Callable[[], R],
        op: Callable[[R], p.Result[U]],
        cleanup: Callable[[R], None] | None = None,
    ) -> p.Result[U]:
        """Execute an operation over an acquired resource with guaranteed cleanup.

        Args:
            factory: Callable producing or acquiring the managed resource.
            op: Operation executing over the acquired resource and returning a Result.
            cleanup: Optional callable releasing the resource upon completion.

        Returns:
            p.Result[U]: The result of the operation, or a failure if acquisition,
                execution, or cleanup raises an exception.

        """
        try:
            resource = factory()
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return cls._failure(str(exc), exception=exc)
        result: p.Result[U]
        try:
            result = cls.from_result(op(resource))
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            result = cls._failure(str(exc), exception=exc)
        if cleanup:
            try:
                cleanup(resource)
            except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
                return cls._failure(str(exc), exception=exc)
        return result

    @staticmethod
    def _model_error_message(error: BaseException) -> str:
        if isinstance(error, m.ValidationError):
            return str(error.errors())
        errors_fn = getattr(error, "errors", None)
        if callable(errors_fn):
            return str(errors_fn())
        return str(error)

    @classmethod
    def safe[U, **PFunc](
        cls: type[Self],
        func: Callable[PFunc, U],
        *,
        exceptions: tuple[type[Exception], ...] = c.CATCHABLE_RUNTIME_EXCEPTIONS,
    ) -> Callable[PFunc, p.Result[U]]:
        """Wrap a function to intercept runtime exceptions and return Result instances.

        Args:
            func: The callable function to decorate and shield.
            exceptions: Tuple of catchable exception classes to convert into failures.

        Returns:
            Callable[PFunc, p.Result[U]]: Decorated callable returning ``p.Result[U]``.

        """

        def wrapper(*args: PFunc.args, **kwargs: PFunc.kwargs) -> p.Result[U]:
            try:
                return cls._create_success(func(*args, **kwargs))
            except exceptions as exc:
                if isinstance(exc, c.EXC_CANCELLATION):
                    raise
                return cls._failure(str(exc), exception=exc)

        return wrapper


__all__: list[str] = ["FlextResultComposition"]
